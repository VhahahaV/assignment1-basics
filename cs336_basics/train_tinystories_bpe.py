from __future__ import annotations

import argparse
import cProfile
import pickle
import pstats
import resource
import threading
import time
from pathlib import Path

import psutil
from tests.adapters import get_tokenizer, run_train_bpe
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on TinyStories and report runtime metrics."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.downscaled-1pct.txt"),
        help="Path to TinyStories training text file.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Total tokenizer vocabulary size (including special tokens).",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to reserve (can be provided multiple times).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
        help="Directory to save trained vocab/merges as pickle files.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving trained vocab/merges to disk.",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default="Once upon a time, a tiny dragon learned to read.",
        help="Sample text for a quick encode/decode sanity check.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=2.0,
        help="How often to refresh live training progress output.",
    )
    parser.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Disable live progress updates during training.",
    )
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=12,
        help="Number of top cumulative-time functions to show in the profiler summary.",
    )
    return parser.parse_args()


def _read_text_stats(input_path: Path) -> tuple[int, int]:
    text = input_path.read_text(encoding="utf-8")
    return len(text), len(text.encode("utf-8"))


def _run_with_live_progress(
    *,
    interval_seconds: float,
    enabled: bool,
    train_fn,
):
    """Run training function while periodically updating a live progress indicator."""
    if not enabled:
        return train_fn()

    stop_event = threading.Event()
    start = time.perf_counter()

    def _progress_worker() -> None:
        with tqdm(
            total=None,
            desc="training_bpe",
            unit="tick",
            dynamic_ncols=True,
            mininterval=0.0,
        ) as pbar:
            while not stop_event.wait(interval_seconds):
                elapsed = time.perf_counter() - start
                pbar.update(1)
                pbar.set_postfix_str(f"elapsed={elapsed:.1f}s", refresh=True)

    thread = threading.Thread(target=_progress_worker, daemon=True)
    thread.start()
    try:
        return train_fn()
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, interval_seconds + 1.0))


def _safe_decode_token(token_bytes: bytes) -> str:
    return token_bytes.decode("utf-8", errors="replace")


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    special_tokens = list(dict.fromkeys(args.special_token or ["<|endoftext|>"]))
    chars_count, bytes_count = _read_text_stats(args.input_path)

    print("== TinyStories BPE Training ==")
    print(f"input_path      : {args.input_path}")
    print(f"vocab_size      : {args.vocab_size}")
    print(f"special_tokens  : {special_tokens}")
    print(f"input_chars     : {chars_count:,}")
    print(f"input_bytes     : {bytes_count:,} ({bytes_count / (1024 * 1024):.2f} MiB)")
    expected_merges = max(args.vocab_size - (256 + len(special_tokens)), 0)
    print(f"expected_merges : {expected_merges:,}")
    print(f"live_progress   : {not args.no_live_progress} (interval={args.progress_interval_seconds:.1f}s)")
    print(f"save_artifacts  : {not args.no_save} (dir={args.save_dir})")
    print(f"profile_top_k   : {max(args.profile_top_k, 1)}")

    profiler = cProfile.Profile()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    train_core = lambda: run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    vocab, merges = _run_with_live_progress(
        interval_seconds=max(args.progress_interval_seconds, 0.2),
        enabled=not args.no_live_progress,
        train_fn=lambda: profiler.runcall(train_core),
    )
    elapsed = time.perf_counter() - start
    rss_after = process.memory_info().rss
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = peak_rss_kib / 1024

    longest_token_id, longest_token_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    longest_token_len = len(longest_token_bytes)
    longest_token_preview = _safe_decode_token(longest_token_bytes[:80])
    if len(longest_token_bytes) > 80:
        longest_token_preview += "..."

    print("\n== Training Result ==")
    print(f"elapsed_seconds : {elapsed:.3f}")
    print(f"throughput_mib  : {(bytes_count / (1024 * 1024)) / elapsed:.2f} MiB/s")
    print(f"throughput_char : {chars_count / elapsed:,.0f} chars/s")
    print(f"rss_before_mib  : {rss_before / (1024 * 1024):.2f}")
    print(f"rss_after_mib   : {rss_after / (1024 * 1024):.2f}")
    print(f"rss_delta_mib   : {(rss_after - rss_before) / (1024 * 1024):.2f}")
    print(f"peak_rss_mib    : {peak_rss_mib:.2f}")
    print(f"final_vocab_size: {len(vocab):,}")
    print(f"num_merges      : {len(merges):,}")
    print(f"first_5_merges  : {merges[:5]}")
    print(f"longest_token_id: {longest_token_id}")
    print(f"longest_token_len_bytes: {longest_token_len}")
    print(f"longest_token_preview: {longest_token_preview!r}")

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    top_k = max(args.profile_top_k, 1)
    top_entries = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:top_k]
    hottest_fn = None
    hottest_cum = -1.0
    for (filename, _line, func_name), (_cc, _nc, tt, ct, _callers) in top_entries:
        if func_name in {"<module>"} or filename.endswith("threading.py") or filename == "~":
            continue
        if ct > hottest_cum:
            hottest_cum = ct
            hottest_fn = f"{Path(filename).name}:{func_name}"

    print("\n== Profiling (Top by Cumulative Time) ==")
    stats.print_stats(top_k)
    if hottest_fn is not None:
        print(f"profiling_hotspot: {hottest_fn} (cumtime={hottest_cum:.3f}s)")

    tokenizer = get_tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    encoded = tokenizer.encode(args.sample_text)
    decoded = tokenizer.decode(encoded)

    print("\n== Quick Sanity Check ==")
    print(f"sample_text     : {args.sample_text!r}")
    print(f"num_tokens      : {len(encoded)}")
    print(f"first_20_tokens : {encoded[:20]}")
    print(f"decoded_equal   : {decoded == args.sample_text}")

    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = args.save_dir / "vocab.pkl"
        merges_path = args.save_dir / "merges.pkl"
        with vocab_path.open("wb") as f:
            pickle.dump(vocab, f)
        with merges_path.open("wb") as f:
            pickle.dump(merges, f)
        print("\n== Saved Artifacts ==")
        print(f"vocab_pickle    : {vocab_path}")
        print(f"merges_pickle   : {merges_path}")


if __name__ == "__main__":
    main()
