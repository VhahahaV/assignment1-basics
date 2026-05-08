from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from tests.adapters import (
    get_adamw_cls,
    get_tokenizer,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
    run_transformer_lm,
)


class AdapterTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = nn.ModuleList(
            [_AdapterTransformerBlockParams(d_model=d_model, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.ln_final_weight = nn.Parameter(torch.ones(d_model))
        self.lm_head_weight = nn.Parameter(torch.empty(vocab_size, d_model))

        self._reset_parameters(init_std=init_std)

    def _reset_parameters(self, *, init_std: float) -> None:
        nn.init.normal_(self.token_embeddings, mean=0.0, std=init_std)
        nn.init.normal_(self.lm_head_weight, mean=0.0, std=init_std)
        nn.init.ones_(self.ln_final_weight)
        for layer in self.layers:
            layer.reset_parameters(init_std=init_std)

    def _weights_dict(self) -> dict[str, Tensor]:
        weights: dict[str, Tensor] = {
            "token_embeddings.weight": self.token_embeddings,
            "ln_final.weight": self.ln_final_weight,
            "lm_head.weight": self.lm_head_weight,
        }
        for idx, layer in enumerate(self.layers):
            prefix = f"layers.{idx}."
            weights[f"{prefix}attn.q_proj.weight"] = layer.q_proj_weight
            weights[f"{prefix}attn.k_proj.weight"] = layer.k_proj_weight
            weights[f"{prefix}attn.v_proj.weight"] = layer.v_proj_weight
            weights[f"{prefix}attn.output_proj.weight"] = layer.output_proj_weight
            weights[f"{prefix}ln1.weight"] = layer.ln1_weight
            weights[f"{prefix}ffn.w1.weight"] = layer.ffn_w1_weight
            weights[f"{prefix}ffn.w2.weight"] = layer.ffn_w2_weight
            weights[f"{prefix}ffn.w3.weight"] = layer.ffn_w3_weight
            weights[f"{prefix}ln2.weight"] = layer.ln2_weight
        return weights

    def forward(self, token_ids: torch.LongTensor) -> Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self._weights_dict(),
            in_indices=token_ids,
        )


class _AdapterTransformerBlockParams(nn.Module):
    def __init__(self, *, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.output_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ffn_w1_weight = nn.Parameter(torch.empty(d_ff, d_model))
        self.ffn_w2_weight = nn.Parameter(torch.empty(d_model, d_ff))
        self.ffn_w3_weight = nn.Parameter(torch.empty(d_ff, d_model))
        self.ln2_weight = nn.Parameter(torch.ones(d_model))

    def reset_parameters(self, *, init_std: float) -> None:
        nn.init.normal_(self.q_proj_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.k_proj_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.v_proj_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.output_proj_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.ffn_w1_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.ffn_w2_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.ffn_w3_weight, mean=0.0, std=init_std)
        nn.init.ones_(self.ln1_weight)
        nn.init.ones_(self.ln2_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny Transformer LM on TinyStories with adapters-only interfaces."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.downscaled-1pct.txt"),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=2000)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("artifacts/tinystories_transformer.ckpt"))
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_tokenizer(tokenizer_dir: Path):
    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer artifacts not found in {tokenizer_dir}")
    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)
    with merges_path.open("rb") as f:
        merges = pickle.load(f)
    return get_tokenizer(vocab=vocab, merges=merges, special_tokens=None)


def _encode_dataset(data_path: Path, tokenizer) -> np.ndarray:
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    if len(token_ids) < 2:
        raise ValueError("Encoded dataset is too short.")
    return np.asarray(token_ids, dtype=np.int64)


@torch.no_grad()
def _estimate_loss(
    *,
    model: AdapterTransformerLM,
    dataset: np.ndarray,
    eval_steps: int,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = run_cross_entropy(
            inputs=logits.reshape(-1, logits.shape[-1]),
            targets=y.reshape(-1),
        )
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = _load_tokenizer(args.tokenizer_dir)
    dataset = _encode_dataset(args.data_path, tokenizer)
    vocab_size = len(tokenizer.vocab)

    if args.context_length + 1 >= dataset.shape[0]:
        raise ValueError("context_length is too large for the encoded dataset.")

    model = AdapterTransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    adamw_cls = get_adamw_cls()
    optimizer = adamw_cls(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume and args.checkpoint_path.exists():
        start_iter = run_load_checkpoint(args.checkpoint_path, model, optimizer) + 1
        print(f"resumed_from_iter={start_iter}")

    print("== TinyStories Transformer Training ==")
    print(f"data_path={args.data_path}")
    print(f"tokenizer_dir={args.tokenizer_dir}")
    print(f"dataset_tokens={dataset.shape[0]}")
    print(f"vocab_size={vocab_size}")
    print(f"device={args.device}")
    print(
        f"config: B={args.batch_size}, T={args.context_length}, d_model={args.d_model}, "
        f"layers={args.num_layers}, heads={args.num_heads}, d_ff={args.d_ff}"
    )
    print(f"max_iters={args.max_iters}, start_iter={start_iter}")

    for it in range(start_iter, args.max_iters):
        lr = run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = run_get_batch(
            dataset=dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = run_cross_entropy(
            inputs=logits.reshape(-1, logits.shape[-1]),
            targets=y.reshape(-1),
        )
        loss.backward()
        run_gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)
        optimizer.step()

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            train_loss = float(loss.item())
            eval_loss = _estimate_loss(
                model=model,
                dataset=dataset,
                eval_steps=args.eval_steps,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )
            print(
                f"iter={it:06d} lr={lr:.6e} train_loss={train_loss:.4f} "
                f"eval_loss={eval_loss:.4f}"
            )

        if args.save_interval > 0 and (it + 1) % args.save_interval == 0:
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            run_save_checkpoint(model, optimizer, it, args.checkpoint_path)
            print(f"checkpoint_saved={args.checkpoint_path} at_iter={it}")

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_save_checkpoint(model, optimizer, args.max_iters - 1, args.checkpoint_path)
    print(f"final_checkpoint={args.checkpoint_path}")


if __name__ == "__main__":
    main()
