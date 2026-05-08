from __future__ import annotations

import os
import math
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex
import tiktoken
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm

GPT2_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    # raise NotImplementedError
    if weights.shape != (d_out, d_in):
        raise ValueError(f"Expected weights shape {(d_out, d_in)}, got {tuple(weights.shape)}")
    if in_features.shape[-1] != d_in:
        raise ValueError(f"Expected in_features.shape[-1] == {d_in}, got {in_features.shape[-1]}")
    # torch.nn.Linear computes x @ W^T (+ b). Here we only apply the weight.
    return torch.matmul(in_features, weights.transpose(-1, -2)) # shape : (..., d_out)
 


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    if weights.shape != (vocab_size, d_model):
        raise ValueError(f"Expected weights shape {(vocab_size, d_model)}, got {tuple(weights.shape)}")
    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # SwiGLU(x) = W2 · ( SiLU(x @ W1^T) ⊙ (x @ W3^T) )
    gate = torch.nn.functional.silu(in_features @ w1_weight.T)
    up   = in_features @ w3_weight.T
    return (gate * up) @ w2_weight.T




def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    d_k = Q.shape[-1]
    scale = 1.0 / (d_k**0.5)

    # (..., queries, d_k) @ (..., d_k, keys) -> (..., queries, keys)
    attention_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.shape != attention_logits.shape:
            raise ValueError(
                f"mask shape must match attention logits shape {tuple(attention_logits.shape)}, "
                f"got {tuple(mask.shape)}"
            )
        # In a boolean attention mask, True means "keep", False means "mask out".
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))

    # 修改意见： 这里要用 run_softmax 函数，而不是 torch.softmax
    attention_weights = torch.softmax(attention_logits, dim=-1)
    # (..., queries, keys) @ (..., keys, d_v) -> (..., queries, d_v)
    return torch.matmul(attention_weights, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    head_dim = d_model // num_heads
    sequence_length = in_features.shape[-2]
    leading_dims = in_features.shape[:-2]

    # Project once for all heads, then reshape into per-head representations.
    q = torch.matmul(in_features, q_proj_weight.transpose(-1, -2))
    k = torch.matmul(in_features, k_proj_weight.transpose(-1, -2))
    v = torch.matmul(in_features, v_proj_weight.transpose(-1, -2))

    q = q.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)
    k = k.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)
    v = v.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)

    # Causal self-attention: position i can only attend to positions <= i.
    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=in_features.device)
    )
    causal_mask = causal_mask.view(*([1] * (q.ndim - 2)), sequence_length, sequence_length)
    causal_mask = causal_mask.expand(*q.shape[:-1], sequence_length)

    attention_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)

    # Merge heads back, then apply the output projection.
    merged_heads = attention_output.transpose(-3, -2).reshape(*leading_dims, sequence_length, d_model)
    return torch.matmul(merged_heads, o_proj_weight.transpose(-1, -2))


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if q_proj_weight.shape != (d_model, d_model):
        raise ValueError(f"Expected q_proj_weight shape {(d_model, d_model)}, got {tuple(q_proj_weight.shape)}")
    if k_proj_weight.shape != (d_model, d_model):
        raise ValueError(f"Expected k_proj_weight shape {(d_model, d_model)}, got {tuple(k_proj_weight.shape)}")
    if v_proj_weight.shape != (d_model, d_model):
        raise ValueError(f"Expected v_proj_weight shape {(d_model, d_model)}, got {tuple(v_proj_weight.shape)}")
    if o_proj_weight.shape != (d_model, d_model):
        raise ValueError(f"Expected o_proj_weight shape {(d_model, d_model)}, got {tuple(o_proj_weight.shape)}")
    if in_features.shape[-1] != d_model:
        raise ValueError(f"Expected in_features.shape[-1] == {d_model}, got {in_features.shape[-1]}")

    head_dim = d_model // num_heads
    sequence_length = in_features.shape[-2]
    leading_dims = in_features.shape[:-2]

    q = torch.matmul(in_features, q_proj_weight.transpose(-1, -2))
    k = torch.matmul(in_features, k_proj_weight.transpose(-1, -2))
    v = torch.matmul(in_features, v_proj_weight.transpose(-1, -2))

    q = q.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)
    k = k.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)
    v = v.reshape(*leading_dims, sequence_length, num_heads, head_dim).transpose(-3, -2)

    if token_positions is None:
        token_positions = torch.arange(sequence_length, device=in_features.device)

    q = run_rope(
        d_k=head_dim,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=q,
        token_positions=token_positions,
    )
    k = run_rope(
        d_k=head_dim,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=k,
        token_positions=token_positions,
    )

    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=in_features.device)
    )
    causal_mask = causal_mask.view(*([1] * (q.ndim - 2)), sequence_length, sequence_length)
    causal_mask = causal_mask.expand(*q.shape[:-1], sequence_length)

    attention_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)
    merged_heads = attention_output.transpose(-3, -2).reshape(*leading_dims, sequence_length, d_model)
    return torch.matmul(merged_heads, o_proj_weight.transpose(-1, -2))


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    if d_k % 2 != 0:
        raise ValueError(f"RoPE requires even d_k, got {d_k}")
    if in_query_or_key.shape[-1] != d_k:
        raise ValueError(
            f"Expected in_query_or_key.shape[-1] == {d_k}, got {in_query_or_key.shape[-1]}"
        )

    if token_positions.dtype != torch.long:
        token_positions = token_positions.to(dtype=torch.long)

    if token_positions.numel() > 0 and torch.any(token_positions < 0):
        raise ValueError("token_positions must be non-negative")
    if token_positions.numel() > 0 and torch.any(token_positions >= max_seq_len):
        raise ValueError("token_positions contain values that exceed max_seq_len")

    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    half_dim = d_k // 2
    dim_indices = torch.arange(0, half_dim, device=device, dtype=torch.float32)
    inv_freq = theta ** (-2.0 * dim_indices / d_k)

    angles = token_positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    required_keys = {
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.output_proj.weight",
        "ln1.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
        "ln2.weight",
    }
    missing_keys = required_keys.difference(weights.keys())
    if missing_keys:
        raise KeyError(f"Missing required transformer block keys: {sorted(missing_keys)}")

    ln1_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln1.weight"],
        in_features=in_features,
    )
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=ln1_out,
        token_positions=None,
    )
    residual_after_attn = in_features + attn_out

    ln2_out = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln2.weight"],
        in_features=residual_after_attn,
    )
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=ln2_out,
    )
    return residual_after_attn + ffn_out


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    sequence_length = in_indices.shape[-1]
    if sequence_length > context_length:
        raise ValueError(
            f"Input sequence length ({sequence_length}) exceeds context length ({context_length})"
        )

    token_embeddings = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    hidden_states = token_embeddings
    for layer_idx in range(num_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {
            key.replace(layer_prefix, ""): value
            for key, value in weights.items()
            if key.startswith(layer_prefix)
        }
        hidden_states = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=hidden_states,
        )

    normalized = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=hidden_states,
    )

    return run_linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=normalized,
    )


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    if weights.shape != (d_model,):
        raise ValueError(f"Expected weights shape {(d_model,)}, got {tuple(weights.shape)}")
    if in_features.shape[-1] != d_model:
        raise ValueError(f"Expected in_features.shape[-1] == {d_model}, got {in_features.shape[-1]}")

    rms = torch.sqrt(torch.mean(in_features * in_features, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    return normalized * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data = torch.as_tensor(dataset, dtype=torch.long)
    if data.ndim != 1:
        raise ValueError(f"dataset must be a 1D array, got shape {tuple(data.shape)}")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data.shape[0] <= context_length:
        raise ValueError("dataset is too short for the requested context_length")

    max_start = data.shape[0] - context_length
    start_indices = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(context_length)
    x_positions = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    y_positions = x_positions + 1

    x = data[x_positions].to(device)
    y = data[y_positions].to(device)
    return x, y


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_values = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_values
    exp_shifted = torch.exp(shifted)
    partition = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / partition


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape (batch_size, vocab_size), got {tuple(inputs.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (batch_size,), got {tuple(targets.shape)}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"inputs and targets must share batch size, got {inputs.shape[0]} and {targets.shape[0]}"
        )

    # 修改意见： 这里要用 run_softmax 函数，而不是 torch.logsumexp
    logsumexp = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = logsumexp - target_logits
    return nll.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate

    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])


class _BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.special_tokens = special_tokens or []

        token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        self.special_token_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in token_to_id:
                raise ValueError(f"Special token {token!r} not found in provided vocab.")
            self.special_token_to_id[token] = token_to_id[token_bytes]

        special_token_ids = set(self.special_token_to_id.values())
        mergeable_ranks = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
            if token_id not in special_token_ids
        }

        self.encoding = tiktoken.Encoding(
            name="cs336_custom_bpe",
            pat_str=GPT2_PRETOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_token_to_id,
        )

    def encode(self, text: str) -> list[int]:
        if self.special_token_to_id:
            return self.encoding.encode(text, allowed_special=set(self.special_token_to_id))
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return _BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    special_tokens = special_tokens or []
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must be unique")

    # Remove special token spans from merge training so they never get merged into other tokens.
    train_segments: list[str]
    if special_tokens:
        escaped = sorted((regex.escape(tok) for tok in special_tokens), key=len, reverse=True)
        split_pattern = "|".join(escaped)
        pieces = regex.split(f"({split_pattern})", text)
        train_segments = [piece for piece in pieces if piece and piece not in set(special_tokens)]
    else:
        train_segments = [text]

    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    for segment in tqdm(train_segments, desc="tokenizing", unit="segment"):
        for pretoken in regex.findall(GPT2_PRETOKEN_PATTERN, segment):
            token_bytes = pretoken.encode("utf-8")
            if not token_bytes:
                continue
            word = tuple(bytes([b]) for b in token_bytes)
            word_freqs[word] += 1

    vocab: dict[int, bytes] = {}
    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for byte_value in range(256):
        vocab[next_id] = bytes([byte_value])
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    total_merge_steps = max(vocab_size - next_id, 0)
    progress = tqdm(total=total_merge_steps, desc="bpe_train", unit="merge")

    try:
        while next_id < vocab_size:
            pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq

            if not pair_counts:
                break

            # Tiebreak: pick the lexicographically greatest pair among max-frequency pairs.
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            merged_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = merged_token
            next_id += 1
            merged_token_str = merged_token.decode("utf-8", errors="replace")
            print(f"[merge {len(merges)}] merged_token={merged_token_str!r}")
            progress.update(1)

            updated_word_freqs: Counter[tuple[bytes, ...]] = Counter()
            first, second = best_pair
            for word, freq in word_freqs.items():
                merged_word: list[bytes] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        merged_word.append(merged_token)
                        i += 2
                    else:
                        merged_word.append(word[i])
                        i += 1
                updated_word_freqs[tuple(merged_word)] += freq
            word_freqs = updated_word_freqs
    finally:
        progress.close()

    return vocab, merges
