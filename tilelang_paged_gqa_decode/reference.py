from __future__ import annotations

import math

import torch


HEAD_DIM = 128


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> list[int]:
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must have shape [B, Hq, {HEAD_DIM}]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have the same four-dimensional shape")
    batch, num_q_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, cache_dim = k_cache.shape
    if cache_dim != head_dim:
        raise ValueError("KV head dimension must equal query head dimension")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have shape [B, max_pages]")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must use int32")
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must have shape [B] and dtype int32")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("q, k_cache and v_cache must use bfloat16")
    tensors = (q, k_cache, v_cache, block_table, seq_lens)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if page_size not in (16, 32):
        raise ValueError("page_size must be 16 or 32")
    if num_q_heads // num_kv_heads not in (4, 8):
        raise ValueError("GQA group size must be 4 or 8")

    lengths = [int(x) for x in seq_lens.detach().cpu().tolist()]
    table = block_table.detach().cpu()
    for batch_index, seq_len in enumerate(lengths):
        if seq_len <= 0:
            raise ValueError("all sequence lengths must be positive")
        page_count = math.ceil(seq_len / page_size)
        if page_count > block_table.shape[1]:
            raise ValueError("block_table does not have enough entries")
        pages = table[batch_index, :page_count]
        if torch.any(pages < 0) or torch.any(pages >= num_pages):
            raise ValueError("valid block_table entries are out of range")
    return lengths


@torch.no_grad()
def paged_gqa_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """FP32 PyTorch reference for single-token paged GQA decode."""
    lengths = _validate_inputs(q, k_cache, v_cache, block_table, seq_lens)
    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.full_like(q, float("nan"))
    table = block_table.detach().cpu()

    for batch_index, seq_len in enumerate(lengths):
        page_count = math.ceil(seq_len / page_size)
        physical_pages = table[batch_index, :page_count].to(torch.long).to(q.device)

        k_logical = (
            k_cache.index_select(0, physical_pages)
            .reshape(-1, num_kv_heads, head_dim)[:seq_len]
        )
        v_logical = (
            v_cache.index_select(0, physical_pages)
            .reshape(-1, num_kv_heads, head_dim)[:seq_len]
        )

        q_grouped = q[batch_index].float().view(
            num_kv_heads, group_size, head_dim
        )
        k_hsd = k_logical.float().transpose(0, 1)
        v_hsd = v_logical.float().transpose(0, 1)
        scores = torch.einsum("hgd,hsd->hgs", q_grouped, k_hsd) * scale
        probabilities = torch.softmax(scores, dim=-1)
        out_grouped = torch.einsum("hgs,hsd->hgd", probabilities, v_hsd)
        output[batch_index].copy_(
            out_grouped.reshape(num_q_heads, head_dim).to(q.dtype)
        )
    return output
