from __future__ import annotations

import math

import torch


HEAD_DIM = 128
LOWER_BOUND = -5.0
NORM_EPS = 1e-6


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> list[int]:
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must have shape [T, H, {HEAD_DIM}]")
    if any(t.shape != q.shape for t in (k, v, g_raw)):
        raise ValueError("k, v and g_raw must have the same shape as q")
    total_tokens, num_heads, _ = q.shape
    if beta_raw.shape != (total_tokens, num_heads):
        raise ValueError("beta_raw must have shape [T, H]")
    if a_log.shape != (num_heads,):
        raise ValueError("a_log must have shape [H]")
    if dt_bias.shape != (num_heads, HEAD_DIM):
        raise ValueError("dt_bias must have shape [H, 128]")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (
        num_heads, HEAD_DIM, HEAD_DIM
    ):
        raise ValueError("initial_state must have shape [B, H, 128, 128]")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
    tensors = (
        q, k, v, g_raw, beta_raw, a_log, dt_bias,
        initial_state, cu_seqlens,
    )
    expected_dtypes = {
        "q": (q, torch.bfloat16),
        "k": (k, torch.bfloat16),
        "v": (v, torch.bfloat16),
        "g_raw": (g_raw, torch.bfloat16),
        "beta_raw": (beta_raw, torch.bfloat16),
        "a_log": (a_log, torch.float32),
        "dt_bias": (dt_bias, torch.float32),
        "initial_state": (initial_state, torch.bfloat16),
    }
    for name, (tensor, dtype) in expected_dtypes.items():
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must use {dtype}")
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")

    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if len(cu) != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens length must equal B + 1")
    if not cu or cu[0] != 0 or cu[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at 0 and end at T")
    if any(b <= a for a, b in zip(cu[:-1], cu[1:])):
        raise ValueError("all sequences must be non-empty")

    return cu


@torch.no_grad()
def kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 token-recurrent reference.

    This implementation is intended for small and medium correctness cases.
    The official judge may use a separately validated accelerated oracle for
    full-size performance inputs. Both implement the same equations.
    """
    cu = _validate_inputs(
        q, k, v, g_raw, beta_raw, a_log, dt_bias,
        initial_state, cu_seqlens,
    )
    total_tokens, num_heads, _ = q.shape
    num_sequences = len(cu) - 1

    out_fp32 = torch.full(
        (total_tokens, num_heads, HEAD_DIM),
        float("nan"),
        dtype=torch.float32,
        device=q.device,
    )
    final_fp32 = torch.full(
        (num_sequences, num_heads, HEAD_DIM, HEAD_DIM),
        float("nan"),
        dtype=torch.float32,
        device=q.device,
    )

    a_scale = torch.exp(a_log.float()).view(num_heads, 1)
    dt_bias_fp32 = dt_bias.float()
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for seq_id in range(num_sequences):
        start, end = cu[seq_id], cu[seq_id + 1]
        state = initial_state[seq_id].float().clone()

        for absolute_pos in range(start, end):
            q_t = q[absolute_pos].float()
            k_t = k[absolute_pos].float()
            v_t = v[absolute_pos].float()

            q_t = q_t * torch.rsqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True) + NORM_EPS)
            k_t = k_t * torch.rsqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True) + NORM_EPS)

            gate_x = a_scale * (g_raw[absolute_pos].float() + dt_bias_fp32)
            log_a = LOWER_BOUND * torch.sigmoid(gate_x)
            a = torch.exp(log_a)
            beta = torch.sigmoid(beta_raw[absolute_pos].float())

            state_decay = state * a.unsqueeze(-1)
            old_value = torch.bmm(k_t.unsqueeze(1), state_decay).squeeze(1)
            residual = v_t - old_value
            state = (
                state_decay
                + beta[:, None, None] * k_t[:, :, None] * residual[:, None, :]
            )

            out_fp32[absolute_pos] = (
                scale * torch.bmm(q_t.unsqueeze(1), state).squeeze(1)
            )

        final_fp32[seq_id].copy_(state)

    return (
        out_fp32.to(torch.bfloat16),
        final_fp32.to(torch.bfloat16),
    )
