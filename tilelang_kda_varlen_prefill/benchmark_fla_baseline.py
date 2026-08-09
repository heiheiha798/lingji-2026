from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch


FLA_SOURCE_DIR = Path("/data/home/tianjianyang/code/tmp/flash-linear-attention")
if not (FLA_SOURCE_DIR / "fla" / "ops" / "kda").is_dir():
    raise RuntimeError(f"FLA KDA source directory does not exist: {FLA_SOURCE_DIR}")

# Pin the comparison to FLA's Triton implementation. Optional FlashKDA and
# TileLang backend dispatch must not change the measured implementation.
os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
os.environ["FLA_CACHE_MODE"] = "disabled"
sys.path.insert(0, str(FLA_SOURCE_DIR))

import benchmark  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402


class FLASubmission:
    def build(self, spec):
        return spec

    @torch.inference_mode()
    def run(
        self,
        state,
        q,
        k,
        v,
        g_raw,
        beta_raw,
        a_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        workspace,
        out,
        final_state,
    ) -> None:
        fla_out, fla_final_state = chunk_kda(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g_raw.unsqueeze(0),
            beta=beta_raw.unsqueeze(0),
            scale=1.0 / math.sqrt(128),
            initial_state=initial_state.float(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            dt_bias=dt_bias,
            chunk_size=64,
        )
        out.copy_(fla_out.squeeze(0))
        final_state.copy_(fla_final_state)


benchmark._load_submission = FLASubmission


if __name__ == "__main__":
    benchmark.main()
