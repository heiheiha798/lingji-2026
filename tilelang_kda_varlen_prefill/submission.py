from __future__ import annotations

from typing import Any

import torch

from reference import kda_reference


class Submission:
    """可运行的 PyTorch 起始版本。正式提交时请改为 TileLang 实现。"""

    def build(self, spec: Any) -> Any:
        return spec

    @torch.no_grad()
    def run(
        self,
        state: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        workspace: torch.Tensor,
        out: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        del state, workspace
        ref_out, ref_final = kda_reference(
            q,
            k,
            v,
            g_raw,
            beta_raw,
            a_log,
            dt_bias,
            initial_state,
            cu_seqlens,
        )
        out.copy_(ref_out)
        final_state.copy_(ref_final)
