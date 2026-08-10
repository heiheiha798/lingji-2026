from __future__ import annotations

from typing import Any

import torch

from reference import paged_gqa_reference


class Submission:
    """可运行的 PyTorch 起始版本。正式提交时请改为 TileLang 实现。"""

    def build(self, spec: Any) -> Any:
        return spec

    @torch.no_grad()
    def run(
        self,
        state: Any,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        del state, workspace
        out.copy_(
            paged_gqa_reference(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
            )
        )
