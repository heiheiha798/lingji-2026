#!/usr/bin/env python3
"""Run the official public Ice Carver build, evaluator, and scorer."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.execvpe(
    "bash",
    ["bash", str(ROOT / "scripts" / "run_public.sh")],
    os.environ.copy(),
)
