"""Single-entry CUDA string lexicographic sort Judge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.utils.cpp_extension import load

from .contract import CaseSpec, validate_inputs
from .data import DataError, load_case, verify_sha256
from .manifest import ManifestError, load_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "cases.json"
CUDA_SOURCE = ROOT / "submission.cu"
BINDING_SOURCE = Path(__file__).with_name("binding.cpp")
GPU_NAME_CONTAINS = "NVIDIA GeForce RTX 4090"
COLD_WALL_TIME_LIMIT_SEC = 10 * 60
FULL_SCORE_GPU_TIME_US = 220.0


@functools.lru_cache(maxsize=1)
def _load_extension() -> Any:
    digest = hashlib.sha256()
    for source in (BINDING_SOURCE, CUDA_SOURCE):
        digest.update(source.read_bytes())
    source_hash = digest.hexdigest()[:12]
    return load(
        name=f"string_lexsort_submission_{source_hash}",
        sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_89,code=sm_89",
        ],
        with_cuda=True,
        verbose=False,
    )


class JudgeFailure(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    n: int
    width: int
    weight: float
    gpu_time_us: float | None


def _check_environment() -> str:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise JudgeFailure("JUDGE_ERROR", "exactly one visible CUDA GPU is required")
    name = torch.cuda.get_device_name(0)
    if GPU_NAME_CONTAINS not in name:
        raise JudgeFailure(
            "JUDGE_ERROR",
            f"expected GPU containing {GPU_NAME_CONTAINS!r}, got {name!r}",
        )
    return name


def _check_exact_output(
    case: CaseSpec,
    observed: torch.Tensor,
    golden: np.ndarray,
) -> None:
    if (
        tuple(observed.shape) != (case.n,)
        or observed.dtype != torch.int32
        or not observed.is_contiguous()
        or not observed.is_cuda
    ):
        raise JudgeFailure("WRONG_ANSWER", f"{case.case_id}: invalid output tensor")
    actual = observed.detach().cpu().numpy()
    if not np.array_equal(actual, golden):
        mismatch = np.flatnonzero(actual != golden)
        first = int(mismatch[0]) if mismatch.size else -1
        raise JudgeFailure(
            "WRONG_ANSWER",
            f"{case.case_id}: wrong permutation at output position {first}",
        )


def _check_inputs_unchanged(
    case: CaseSpec,
    strings: torch.Tensor,
    lengths: torch.Tensor,
    strings_cpu: np.ndarray,
    lengths_cpu: np.ndarray,
) -> None:
    if not np.array_equal(strings.detach().cpu().numpy(), strings_cpu):
        raise JudgeFailure(
            "WRONG_ANSWER", f"{case.case_id}: submission modified strings"
        )
    if not np.array_equal(lengths.detach().cpu().numpy(), lengths_cpu):
        raise JudgeFailure(
            "WRONG_ANSWER", f"{case.case_id}: submission modified lengths"
        )


def _evaluate_case(
    extension: Any,
    case: CaseSpec,
) -> CaseResult:
    data = load_case(DATA_DIR, case)
    strings = torch.from_numpy(data.strings.copy()).to(device="cuda")
    lengths = torch.from_numpy(data.lengths.copy()).to(device="cuda")
    indices_out = torch.empty(case.n, dtype=torch.int32, device="cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    try:
        workspace_nbytes = extension.workspace_bytes(case.n, case.width)
        if (
            isinstance(workspace_nbytes, bool)
            or not isinstance(workspace_nbytes, int)
            or workspace_nbytes < 0
        ):
            raise JudgeFailure(
                "RUNTIME_ERROR",
                f"{case.case_id}: workspace_bytes must return a nonnegative integer",
            )
        workspace = torch.empty(
            workspace_nbytes, dtype=torch.uint8, device="cuda"
        )
        validate_inputs(case, strings, lengths, indices_out)
        torch.cuda.current_stream().synchronize()

        # The exact validation run is also the complete warmup for this case.
        indices_out.fill_(-1)
        with torch.inference_mode():
            extension.lexsort_cuda(strings, lengths, indices_out, workspace)
        torch.cuda.current_stream().synchronize()
        _check_exact_output(case, indices_out, data.golden_indices)
        _check_inputs_unchanged(
            case, strings, lengths, data.strings, data.lengths
        )

        if not case.scored:
            return CaseResult(
                case_id=case.case_id,
                n=case.n,
                width=case.width,
                weight=case.weight,
                gpu_time_us=None,
            )

        indices_out.fill_(-1)
        start_event.record(torch.cuda.current_stream())
        with torch.inference_mode():
            extension.lexsort_cuda(strings, lengths, indices_out, workspace)
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        gpu_time_us = start_event.elapsed_time(end_event) * 1000.0
        if not math.isfinite(gpu_time_us) or gpu_time_us <= 0.0:
            raise JudgeFailure(
                "INVALID_TIME", f"{case.case_id}: invalid CUDA Event time"
            )

        # This comparison is after the final stop Event and does not affect time.
        _check_exact_output(case, indices_out, data.golden_indices)
        _check_inputs_unchanged(
            case, strings, lengths, data.strings, data.lengths
        )
        return CaseResult(
            case_id=case.case_id,
            n=case.n,
            width=case.width,
            weight=case.weight,
            gpu_time_us=gpu_time_us,
        )
    except torch.OutOfMemoryError as error:
        raise JudgeFailure("OOM", f"{case.case_id}: CUDA out of memory") from error


def _weighted_gpu_time_us(results: list[CaseResult]) -> float:
    terms = []
    for result in results:
        if result.weight == 0.0:
            continue
        assert result.gpu_time_us is not None
        terms.append(result.weight * math.log(result.gpu_time_us))
    return math.exp(math.fsum(terms))


def _score(weighted_gpu_time_us: float) -> float:
    return min(100.0, 100.0 * FULL_SCORE_GPU_TIME_US / weighted_gpu_time_us)


def _result_mapping(
    *,
    status: str,
    wall_time_sec: float,
    gpu_name: str | None,
    results: list[CaseResult],
    message: str | None = None,
) -> dict[str, Any]:
    weighted_gpu_time_us = (
        _weighted_gpu_time_us(results)
        if status == "PASS" and results
        else None
    )
    mapping: dict[str, Any] = {
        "status": status,
        "wall_time_sec": wall_time_sec,
        "gpu_name": gpu_name,
        "weighted_gpu_time_us": weighted_gpu_time_us,
        "score": _score(weighted_gpu_time_us) if weighted_gpu_time_us else None,
        "cases": {item.case_id: asdict(item) for item in results},
    }
    if message:
        mapping["message"] = message
    return mapping


def _print_report(result: dict[str, Any]) -> None:
    cases = result["cases"]
    if cases:
        print("CASE          N   W      GPU_TIME(us)", flush=True)
        for case_id, row in cases.items():
            if row["weight"] == 0.0:
                print(
                    f"{case_id:<10} {row['n']:>7} {row['width']:>3}"
                    "         CHECK",
                    flush=True,
                )
            else:
                print(
                    f"{case_id:<10} {row['n']:>7} {row['width']:>3}"
                    f" {row['gpu_time_us']:>15.3f}",
                    flush=True,
                )
    print(f"Status: {result['status']}", flush=True)
    if result.get("message"):
        print(f"Message: {result['message']}", flush=True)
    if result.get("weighted_gpu_time_us") is not None:
        print(
            f"WeightedGpuTimeUs: {result['weighted_gpu_time_us']:.6f}",
            flush=True,
        )
        print(f"Score: {result['score']:.6f}", flush=True)
    print(f"WallTimeSec: {result['wall_time_sec']:.2f}", flush=True)
    print(
        "RESULT_JSON="
        + json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )


def main() -> int:
    started = time.perf_counter()
    results: list[CaseResult] = []
    gpu_name = None
    try:
        manifest = load_manifest(MANIFEST_PATH)
        verify_sha256(manifest, DATA_DIR)
        gpu_name = _check_environment()
        try:
            extension = _load_extension()
        except Exception as error:
            raise JudgeFailure(
                "COMPILE_ERROR", "cannot compile or load submission.cu"
            ) from error
        for case in manifest.cases:
            try:
                result = _evaluate_case(extension, case)
                results.append(result)
                if result.weight > 0.0:
                    assert result.gpu_time_us is not None
                    print(
                        f"{result.case_id}: correctness PASS, "
                        f"GPU_TIME={result.gpu_time_us:.3f} us",
                        flush=True,
                    )
                else:
                    print(f"{result.case_id}: correctness PASS", flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        wall_time = time.perf_counter() - started
        if wall_time > COLD_WALL_TIME_LIMIT_SEC:
            raise JudgeFailure(
                "TIME_LIMIT",
                f"cold wall time {wall_time:.2f}s exceeds package limit",
            )
        mapping = _result_mapping(
            status="PASS",
            wall_time_sec=wall_time,
            gpu_name=gpu_name,
            results=results,
        )
        _print_report(mapping)
        return 0
    except JudgeFailure as error:
        status, message = error.status, str(error)
    except (DataError, ManifestError) as error:
        status, message = "JUDGE_ERROR", str(error)
    except Exception as error:
        status = "RUNTIME_ERROR"
        message = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    mapping = _result_mapping(
        status=status,
        wall_time_sec=time.perf_counter() - started,
        gpu_name=gpu_name,
        results=results,
        message=message,
    )
    _print_report(mapping)
    return 2 if status == "JUDGE_ERROR" else 1


__all__ = ["main"]
