from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import multiprocessing
import os
import platform
import random
import secrets
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from reference import kda_reference


PROBLEM_ID = "tilelang_kda_varlen_prefill"
INPUT_NAMES = (
    "q", "k", "v", "g_raw", "beta_raw", "a_log", "dt_bias",
    "initial_state", "cu_seqlens",
)
SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class OpSpec:
    total_tokens: int
    num_sequences: int
    num_heads: int
    head_dim: int = 128
    chunk_size: int = 64
    workspace_bytes: int = 128 * 1024 * 1024
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not (1 <= self.total_tokens <= 32768):
            raise ValueError("total_tokens must be in [1, 32768]")
        if not (1 <= self.num_sequences <= 32):
            raise ValueError("num_sequences must be in [1, 32]")
        if self.num_heads not in (8, 16):
            raise ValueError("num_heads must be 8 or 16")
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128")
        if self.chunk_size != 64:
            raise ValueError("chunk_size must be 64")
        if self.workspace_bytes != 128 * 1024 * 1024:
            raise ValueError("workspace_bytes must be 128 MiB")
        if self.dtype != torch.bfloat16:
            raise ValueError("dtype must be torch.bfloat16")


def _precompile_spec(spec_key: tuple[int, int, int]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    spec = OpSpec(
        total_tokens=spec_key[0],
        num_sequences=spec_key[1],
        num_heads=spec_key[2],
    )
    state = _load_submission().build(spec)
    return {
        "total_tokens": spec.total_tokens,
        "num_sequences": spec.num_sequences,
        "num_heads": spec.num_heads,
        "seconds": time.perf_counter() - started,
        "cache_keys": [
            getattr(kernel, "_tilelang_cache_key", None) for kernel in state[:2]
        ],
    }


@dataclass(frozen=True)
class ErrorPolicy:
    rtol: float
    atol: float
    required_matched_ratio: float
    max_abs_error: float
    max_nrmse: float
    max_local_nrmse: float


OUT_POLICY = ErrorPolicy(
    rtol=2e-2,
    atol=2e-2,
    required_matched_ratio=0.9999,
    max_abs_error=0.25,
    max_nrmse=1.0e-2,
    max_local_nrmse=2.0e-2,
)
STATE_POLICY = ErrorPolicy(
    rtol=3e-2,
    atol=2e-2,
    required_matched_ratio=0.9999,
    max_abs_error=0.30,
    max_nrmse=1.5e-2,
    max_local_nrmse=2.5e-2,
)


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    a = actual.float()
    e = expected.float()
    rmse = torch.mean((a - e) ** 2).sqrt()
    scale = torch.mean(e ** 2).sqrt().clamp_min(1e-3)
    return float((rmse / scale).item())


def _max_output_window_head_nrmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
    cu_seqlens: torch.Tensor,
    window: int = 64,
) -> float:
    if actual.numel() == 0:
        return 0.0
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    worst = 0.0
    for start, end in zip(cu[:-1], cu[1:]):
        for left in range(start, end, window):
            right = min(left + window, end)
            a = actual[left:right].float().permute(1, 0, 2).reshape(
                actual.shape[1], -1
            )
            e = expected[left:right].float().permute(1, 0, 2).reshape(
                expected.shape[1], -1
            )
            rmse = torch.mean((a - e) ** 2, dim=1).sqrt()
            scale = torch.mean(e ** 2, dim=1).sqrt().clamp_min(1e-3)
            worst = max(worst, float(torch.max(rmse / scale).item()))
    return worst


def _max_state_snapshot_head_nrmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if actual.numel() == 0 or actual.shape[0] == 0:
        return 0.0
    a = actual.float().reshape(actual.shape[0] * actual.shape[1], -1)
    e = expected.float().reshape(expected.shape[0] * expected.shape[1], -1)
    rmse = torch.mean((a - e) ** 2, dim=1).sqrt()
    scale = torch.mean(e ** 2, dim=1).sqrt().clamp_min(1e-3)
    return float(torch.max(rmse / scale).item())


def _check_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    policy: ErrorPolicy,
    *,
    local_nrmse: float,
) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: dtype mismatch {actual.dtype} != {expected.dtype}"
        )
    if actual.numel() == 0:
        return {
            "matched_ratio": 1.0,
            "max_abs_error": 0.0,
            "nrmse": 0.0,
            "max_local_nrmse": 0.0,
        }

    a = actual.float()
    e = expected.float()
    if not torch.isfinite(a).all():
        raise AssertionError(f"{name}: output contains NaN or Inf")
    if not torch.isfinite(e).all():
        raise AssertionError(f"{name}: reference contains NaN or Inf")

    diff = torch.abs(a - e)
    tolerance = policy.atol + policy.rtol * torch.abs(e)
    matched_ratio = float((diff <= tolerance).float().mean().item())
    max_abs = float(diff.max().item())
    nrmse = _nrmse(a, e)

    failures: list[str] = []
    if matched_ratio < policy.required_matched_ratio:
        failures.append(
            f"matched_ratio={matched_ratio:.8f} < "
            f"{policy.required_matched_ratio:.8f}"
        )
    if max_abs > policy.max_abs_error:
        failures.append(
            f"max_abs_error={max_abs:.6f} > {policy.max_abs_error:.6f}"
        )
    if nrmse > policy.max_nrmse:
        failures.append(f"nrmse={nrmse:.6f} > {policy.max_nrmse:.6f}")
    if local_nrmse > policy.max_local_nrmse:
        failures.append(
            f"max_local_nrmse={local_nrmse:.6f} > "
            f"{policy.max_local_nrmse:.6f}"
        )
    if failures:
        raise AssertionError(f"{name}: " + "; ".join(failures))

    return {
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs,
        "nrmse": nrmse,
        "max_local_nrmse": local_nrmse,
    }


def validate_outputs(
    out: torch.Tensor,
    final_state: torch.Tensor,
    ref_out: torch.Tensor,
    ref_final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        "out": _check_tensor(
            "out",
            out,
            ref_out,
            OUT_POLICY,
            local_nrmse=_max_output_window_head_nrmse(
                out, ref_out, cu_seqlens
            ),
        ),
        "final_state": _check_tensor(
            "final_state",
            final_state,
            ref_final_state,
            STATE_POLICY,
            local_nrmse=_max_state_snapshot_head_nrmse(
                final_state, ref_final_state
            ),
        ),
    }


def assert_inputs_unchanged(
    names: Iterable[str],
    tensors: Iterable[torch.Tensor],
    snapshots: Iterable[torch.Tensor],
) -> None:
    for name, tensor, snapshot in zip(names, tensors, snapshots):
        if tensor.shape != snapshot.shape or tensor.dtype != snapshot.dtype:
            raise AssertionError(f"{name}: input metadata changed")
        if not torch.equal(tensor, snapshot):
            raise AssertionError(f"{name}: input tensor was modified")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_submission():
    from submission import Submission

    return Submission()


def _preset(name: str) -> dict[str, int | float]:
    if name == "official":
        return {
            "warmup": 10,
            "trials": 5,
            "min_trial_ms": 200.0,
            "max_iterations": 1_000_000,
            "buffers": 4,
        }
    return {
        "warmup": 2,
        "trials": 3,
        "min_trial_ms": 20.0,
        "max_iterations": 10_000,
        "buffers": 1,
    }


def _make_spec(case: dict) -> tuple[OpSpec, list[int]]:
    lengths = [int(x) for x in case["lengths"]]
    spec = OpSpec(
        total_tokens=sum(lengths),
        num_sequences=len(lengths),
        num_heads=int(case["heads"]),
    )
    return spec, lengths


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    *,
    mean: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    value = torch.randn(
        shape, generator=generator, device=device, dtype=torch.float32
    )
    return (value * scale + mean).to(torch.bfloat16)


def _generate_input(
    spec: OpSpec,
    lengths: list[int],
    profile: dict,
    data_seed: int,
    device: torch.device,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(data_seed)

    q = _randn(
        (spec.total_tokens, spec.num_heads, 128), generator, device,
        scale=float(profile.get("q_scale", 0.5)),
    )
    k = _randn(
        (spec.total_tokens, spec.num_heads, 128), generator, device,
        scale=float(profile.get("k_scale", 0.5)),
    )
    v = _randn(
        (spec.total_tokens, spec.num_heads, 128), generator, device,
        scale=float(profile.get("v_scale", 0.5)),
    )
    g_raw = _randn(
        (spec.total_tokens, spec.num_heads, 128), generator, device,
        mean=float(profile.get("g_mean", 0.0)),
        scale=float(profile.get("g_scale", 0.5)),
    )
    beta_raw = _randn(
        (spec.total_tokens, spec.num_heads), generator, device,
        scale=float(profile.get("beta_scale", 1.0)),
    )
    a_log = torch.empty(
        (spec.num_heads,), dtype=torch.float32, device=device
    ).uniform_(-0.1, 0.1, generator=generator)
    dt_bias = torch.randn(
        (spec.num_heads, 128),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * float(profile.get("dt_bias_scale", 0.1))
    initial_state = _randn(
        (spec.num_sequences, spec.num_heads, 128, 128),
        generator,
        device,
        scale=float(profile.get("initial_state_scale", 0.02)),
    )

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    final_state = torch.empty(
        (spec.num_sequences, spec.num_heads, 128, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    return (
        q, k, v, g_raw, beta_raw, a_log, dt_bias, initial_state,
        cu_seqlens, workspace, out, final_state,
    )


def _make_buffers(
    spec: OpSpec,
    lengths: list[int],
    case: dict,
    variant_seed: int,
    device: torch.device,
    buffer_count: int,
) -> list[tuple[torch.Tensor, ...]]:
    workspace_size = (
        spec.workspace_bytes
        if device.type == "cuda"
        else min(spec.workspace_bytes, 8 * 1024 * 1024)
    )
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    profile = dict(case.get("profile", {}))
    buffers: list[tuple[torch.Tensor, ...]] = []
    for buffer_index in range(buffer_count):
        metadata_lengths = list(lengths)
        # 调整序列顺序时沿用 T 和 B，并更新变长输入的 cu_seqlens。
        if len(metadata_lengths) > 1:
            rng = random.Random(variant_seed + buffer_index * SEED_STRIDE)
            rng.shuffle(metadata_lengths)
        buffers.append(
            _generate_input(
                spec,
                metadata_lengths,
                profile,
                variant_seed + (buffer_index + 17) * SEED_STRIDE,
                device,
                workspace,
            )
        )
    return buffers


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _poison_outputs(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs[-2:]:
        tensor.fill_(float("nan"))


def _correctness_check(
    submission: Any,
    state: Any,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    repeats: int = 3,
) -> dict[str, dict[str, float]]:
    (
        q, k, v, g_raw, beta_raw, a_log, dt_bias, initial_state,
        cu_seqlens, workspace, out, final_state,
    ) = inputs
    expected = kda_reference(
        q, k, v, g_raw, beta_raw, a_log, dt_bias,
        initial_state, cu_seqlens,
    )
    immutable = inputs[:9]
    snapshots = [tensor.clone() for tensor in immutable]
    stats: dict[str, dict[str, float]] = {}

    for repeat in range(repeats):
        workspace.fill_((37 * repeat + 11) % 251)
        _poison_outputs(inputs)
        submission.run(state, *inputs)
        _sync(device)
        stats = validate_outputs(
            out,
            final_state,
            expected[0],
            expected[1],
            cu_seqlens,
        )
        assert_inputs_unchanged(INPUT_NAMES, immutable, snapshots)
    return stats


def _make_l2_scrub(device: torch.device) -> torch.Tensor | None:
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    l2_bytes = int(getattr(props, "l2_cache_size", 32 * 1024 * 1024))
    return torch.zeros(
        max(64 * 1024 * 1024, 2 * l2_bytes),
        dtype=torch.uint8,
        device=device,
    )


def _scrub_l2(buffer: torch.Tensor | None) -> None:
    if buffer is not None:
        buffer.add_(1)


class _Invoker:
    def __init__(
        self,
        submission: Any,
        state: Any,
        buffers: list[tuple[torch.Tensor, ...]],
    ) -> None:
        self.submission = submission
        self.state = state
        self.buffers = buffers
        self.buffer_count = len(buffers)

    def __call__(self, index: int) -> None:
        self.submission.run(self.state, *self.buffers[index])


def _time_invocations(
    invoke: _Invoker,
    device: torch.device,
    count: int,
    start_index: int,
) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for index in range(count):
            invoke((start_index + index) % invoke.buffer_count)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    t0 = time.perf_counter()
    for index in range(count):
        invoke((start_index + index) % invoke.buffer_count)
    return (time.perf_counter() - t0) * 1000.0


def _measure(
    invoke: _Invoker,
    device: torch.device,
    *,
    warmup: int,
    trials: int,
    min_trial_ms: float,
    max_iterations: int,
) -> dict[str, Any]:
    for index in range(warmup):
        invoke(index % invoke.buffer_count)
    _sync(device)

    calibration_count = 1
    calibration_ms = _time_invocations(invoke, device, calibration_count, 0)
    while calibration_ms < 5.0 and calibration_count < max_iterations:
        multiplier = max(2, int(math.ceil(5.0 / max(calibration_ms, 1e-4))))
        calibration_count = min(max_iterations, calibration_count * multiplier)
        calibration_ms = _time_invocations(
            invoke, device, calibration_count, 0
        )

    estimate_ms = max(calibration_ms / calibration_count, 1e-6)
    iterations = max(
        1,
        min(max_iterations, int(math.ceil(min_trial_ms / estimate_ms))),
    )
    scrub = _make_l2_scrub(device)
    samples_us: list[float] = []
    target_trials = trials

    while len(samples_us) < target_trials:
        trial_index = len(samples_us)
        _scrub_l2(scrub)
        elapsed_ms = _time_invocations(
            invoke,
            device,
            iterations,
            trial_index * iterations,
        )
        samples_us.append(elapsed_ms * 1000.0 / iterations)

        if len(samples_us) == trials:
            median = statistics.median(samples_us)
            mad = statistics.median(abs(x - median) for x in samples_us)
            if median > 0 and mad / median > 0.02 and trials < 9:
                target_trials = min(9, trials + 2)

    median = float(statistics.median(samples_us))
    mad = float(statistics.median(abs(x - median) for x in samples_us))
    relative_mad = 0.0 if median == 0 else mad / median
    return {
        "latency_us": median,
        "trial_us": samples_us,
        "iterations": iterations,
        "relative_mad": relative_mad,
        "noisy": relative_mad > 0.03,
    }


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _weighted_geomean(rows: list[dict[str, Any]], key: str) -> float:
    weight_sum = sum(float(row["weight"]) for row in rows)
    if weight_sum <= 0:
        raise ValueError("case weights must sum to a positive value")
    return math.exp(
        sum(
            (float(row["weight"]) / weight_sum) * math.log(float(row[key]))
            for row in rows
        )
    )


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import tilelang

        result["tilelang"] = getattr(tilelang, "__version__", "unknown")
    except ImportError:
        result["tilelang"] = "not-installed"
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["compute_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    return result


def _run_family(
    submission: Any,
    case: dict,
    device: torch.device,
    settings: dict[str, int | float],
    *,
    seed_count: int,
    check_before_bench: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, lengths = _make_spec(case)
    state = submission.build(spec)
    variants: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {}

    for variant_index in range(seed_count):
        variant_seed = int(case["seed"]) + variant_index * SEED_STRIDE
        buffers = _make_buffers(
            spec,
            lengths,
            case,
            variant_seed,
            device,
            int(settings["buffers"]),
        )
        if check_before_bench:
            correctness[str(variant_seed)] = _correctness_check(
                submission, state, buffers[0], device
            )
        measurement = _measure(
            _Invoker(submission, state, buffers),
            device,
            warmup=int(settings["warmup"]),
            trials=int(settings["trials"]),
            min_trial_ms=float(settings["min_trial_ms"]),
            max_iterations=int(settings["max_iterations"]),
        )
        measurement["seed"] = variant_seed
        variants.append(measurement)
        del buffers
        gc.collect()

    row = {
        "name": case["name"],
        "weight": float(case.get("weight", 1.0)),
        "latency_us": _geomean([v["latency_us"] for v in variants]),
        "variants": variants,
    }
    return row, correctness


def _random_correctness_cases(seed: int, count: int) -> list[dict[str, Any]]:
    """Generate small legal cases for correctness only; never used for timing."""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for index in range(count):
        batch_size = rng.choice((1, 2, 4, 8))
        lengths = [rng.randint(1, 512) for _ in range(batch_size)]
        lengths[0] = rng.choice((1, 63, 64, 65, 127, 128, 129))
        cases.append(
            {
                "name": f"random_correctness_{index + 1}",
                "heads": rng.choice((8, 16)),
                "lengths": lengths,
                "seed": rng.randrange(1, 2**31),
                "weight": 1.0,
            }
        )
    return cases


def _write_json(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已写入：{path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument(
        "--case",
        choices=["basic", "correctness", "performance", "official"],
        default=None,
    )
    parser.add_argument("--preset", choices=["local", "official"], default="local")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--random-correctness",
        action="store_true",
        help="运行不计性能的随机正确性测试",
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=1,
        help="并行预编译不同静态 shape，再串行运行测试",
    )
    args = parser.parse_args()

    if args.random_count <= 0:
        parser.error("--random-count must be positive")
    if args.compile_workers <= 0:
        parser.error("--compile-workers must be positive")
    if args.random_correctness and args.mode != "test":
        parser.error("random correctness cases do not support performance timing")

    settings = _preset(args.preset)

    case_group = args.case or (
        "correctness" if args.mode == "test" else "performance"
    )
    random_seed: int | None = None
    if args.random_correctness:
        random_seed = (
            args.random_seed
            if args.random_seed is not None
            else secrets.randbits(63)
        )
        cases = _random_correctness_cases(random_seed, args.random_count)
        case_group = "random_correctness"
        print(f"随机正确性测试 seed={random_seed}")
    else:
        cases = json.loads(
            Path(__file__).with_name("cases.json").read_text(encoding="utf-8")
        )[case_group]

    precompile: dict[str, Any] | None = None
    if args.compile_workers > 1:
        spec_keys = list(
            dict.fromkeys(
                (
                    sum(int(length) for length in case["lengths"]),
                    len(case["lengths"]),
                    int(case["heads"]),
                )
                for case in cases
            )
        )
        worker_count = min(args.compile_workers, len(spec_keys))
        started = time.perf_counter()
        compiled_specs: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_precompile_spec, key) for key in spec_keys]
            for future in concurrent.futures.as_completed(futures):
                compiled = future.result()
                compiled_specs.append(compiled)
                print(
                    "缓存就绪  "
                    f"T={compiled['total_tokens']} "
                    f"B={compiled['num_sequences']} "
                    f"H={compiled['num_heads']}  "
                    f"{compiled['seconds']:.3f} s"
                )
        compiled_specs.sort(
            key=lambda item: (
                item["total_tokens"],
                item["num_sequences"],
                item["num_heads"],
            )
        )
        precompile = {
            "workers": worker_count,
            "unique_specs": len(spec_keys),
            "wall_seconds": time.perf_counter() - started,
            "specs": compiled_specs,
        }
        print(
            f"并行预编译完成：{len(spec_keys)} 个静态 shape，"
            f"workers={worker_count}，wall={precompile['wall_seconds']:.3f} s"
        )

    device = _resolve_device(args.device)
    if device.type == "cpu":
        # CPU 参考计算最多使用 4 个线程。该设置仅作用于 CPU。
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    submission = _load_submission()

    if args.mode == "test":
        correctness: dict[str, Any] = {}
        for case in cases:
            spec, lengths = _make_spec(case)
            state = submission.build(spec)
            buffers = _make_buffers(
                spec, lengths, case, int(case["seed"]), device, 1
            )
            stats = _correctness_check(submission, state, buffers[0], device)
            correctness[case["name"]] = stats
            summary = ", ".join(
                f"{name}: nrmse={values['nrmse']:.3e}"
                for name, values in stats.items()
            )
            print(f"通过  {case['name']}  {summary}")
        print(f"全部 {len(cases)} 个正确性用例通过，设备：{device}")
        _write_json(
            args.json_out,
            {
                "schema_version": 2,
                "problem": PROBLEM_ID,
                "report_type": "correctness",
                "environment": _environment(device),
                "case_group": case_group,
                "random_seed": random_seed,
                "precompile": precompile,
                "passed": True,
                "correctness": correctness,
            },
        )
        return

    seed_count = 4 if case_group == "official" else 1

    # 主办方在正式评测中使用加速参考实现检查大规模结果。
    check_before_bench = case_group != "official"
    if case_group == "official":
        print(
            "提示：公开性能用例的本地命令用于测量性能。"
            "正式评测会单独检查正确性。"
        )

    rows: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {}
    for case in cases:
        row, stats = _run_family(
            submission,
            case,
            device,
            settings,
            seed_count=seed_count,
            check_before_bench=check_before_bench,
        )
        rows.append(row)
        if stats:
            correctness[case["name"]] = stats
        max_mad = max(v["relative_mad"] for v in row["variants"])
        print(
            f"{row['name']:<30} {row['latency_us']:>12.3f} us  "
            f"数据组={len(row['variants'])}  波动={max_mad * 100:>5.2f}%"
        )

    result: dict[str, Any] = {
        "schema_version": 2,
        "problem": PROBLEM_ID,
        "report_type": "correctness_and_performance",
        "environment": _environment(device),
        "case_group": case_group,
        "preset": args.preset,
        "settings": settings,
        "precompile": precompile,
        "seed_count": seed_count,
        "correctness_mode": (
            "local_reference" if check_before_bench else "separate_official_oracle"
        ),
        "weighted_geomean_us": _weighted_geomean(rows, "latency_us"),
        "correctness": correctness,
        "cases": rows,
    }
    if check_before_bench:
        print("正确性检查通过")
    print(f"加权几何平均延迟：{result['weighted_geomean_us']:.3f} 微秒")
    _write_json(args.json_out, result)


if __name__ == "__main__":
    main()
