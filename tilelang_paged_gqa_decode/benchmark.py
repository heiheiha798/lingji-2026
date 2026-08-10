from __future__ import annotations

import argparse
import gc
import json
import math
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
import torch.nn.functional as F

from reference import paged_gqa_reference


PROBLEM_ID = "tilelang_paged_gqa_decode"
INPUT_NAMES = ("q", "k_cache", "v_cache", "block_table", "seq_lens")
MAX_VALID_TOKENS = 524_288
SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class OpSpec:
    batch_size: int
    num_q_heads: int
    num_kv_heads: int
    max_seq_len: int
    num_pages: int
    head_dim: int = 128
    page_size: int = 16
    workspace_bytes: int = 128 * 1024 * 1024
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.batch_size not in (1, 8, 16, 32, 64, 128):
            raise ValueError("unsupported batch_size")
        if self.num_q_heads not in (32, 64):
            raise ValueError("num_q_heads must be 32 or 64")
        if self.num_kv_heads <= 0 or self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.num_q_heads // self.num_kv_heads not in (4, 8):
            raise ValueError("GQA group size must be 4 or 8")
        if not (1 <= self.max_seq_len <= 32768):
            raise ValueError("max_seq_len must be in [1, 32768]")
        if self.num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128")
        if self.page_size not in (16, 32):
            raise ValueError("page_size must be 16 or 32")
        if self.workspace_bytes != 128 * 1024 * 1024:
            raise ValueError("workspace_bytes must be 128 MiB")
        if self.dtype != torch.bfloat16:
            raise ValueError("dtype must be torch.bfloat16")


@dataclass(frozen=True)
class ErrorPolicy:
    rtol: float = 1e-2
    atol: float = 1e-2
    required_matched_ratio: float = 1.0
    max_abs_error: float = 5e-2
    max_nrmse: float = 1e-2
    min_head_cosine: float = 0.999


POLICY = ErrorPolicy()


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    a = actual.float()
    e = expected.float()
    rmse = torch.mean((a - e) ** 2).sqrt()
    scale = torch.mean(e ** 2).sqrt().clamp_min(1e-3)
    return float((rmse / scale).item())


def validate_output(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"out: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"out: dtype mismatch {actual.dtype} != {expected.dtype}"
        )
    a = actual.float()
    e = expected.float()
    if not torch.isfinite(a).all():
        raise AssertionError("out: output contains NaN or Inf")
    if not torch.isfinite(e).all():
        raise AssertionError("out: reference contains NaN or Inf")

    diff = torch.abs(a - e)
    tolerance = POLICY.atol + POLICY.rtol * torch.abs(e)
    matched_ratio = float((diff <= tolerance).float().mean().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    nrmse = _nrmse(a, e)

    flat_a = a.reshape(a.shape[0] * a.shape[1], -1)
    flat_e = e.reshape(e.shape[0] * e.shape[1], -1)
    ref_norm = torch.linalg.vector_norm(flat_e, dim=1)
    cosine = F.cosine_similarity(flat_a, flat_e, dim=1, eps=1e-8)
    cosine = torch.where(ref_norm < 1e-7, torch.ones_like(cosine), cosine)
    min_head_cosine = float(cosine.min().item()) if cosine.numel() else 1.0

    failures: list[str] = []
    if matched_ratio < POLICY.required_matched_ratio:
        failures.append(
            f"matched_ratio={matched_ratio:.8f} < "
            f"{POLICY.required_matched_ratio:.8f}"
        )
    if max_abs > POLICY.max_abs_error:
        failures.append(
            f"max_abs_error={max_abs:.6f} > {POLICY.max_abs_error:.6f}"
        )
    if nrmse > POLICY.max_nrmse:
        failures.append(f"nrmse={nrmse:.6f} > {POLICY.max_nrmse:.6f}")
    if min_head_cosine < POLICY.min_head_cosine:
        failures.append(
            f"min_head_cosine={min_head_cosine:.8f} < "
            f"{POLICY.min_head_cosine:.8f}"
        )
    if failures:
        raise AssertionError("out: " + "; ".join(failures))

    return {
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs,
        "nrmse": nrmse,
        "min_head_cosine": min_head_cosine,
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


def _generate_lengths(case: dict, rng: random.Random) -> list[int]:
    spec = case["seq_lens"]
    mode = spec["mode"]
    batch = int(case["batch_size"])

    if mode == "explicit":
        values = [int(x) for x in spec["values"]]
    elif mode == "uniform":
        values = [
            rng.randint(int(spec["low"]), int(spec["high"]))
            for _ in range(batch)
        ]
    elif mode == "jittered_base":
        values = [
            max(
                1,
                round(
                    float(base)
                    * rng.uniform(
                        float(spec["low_scale"]),
                        float(spec["high_scale"]),
                    )
                ),
            )
            for base in spec["base"]
        ]
    elif mode == "mixture":
        values = []
        for group in spec["groups"]:
            values.extend(
                rng.randint(int(group["low"]), int(group["high"]))
                for _ in range(int(group["count"]))
            )
    else:
        raise ValueError(f"unknown seq_lens mode: {mode}")

    if len(values) != batch:
        raise ValueError(f"generated {len(values)} lengths for batch={batch}")
    rng.shuffle(values)
    return values


def _upper_lengths(case: dict) -> list[int]:
    spec = case["seq_lens"]
    mode = spec["mode"]
    batch = int(case["batch_size"])
    if mode == "explicit":
        return [int(x) for x in spec["values"]]
    if mode == "uniform":
        return [int(spec["high"])] * batch
    if mode == "jittered_base":
        return [
            max(1, math.ceil(float(x) * float(spec["high_scale"])))
            for x in spec["base"]
        ]
    if mode == "mixture":
        values: list[int] = []
        for group in spec["groups"]:
            values.extend([int(group["high"])] * int(group["count"]))
        return values
    raise ValueError(f"unknown seq_lens mode: {mode}")


def _capacity_pages(case: dict) -> int:
    page_size = int(case["page_size"])
    pages = sum(math.ceil(length / page_size) for length in _upper_lengths(case))
    return pages + max(8, int(case["batch_size"]) // 2)


def _make_spec(case: dict) -> OpSpec:
    max_seq_len = int(case["max_seq_len"])
    if max(_upper_lengths(case)) > max_seq_len:
        raise ValueError("max_seq_len is smaller than the declared workload range")
    return OpSpec(
        batch_size=int(case["batch_size"]),
        num_q_heads=int(case["num_q_heads"]),
        num_kv_heads=int(case["num_kv_heads"]),
        max_seq_len=max_seq_len,
        num_pages=_capacity_pages(case),
        page_size=int(case["page_size"]),
    )


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    return (
        torch.randn(
            shape, generator=generator, device=device, dtype=torch.float32
        )
        * scale
    ).to(torch.bfloat16)


def _metadata(
    spec: OpSpec,
    case: dict,
    metadata_seed: int,
) -> tuple[list[int], torch.Tensor]:
    rng = random.Random(metadata_seed)
    lengths = _generate_lengths(case, rng)
    if any(length > spec.max_seq_len for length in lengths):
        raise ValueError("generated sequence length exceeds max_seq_len")
    if sum(lengths) > MAX_VALID_TOKENS:
        raise ValueError("valid KV token count exceeds the task limit")

    pages_needed = [math.ceil(length / spec.page_size) for length in lengths]
    page_ids = list(range(spec.num_pages))
    rng.shuffle(page_ids)
    table = torch.full(
        (spec.batch_size, math.ceil(spec.max_seq_len / spec.page_size)),
        -1,
        dtype=torch.int32,
    )

    shared_prefix_pages = min(
        int(case.get("shared_prefix_pages", 0)),
        min(pages_needed) if pages_needed else 0,
    )
    common = page_ids[:shared_prefix_pages]
    cursor = shared_prefix_pages
    for batch_index, count in enumerate(pages_needed):
        prefix = min(shared_prefix_pages, count)
        if prefix:
            table[batch_index, :prefix] = torch.tensor(common[:prefix], dtype=torch.int32)
        unique_count = count - prefix
        if unique_count:
            selected = page_ids[cursor : cursor + unique_count]
            if len(selected) != unique_count:
                raise RuntimeError("insufficient physical pages")
            table[batch_index, prefix : prefix + unique_count] = torch.tensor(
                selected, dtype=torch.int32
            )
            cursor += unique_count
    return lengths, table


def _generate_input(
    spec: OpSpec,
    case: dict,
    metadata_seed: int,
    data_seed: int,
    device: torch.device,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    lengths, block_table_cpu = _metadata(spec, case, metadata_seed)
    profile = dict(case.get("profile", {}))
    generator = torch.Generator(device=device)
    generator.manual_seed(data_seed)

    q = _randn(
        (spec.batch_size, spec.num_q_heads, 128),
        generator,
        device,
        float(profile.get("q_scale", 0.5)),
    )
    k_cache = _randn(
        (spec.num_pages, spec.page_size, spec.num_kv_heads, 128),
        generator,
        device,
        float(profile.get("k_scale", 0.5)),
    )
    v_cache = _randn(
        (spec.num_pages, spec.page_size, spec.num_kv_heads, 128),
        generator,
        device,
        float(profile.get("v_scale", 0.5)),
    )
    block_table = block_table_cpu.to(device=device)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    return q, k_cache, v_cache, block_table, seq_lens, workspace, out


def _make_buffers(
    spec: OpSpec,
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
    return [
        _generate_input(
            spec,
            case,
            variant_seed + buffer_index * SEED_STRIDE,
            variant_seed + (buffer_index + 17) * SEED_STRIDE,
            device,
            workspace,
        )
        for buffer_index in range(buffer_count)
    ]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _correctness_check(
    submission: Any,
    state: Any,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    repeats: int = 3,
) -> dict[str, float]:
    q, k_cache, v_cache, block_table, seq_lens, workspace, out = inputs
    expected = paged_gqa_reference(q, k_cache, v_cache, block_table, seq_lens)
    immutable = inputs[:5]
    snapshots = [tensor.clone() for tensor in immutable]
    stats: dict[str, float] = {}

    for repeat in range(repeats):
        workspace.fill_((37 * repeat + 11) % 251)
        out.fill_(float("nan"))
        submission.run(state, *inputs)
        _sync(device)
        stats = validate_output(out, expected)
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
    spec = _make_spec(case)
    state = submission.build(spec)
    variants: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {}

    for variant_index in range(seed_count):
        variant_seed = int(case["seed"]) + variant_index * SEED_STRIDE
        buffers = _make_buffers(
            spec,
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
        batch_size = rng.choice((1, 8, 16, 32))
        num_q_heads = rng.choice((32, 64))
        group_size = rng.choice((4, 8))
        page_size = rng.choice((16, 32))
        max_seq_len = rng.choice((33, 65, 127, 257, 513, 1024))
        lengths = [rng.randint(1, max_seq_len) for _ in range(batch_size)]
        lengths[0] = 1
        if batch_size > 1:
            lengths[-1] = max_seq_len
        cases.append(
            {
                "name": f"random_correctness_{index + 1}",
                "batch_size": batch_size,
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_q_heads // group_size,
                "page_size": page_size,
                "max_seq_len": max_seq_len,
                "seq_lens": {"mode": "explicit", "values": lengths},
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
    args = parser.parse_args()

    if args.random_count <= 0:
        parser.error("--random-count must be positive")
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
    device = _resolve_device(args.device)
    if device.type == "cpu":
        # CPU 参考计算最多使用 4 个线程。该设置仅作用于 CPU。
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    submission = _load_submission()

    if args.mode == "test":
        correctness: dict[str, Any] = {}
        for case in cases:
            spec = _make_spec(case)
            state = submission.build(spec)
            buffers = _make_buffers(spec, case, int(case["seed"]), device, 1)
            stats = _correctness_check(submission, state, buffers[0], device)
            correctness[case["name"]] = stats
            print(
                f"通过  {case['name']}  nrmse={stats['nrmse']:.3e}  "
                f"max_abs={stats['max_abs_error']:.3e}"
            )
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
