"""Single-process, single-entry KDA contest Judge."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping

import torch

from .contract import append_result_key, decode_result_key
from .counter_v1 import SHARD_STEPS, CounterV1Generator, load_generator_luts
from .data import (
    DataError,
    counter_case,
    load_case_goldens,
    load_case_static,
    load_decode_output_shard,
    verify_counter_canaries,
    verify_golden_reference,
)
from .manifest import FrozenCase, ManifestError, SimpleManifest, load_manifest
from .metrics import (
    DecodeOutputMetricsAccumulator,
    TensorCheckError,
    TensorMetrics,
    case_gpu_time_ms,
    compare_tensor,
    enforce_limits,
    weighted_gpu_time_ms,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "cases.json"


class JudgeFailure(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _submission_call(
    action: str,
    call: Callable[[], Any],
    *,
    synchronize: bool = False,
) -> Any:
    try:
        result = call()
        if synchronize:
            torch.cuda.current_stream().synchronize()
        return result
    except torch.OutOfMemoryError as error:
        raise JudgeFailure("OOM", f"submission.{action} ran out of memory") from error
    except Exception as error:
        raise JudgeFailure(
            "RUNTIME_ERROR",
            f"submission.{action} failed: {type(error).__name__}: {error}",
        ) from error


@dataclass(frozen=True)
class AccuracySummary:
    max_limit_ratio: float
    max_output_relative_l2: float
    max_state_relative_l2: float
    max_normalized_max: float
    worst_sequence_relative_l2: float
    worst_head_relative_l2: float
    min_sequence_reference_norm: float
    min_head_reference_norm: float
    checked_output_tensors: int
    checked_state_tensors: int


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    weight: float
    timing_replays: int
    case_gpu_time_ms: float
    accuracy: AccuracySummary


@dataclass
class _AccuracyAccumulator:
    max_limit_ratio: float = 0.0
    max_output_relative_l2: float = 0.0
    max_state_relative_l2: float = 0.0
    max_normalized_max: float = 0.0
    worst_sequence_relative_l2: float = 0.0
    worst_head_relative_l2: float = 0.0
    min_sequence_reference_norm: float = math.inf
    min_head_reference_norm: float = math.inf
    checked_output_tensors: int = 0
    checked_state_tensors: int = 0

    def update(
        self,
        metrics: TensorMetrics,
        *,
        kind: str,
        relative_l2_limit: float,
        normalized_max_limit: float,
    ) -> None:
        ratio = enforce_limits(
            metrics,
            relative_l2_limit=relative_l2_limit,
            normalized_max_limit=normalized_max_limit,
        )
        self.max_limit_ratio = max(self.max_limit_ratio, ratio)
        if kind == "output":
            self.max_output_relative_l2 = max(
                self.max_output_relative_l2, metrics.relative_l2
            )
            self.checked_output_tensors += 1
        elif kind == "state":
            self.max_state_relative_l2 = max(
                self.max_state_relative_l2, metrics.relative_l2
            )
            self.checked_state_tensors += 1
        else:
            raise ValueError("accuracy kind must be output or state")
        self.max_normalized_max = max(
            self.max_normalized_max, metrics.normalized_max
        )
        self.worst_sequence_relative_l2 = max(
            self.worst_sequence_relative_l2,
            metrics.worst_sequence_relative_l2,
        )
        self.worst_head_relative_l2 = max(
            self.worst_head_relative_l2,
            metrics.worst_head_relative_l2,
        )
        self.min_sequence_reference_norm = min(
            self.min_sequence_reference_norm,
            metrics.min_sequence_reference_norm,
        )
        self.min_head_reference_norm = min(
            self.min_head_reference_norm,
            metrics.min_head_reference_norm,
        )

    def freeze(self) -> AccuracySummary:
        if not self.checked_output_tensors or not self.checked_state_tensors:
            raise TensorCheckError("validation did not check output and state")
        return AccuracySummary(
            max_limit_ratio=self.max_limit_ratio,
            max_output_relative_l2=self.max_output_relative_l2,
            max_state_relative_l2=self.max_state_relative_l2,
            max_normalized_max=self.max_normalized_max,
            worst_sequence_relative_l2=self.worst_sequence_relative_l2,
            worst_head_relative_l2=self.worst_head_relative_l2,
            min_sequence_reference_norm=self.min_sequence_reference_norm,
            min_head_reference_norm=self.min_head_reference_norm,
            checked_output_tensors=self.checked_output_tensors,
            checked_state_tensors=self.checked_state_tensors,
        )


class CaseEvaluator:
    def __init__(
        self,
        submission: Any,
        manifest: SimpleManifest,
        frozen: FrozenCase,
        luts: Any,
    ) -> None:
        self.submission = submission
        self.manifest = manifest
        self.frozen = frozen
        self.case = frozen.case

        # Everything owned by the Judge is allocated before submission.prepare.
        try:
            self.counter = counter_case(frozen, manifest)
            self.generator = CounterV1Generator(self.counter, luts, device="cuda")
            self.static = load_case_static(
                manifest,
                frozen,
                self.generator,
                data_dir=DATA_DIR,
                device="cuda",
            )
            self.goldens = load_case_goldens(
                manifest,
                frozen,
                data_dir=DATA_DIR,
                device="cuda",
            )
            self.append_outputs = tuple(
                torch.empty_like(args.v_act) for args in self.static.append_inputs
            )
            self.decode_output = (
                torch.empty(
                    self.case.batch,
                    self.case.config.heads,
                    self.case.config.value_dim,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                if self.case.decode_steps
                else None
            )
            self.export_buffer = torch.empty_like(self.static.initial_state)
            self.decode_buffers = (
                self.generator.allocate_decode_buffers()
                if self.case.decode_steps
                else None
            )
            decode_shard_shape = (
                SHARD_STEPS,
                self.case.batch,
                self.case.config.heads,
                self.case.config.value_dim,
            )
            self.decode_actual_shard = (
                torch.empty(
                    decode_shard_shape, dtype=torch.bfloat16, device="cuda"
                )
                if self.case.decode_steps
                else None
            )
            self.decode_golden_shard = (
                torch.empty_like(self.decode_actual_shard)
                if self.decode_actual_shard is not None
                else None
            )
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().synchronize()
        except torch.OutOfMemoryError as error:
            raise JudgeFailure(
                "OOM",
                f"GPU memory is unavailable while preparing {self.frozen.case_id}",
            ) from error

        self.context = _submission_call(
            "prepare",
            lambda: submission.prepare(
                self.case.config, self.static.layer, self.case
            ),
            synchronize=True,
        )

    def evaluate(self) -> CaseResult:
        accuracy = self._run_validation_round()
        replay_times = []
        for replay in range(self.frozen.timing_replays):
            replay_times.append(
                self._run_timing_round(
                    verify_final=replay == self.frozen.timing_replays - 1,
                    accuracy=accuracy,
                )
            )
        case_time = case_gpu_time_ms(
            replay_times,
            expected_replays=self.frozen.timing_replays,
        )
        return CaseResult(
            case_id=self.frozen.case_id,
            weight=self.frozen.weight,
            timing_replays=self.frozen.timing_replays,
            case_gpu_time_ms=case_time,
            accuracy=accuracy.freeze(),
        )

    def _run_validation_round(self) -> _AccuracyAccumulator:
        accuracy = _AccuracyAccumulator()
        private_state = _submission_call(
            "load_state",
            lambda: self.submission.load_state(
                self.context, self.static.initial_state
            ),
            synchronize=True,
        )
        for index, (args, output) in enumerate(
            zip(self.static.append_inputs, self.append_outputs)
        ):
            output.fill_(float("nan"))
            _submission_call(
                "append_chunk",
                lambda: self.submission.append_chunk(
                    self.context, private_state, args, output
                ),
                synchronize=True,
            )
            key = append_result_key(index)
            self._check_output(
                output,
                self.goldens.append_output[key],
                accuracy,
                sequence_lengths=self.case.append_lengths[index],
            )
            self._export_and_check(private_state, key, accuracy)

        if self.case.decode_steps:
            assert (
                self.decode_output is not None
                and self.decode_buffers is not None
                and self.decode_actual_shard is not None
                and self.decode_golden_shard is not None
            )
            state_checkpoints = set(self.manifest.state_checkpoints)
            decode_metrics = DecodeOutputMetricsAccumulator(
                expected_steps=self.case.decode_steps,
                batch=self.case.batch,
                heads=self.case.config.heads,
                value_dim=self.case.config.value_dim,
                device="cuda",
            )
            for shard_start in range(
                0, self.case.decode_steps, SHARD_STEPS
            ):
                count = min(
                    SHARD_STEPS,
                    self.case.decode_steps - shard_start,
                )
                self.generator.fill_decode_shard(
                    self.decode_buffers, shard_start, count
                )
                actual_shard = self.decode_actual_shard[:count]
                actual_shard.fill_(float("nan"))
                for local_index in range(count):
                    token_index = shard_start + local_index
                    token = self.generator.select_current_token(
                        self.decode_buffers, token_index
                    )
                    self.decode_output.fill_(float("nan"))
                    _submission_call(
                        "decode_step",
                        lambda: self.submission.decode_step(
                            self.context, private_state, token, self.decode_output
                        ),
                    )
                    actual_shard[local_index].copy_(self.decode_output)
                    step = token_index + 1
                    key = decode_result_key(step)
                    if step in state_checkpoints:
                        self._export_and_check(private_state, key, accuracy)
                _submission_call(
                    "decode_step", lambda: None, synchronize=True
                )
                golden_shard = load_decode_output_shard(
                    self.goldens,
                    shard_start=shard_start,
                    count=count,
                    destination=self.decode_golden_shard,
                )
                decode_metrics.update(
                    actual_shard,
                    golden_shard,
                    shard_start=shard_start,
                )
            accuracy.update(
                decode_metrics.finalize(),
                kind="output",
                relative_l2_limit=self.case.limits.output_relative_l2,
                normalized_max_limit=self.case.limits.normalized_max,
            )
        return accuracy

    def _run_timing_round(
        self,
        *,
        verify_final: bool,
        accuracy: _AccuracyAccumulator,
    ) -> float:
        private_state = _submission_call(
            "load_state",
            lambda: self.submission.load_state(
                self.context, self.static.initial_state
            ),
            synchronize=True,
        )
        total_ms = 0.0
        final_output = None
        final_key = None
        final_golden = None
        for index, (args, output) in enumerate(
            zip(self.static.append_inputs, self.append_outputs)
        ):
            if (
                verify_final
                and not self.case.decode_steps
                and index == self.case.append_calls - 1
            ):
                output.fill_(float("nan"))
            total_ms += self._time_call(
                "append_chunk",
                lambda args=args, output=output: self.submission.append_chunk(
                    self.context, private_state, args, output
                )
            )
            final_output = output
            final_key = append_result_key(index)
            final_golden = self.goldens.append_output[final_key]

        if self.case.decode_steps:
            assert self.decode_output is not None and self.decode_buffers is not None
            for shard_start in range(
                0, self.case.decode_steps, SHARD_STEPS
            ):
                count = min(
                    SHARD_STEPS,
                    self.case.decode_steps - shard_start,
                )
                self.generator.fill_decode_shard(
                    self.decode_buffers, shard_start, count
                )
                for local_index in range(count):
                    token_index = shard_start + local_index
                    token = self.generator.select_current_token(
                        self.decode_buffers, token_index
                    )
                    if verify_final and token_index + 1 == self.case.decode_steps:
                        self.decode_output.fill_(float("nan"))
                    total_ms += self._time_call(
                        "decode_step",
                        lambda token=token: self.submission.decode_step(
                            self.context,
                            private_state,
                            token,
                            self.decode_output,
                        )
                    )
            final_output = self.decode_output
            final_key = decode_result_key(self.case.decode_steps)
            if verify_final:
                assert self.decode_golden_shard is not None
                final_shard_start = (
                    (self.case.decode_steps - 1)
                    // SHARD_STEPS
                    * SHARD_STEPS
                )
                final_shard_count = self.case.decode_steps - final_shard_start
                final_shard = load_decode_output_shard(
                    self.goldens,
                    shard_start=final_shard_start,
                    count=final_shard_count,
                    destination=self.decode_golden_shard,
                )
                final_golden = final_shard[-1]

        if not math.isfinite(total_ms) or total_ms <= 0:
            raise JudgeFailure("RUNTIME_ERROR", "CUDA Event time is not positive")
        if verify_final:
            assert (
                final_output is not None
                and final_key is not None
                and final_golden is not None
            )
            self._check_output(
                final_output,
                final_golden,
                accuracy,
                sequence_lengths=(
                    self.case.append_lengths[-1]
                    if not self.case.decode_steps
                    else None
                ),
            )
            self._export_and_check(private_state, final_key, accuracy)
        return total_ms

    def _time_call(self, action: str, call: Callable[[], None]) -> float:
        stream = torch.cuda.current_stream()
        try:
            self.start_event.record(stream)
            call()
            self.end_event.record(stream)
            self.end_event.synchronize()
            return float(self.start_event.elapsed_time(self.end_event))
        except torch.OutOfMemoryError as error:
            raise JudgeFailure(
                "OOM", f"submission.{action} ran out of memory"
            ) from error
        except Exception as error:
            raise JudgeFailure(
                "RUNTIME_ERROR",
                f"submission.{action} failed: {type(error).__name__}: {error}",
            ) from error

    def _check_output(
        self,
        actual: torch.Tensor,
        golden: torch.Tensor,
        accuracy: _AccuracyAccumulator,
        *,
        sequence_lengths: tuple[int, ...] | None = None,
    ) -> None:
        metrics = compare_tensor(
            actual,
            golden,
            required_dtype=torch.bfloat16,
            sequence_lengths=sequence_lengths,
        )
        accuracy.update(
            metrics,
            kind="output",
            relative_l2_limit=self.case.limits.output_relative_l2,
            normalized_max_limit=self.case.limits.normalized_max,
        )

    def _export_and_check(
        self,
        private_state: Any,
        key: str,
        accuracy: _AccuracyAccumulator,
    ) -> None:
        self.export_buffer.fill_(float("nan"))
        _submission_call(
            "export_state",
            lambda: self.submission.export_state(
                self.context, private_state, self.export_buffer
            ),
            synchronize=True,
        )
        metrics = compare_tensor(
            self.export_buffer,
            self.goldens.state[key],
            required_dtype=torch.float32,
        )
        accuracy.update(
            metrics,
            kind="state",
            relative_l2_limit=self.case.limits.state_relative_l2,
            normalized_max_limit=self.case.limits.normalized_max,
        )

def environment_snapshot() -> Mapping[str, Any]:
    query = (
        "name,temperature.gpu,clocks.current.graphics,clocks.current.memory,"
        "clocks.max.graphics,clocks.max.memory,power.limit,pstate"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise RuntimeError("expected exactly one GPU row")
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 8:
            raise RuntimeError("unexpected nvidia-smi GPU row")
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = {
            int(row.strip())
            for row in process_result.stdout.splitlines()
            if row.strip().isdigit()
        }
        return {
            "name": fields[0],
            "temperature_c": float(fields[1]),
            "graphics_clock_mhz": float(fields[2]),
            "memory_clock_mhz": float(fields[3]),
            "max_graphics_clock_mhz": float(fields[4]),
            "max_memory_clock_mhz": float(fields[5]),
            "power_limit_w": float(fields[6]),
            "pstate": fields[7],
            "other_compute_pids": sorted(pids - {os.getpid()}),
        }
    except Exception as error:
        raise JudgeFailure("JUDGE_ERROR", "cannot inspect the GPU environment") from error


def validate_environment(
    manifest: SimpleManifest,
    snapshot: Mapping[str, Any],
    *,
    end: bool,
) -> None:
    expected = manifest.environment
    temperature_limit = float(
        expected["max_end_temperature_c" if end else "max_start_temperature_c"]
    )
    graphics_clock_target = float(expected["required_graphics_clock_mhz"])
    graphics_clock_tolerance = float(expected["graphics_clock_tolerance_mhz"])
    memory_clock_target = float(expected["required_memory_clock_mhz"])
    memory_clock_tolerance = float(expected["memory_clock_tolerance_mhz"])
    failures = []
    if str(expected["gpu_name_contains"]) not in str(snapshot["name"]):
        failures.append("GPU model")
    if snapshot["temperature_c"] > temperature_limit:
        failures.append("GPU temperature")
    if len(snapshot["other_compute_pids"]) > int(expected["other_compute_processes"]):
        failures.append("other compute process")
    if not math.isclose(
        float(snapshot["power_limit_w"]),
        float(expected["required_power_limit_w"]),
        rel_tol=0.0,
        abs_tol=1.0,
    ):
        failures.append("power limit")
    if float(snapshot["max_graphics_clock_mhz"]) < graphics_clock_target:
        failures.append("graphics clock capability")
    if float(snapshot["max_memory_clock_mhz"]) < memory_clock_target:
        failures.append("memory clock capability")
    if not math.isclose(
        float(snapshot["graphics_clock_mhz"]),
        graphics_clock_target,
        rel_tol=0.0,
        abs_tol=graphics_clock_tolerance,
    ):
        failures.append("current graphics clock")
    if not math.isclose(
        float(snapshot["memory_clock_mhz"]),
        memory_clock_target,
        rel_tol=0.0,
        abs_tol=memory_clock_tolerance,
    ):
        failures.append("current memory clock")
    pstate = str(snapshot["pstate"])
    required_pstate = str(expected["required_pstate"])
    # A locked Ada GPU can enter P2 between the last kernel and nvidia-smi.
    # Start-of-run qualification remains strict; post-case snapshots also
    # require the frozen clocks/power checks above before accepting idle P2.
    if (not end and pstate != required_pstate) or (
        end and pstate not in {required_pstate, "P2"}
    ):
        failures.append("performance state")
    if failures:
        raise JudgeFailure(
            "JUDGE_ERROR", "unqualified GPU environment: " + ", ".join(failures)
        )


def _result_mapping(
    *,
    status: str,
    wall_time_sec: float,
    results: list[CaseResult],
    start_environment: Mapping[str, Any] | None,
    end_environment: Mapping[str, Any] | None,
    message: str | None = None,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "status": status,
        "wall_time_sec": wall_time_sec,
        "cases": {
            item.case_id: {
                "weight": item.weight,
                "timing_replays": item.timing_replays,
                "case_gpu_time_ms": item.case_gpu_time_ms,
                "accuracy_ratio": item.accuracy.max_limit_ratio,
                "worst_sequence_relative_l2_report_only": (
                    item.accuracy.worst_sequence_relative_l2
                ),
                "worst_head_relative_l2_report_only": (
                    item.accuracy.worst_head_relative_l2
                ),
                "min_sequence_reference_norm": (
                    item.accuracy.min_sequence_reference_norm
                ),
                "min_head_reference_norm": item.accuracy.min_head_reference_norm,
            }
            for item in results
        },
        "environment": {
            "start": start_environment,
            "end": end_environment,
        },
    }
    if results:
        mapping["accuracy_ratio"] = max(
            item.accuracy.max_limit_ratio for item in results
        )
    if status == "PASS":
        mapping["weighted_gpu_time_ms"] = weighted_gpu_time_ms(
            (item.weight, item.case_gpu_time_ms) for item in results
        )
    if message is not None:
        mapping["message"] = message
    return mapping


def _print_report(result: Mapping[str, Any]) -> None:
    cases = result.get("cases", {})
    if cases:
        print("CASE     REPLAYS  CASE_GPU_MS  ACC_RATIO", flush=True)
        for case_id, row in cases.items():
            print(
                f"{case_id:<8} {row['timing_replays']:>7d}  "
                f"{row['case_gpu_time_ms']:>11.3f}  "
                f"{row['accuracy_ratio']:>9.4f}",
                flush=True,
            )
    print(
        "Validation: " + ("PASS" if result["status"] == "PASS" else result["status"]),
        flush=True,
    )
    if "weighted_gpu_time_ms" in result:
        print(f"WeightedGpuTimeMs: {result['weighted_gpu_time_ms']:.6f}", flush=True)
    print(f"WallTimeSec: {result['wall_time_sec']:.2f}", flush=True)
    print(
        "RESULT_JSON="
        + json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )


def main() -> int:
    started = time.perf_counter()
    results: list[CaseResult] = []
    start_environment = None
    end_environment = None
    try:
        manifest = load_manifest(MANIFEST_PATH, require_canaries=True)
        verify_golden_reference(manifest, data_dir=DATA_DIR)
        expected_gpu_count = int(manifest.environment["gpu_count"])
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != expected_gpu_count
        ):
            raise JudgeFailure(
                "JUDGE_ERROR",
                f"Judge requires exactly {expected_gpu_count} CUDA GPU",
            )
        start_environment = environment_snapshot()
        validate_environment(manifest, start_environment, end=False)
        try:
            luts = load_generator_luts(DATA_DIR / manifest.files["generator_luts"])
            verify_counter_canaries(manifest, luts, device="cuda")
        except Exception as error:
            raise JudgeFailure(
                "JUDGE_ERROR", "cannot initialize the frozen counter generator"
            ) from error
        try:
            from submission import Submission as SubmissionClass
        except torch.OutOfMemoryError as error:
            raise JudgeFailure("OOM", "submission import ran out of memory") from error
        except Exception as error:
            raise JudgeFailure(
                "RUNTIME_ERROR",
                f"cannot import submission.Submission: {type(error).__name__}: {error}",
            ) from error
        submission = _submission_call("constructor", SubmissionClass)
        for frozen in manifest.cases:
            try:
                evaluator = CaseEvaluator(submission, manifest, frozen, luts)
                result = evaluator.evaluate()
                results.append(result)
                case_environment = environment_snapshot()
                try:
                    validate_environment(manifest, case_environment, end=True)
                except JudgeFailure:
                    end_environment = case_environment
                    raise
                print(
                    f"{frozen.case_id}: validation PASS, "
                    f"CaseGpuTimeMs={result.case_gpu_time_ms:.3f} "
                    f"over {result.timing_replays} replays",
                    flush=True,
                )
            finally:
                if "evaluator" in locals():
                    del evaluator
                gc.collect()
                torch.cuda.empty_cache()
        end_environment = environment_snapshot()
        validate_environment(manifest, end_environment, end=True)
        result_mapping = _result_mapping(
            status="PASS",
            wall_time_sec=time.perf_counter() - started,
            results=results,
            start_environment=start_environment,
            end_environment=end_environment,
        )
        _print_report(result_mapping)
        return 0
    except JudgeFailure as error:
        status = error.status
        message = str(error)
    except torch.OutOfMemoryError as error:
        status = "JUDGE_ERROR"
        message = str(error)
    except TensorCheckError as error:
        status = "WRONG_ANSWER"
        message = str(error)
    except (DataError, ManifestError) as error:
        status = "JUDGE_ERROR"
        message = str(error)
    except Exception as error:
        status = "JUDGE_ERROR"
        message = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    result_mapping = _result_mapping(
        status=status,
        wall_time_sec=time.perf_counter() - started,
        results=results,
        start_environment=start_environment,
        end_environment=end_environment,
        message=message,
    )
    _print_report(result_mapping)
    return 2 if status == "JUDGE_ERROR" else 1


__all__ = [
    "AccuracySummary",
    "CaseEvaluator",
    "CaseResult",
    "JudgeFailure",
    "environment_snapshot",
    "main",
    "validate_environment",
]
