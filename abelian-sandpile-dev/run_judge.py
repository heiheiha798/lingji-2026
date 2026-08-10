#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import pathlib
import platform
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build"
DEFAULT_SOLUTION = ROOT / "solution"
LIBRARY_NAME = "libsandpile_submission.so"
FIXED_CASES = [
    {"name": "tiny-correctness", "limit": 60},
    {"name": "global-rain", "limit": 60},
    {"name": "local-avalanche", "limit": 100},
    {"name": "many-sources", "limit": 140},
    {"name": "near-critical", "limit": 280},
    {"name": "large-mixed", "limit": 480},
]
RANDOM_CASE_COUNT = 5


def find_nvcc() -> str:
    requested = os.environ.get("NVCC")
    candidates = [requested, shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("nvcc not found; set NVCC or add CUDA to PATH")


def build_judge(nvcc: str) -> pathlib.Path:
    BUILD.mkdir(exist_ok=True)
    binary = BUILD / "judge"
    command = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-arch=sm_89",
        "-lineinfo",
        "-I",
        str(ROOT / "interface"),
        "-I",
        str(ROOT / "judge"),
        str(ROOT / "judge" / "runner.cu"),
        str(ROOT / "judge" / "reference.cu"),
        "-ldl",
        "-o",
        str(binary),
    ]
    print("[build judge]", " ".join(command))
    subprocess.run(command, check=True)
    return binary


def build_solution(solution_dir: pathlib.Path, nvcc: str) -> pathlib.Path:
    make = shutil.which("make")
    if not make:
        raise RuntimeError("make not found")
    output_dir = (BUILD / "submission").resolve()
    command = [
        make,
        "-B",
        "-C",
        str(solution_dir),
        f"BUILD_DIR={output_dir}",
        f"API_DIR={(ROOT / 'interface').resolve()}",
        f"NVCC={nvcc}",
    ]
    print("[build solution]", " ".join(command))
    subprocess.run(command, check=True)
    library = output_dir / LIBRARY_NAME
    if not library.is_file():
        raise RuntimeError(f"solution build did not produce {library}")
    return library


def parse_result(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError("judge produced no RESULT line")


def run_once(
    binary: pathlib.Path,
    library: pathlib.Path,
    case_id: int,
    timeout: int,
    random_seed: int,
    mode: str = "candidate",
    output_path: pathlib.Path | None = None,
) -> dict:
    environment = os.environ.copy()
    environment["SANDPILE_RANDOM_SEED"] = str(random_seed)
    completed = subprocess.run(
        [
            str(binary),
            str(case_id),
            str(library),
            mode,
            str(output_path) if output_path is not None else "-",
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=environment,
    )
    result = parse_result(completed.stdout)
    result["stderr"] = completed.stderr.strip()
    return result


def files_equal(left: pathlib.Path, right: pathlib.Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_block = left_file.read(1 << 20)
            right_block = right_file.read(1 << 20)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def run_performance_case(
    binary: pathlib.Path,
    library: pathlib.Path,
    case_id: int,
    timeout: int,
    repeats: int,
    random_seed: int,
) -> dict:
    samples = []
    reference_samples = []
    last_candidate = None
    with tempfile.TemporaryDirectory(prefix="sandpile-case-", dir=BUILD) as temp:
        temporary = pathlib.Path(temp)
        golden_path = temporary / "golden.bin"
        candidate_path = temporary / "candidate.bin"
        for repeat in range(repeats):
            result = run_once(
                binary,
                library,
                case_id,
                timeout,
                random_seed,
                "reference",
                golden_path if repeat == 0 else None,
            )
            if not result.get("correct"):
                raise RuntimeError("reference task failed")
            reference_samples.append(float(result["reference_ms"]))
        for _ in range(repeats):
            result = run_once(
                binary,
                library,
                case_id,
                timeout,
                random_seed,
                "candidate",
                candidate_path,
            )
            last_candidate = result
            if result.get("correct"):
                result["correct"] = files_equal(candidate_path, golden_path)
                if not result["correct"]:
                    result["error"] = "candidate output differs from reference"
            if not result.get("correct"):
                result["samples_ms"] = samples
                result["reference_samples_ms"] = reference_samples
                return result
            samples.append(float(result["time_ms"]))
    assert last_candidate is not None
    last_candidate["time_ms"] = statistics.median(samples)
    last_candidate["samples_ms"] = samples
    last_candidate["reference_ms"] = statistics.median(reference_samples)
    last_candidate["reference_samples_ms"] = reference_samples
    return last_candidate


def collect_environment(nvcc: str | None) -> dict:
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvcc": None,
        "gpu": None,
    }
    if nvcc:
        completed = subprocess.run(
            [nvcc, "--version"], text=True, capture_output=True, check=False
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if lines:
            environment["nvcc"] = lines[-1]
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout.strip():
            environment["gpu"] = completed.stdout.strip()
    return environment


def functional_record(result: dict) -> dict:
    return {
        "name": result.get("name"),
        "status": "PASS" if result.get("correct") else "FAIL",
        "correct": bool(result.get("correct")),
        "rows": result.get("rows"),
        "cols": result.get("cols"),
        "seed": result.get("seed"),
    }


def performance_record(case_id: int, result: dict) -> dict:
    passed = bool(result.get("correct"))
    elapsed = float(result["time_ms"]) if passed else None
    reference = float(result["reference_ms"]) if passed else None
    speedup = reference / elapsed if passed and elapsed and elapsed > 0 else None
    return {
        "case_id": case_id,
        "name": result.get("name", FIXED_CASES[case_id]["name"]),
        "status": "PASS" if passed else "FAIL",
        "correct": passed,
        "time_ms": elapsed,
        "samples_ms": result.get("samples_ms", []),
        "reference_ms": reference,
        "reference_samples_ms": result.get("reference_samples_ms", []),
        "speedup": speedup,
        "seed": result.get("seed"),
        "guards_intact": result.get("guards_intact"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and judge an Abelian sandpile solution project"
    )
    parser.add_argument("--solution-dir", type=pathlib.Path, default=DEFAULT_SOLUTION)
    parser.add_argument("--case", type=int, action="append", dest="case_ids")
    parser.add_argument("--quick", action="store_true", help="random checks and cases 0-2")
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        help="write correctness and fixed-case performance to this JSON file",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.random_only and args.skip_random:
        parser.error("--random-only and --skip-random cannot be combined")

    solution_dir = args.solution_dir.resolve()
    if not (solution_dir / "Makefile").is_file():
        parser.error(f"solution Makefile does not exist: {solution_dir / 'Makefile'}")

    nvcc = find_nvcc()
    binary = BUILD / "judge"
    library = BUILD / "submission" / LIBRARY_NAME
    if not args.no_build:
        binary = build_judge(nvcc)
        library = build_solution(solution_dir, nvcc)
    if not binary.is_file() or not library.is_file():
        parser.error("built judge or submission library is missing")

    if args.random_only:
        fixed_ids = []
    elif args.case_ids:
        fixed_ids = args.case_ids
    elif args.quick:
        fixed_ids = [0, 1, 2]
    else:
        fixed_ids = list(range(len(FIXED_CASES)))
    if any(case_id < 0 or case_id >= len(FIXED_CASES) for case_id in fixed_ids):
        parser.error("case id out of range")

    run_random = not args.skip_random and not args.case_ids
    random_seed = args.random_seed
    if random_seed is None:
        random_seed = secrets.randbits(63)

    all_passed = True
    functional_results = []
    if run_random:
        print("\nfunctional correctness")
        print("-" * 68)
        for index in range(RANDOM_CASE_COUNT):
            case_id = len(FIXED_CASES) + index
            try:
                with tempfile.TemporaryDirectory(
                    prefix="sandpile-random-", dir=BUILD
                ) as temp:
                    temporary = pathlib.Path(temp)
                    golden_path = temporary / "golden.bin"
                    candidate_path = temporary / "candidate.bin"
                    reference = run_once(
                        binary,
                        library,
                        case_id,
                        90,
                        random_seed,
                        "reference",
                        golden_path,
                    )
                    result = run_once(
                        binary,
                        library,
                        case_id,
                        90,
                        random_seed,
                        "candidate",
                        candidate_path,
                    )
                    result["correct"] = bool(
                        reference.get("correct")
                        and result.get("correct")
                        and files_equal(candidate_path, golden_path)
                    )
                functional_results.append(functional_record(result))
                all_passed &= bool(result.get("correct"))
            except subprocess.TimeoutExpired:
                functional_results.append(
                    {"name": f"random-{index}", "status": "TIMEOUT", "correct": False}
                )
                all_passed = False
            except (RuntimeError, json.JSONDecodeError) as error:
                functional_results.append(
                    {
                        "name": f"random-{index}",
                        "status": "ERROR",
                        "correct": False,
                        "error": str(error),
                    }
                )
                all_passed = False
        functional_passed = all(item["correct"] for item in functional_results)
        print(
            f"random-correctness    {'PASS' if functional_passed else 'FAIL':^8} "
            f"seed={random_seed}  patterns={RANDOM_CASE_COUNT}"
        )
        if not functional_passed:
            for item in functional_results:
                if not item["correct"]:
                    print(" ", json.dumps(item, ensure_ascii=False))

    performance_results = []
    if fixed_ids:
        print("\nfixed performance cases")
        print("case                 status      time(ms)    ref(ms)    speedup   runs")
        print("-" * 76)
        for case_id in fixed_ids:
            case = FIXED_CASES[case_id]
            try:
                result = run_performance_case(
                    binary,
                    library,
                    case_id,
                    case["limit"],
                    args.repeats,
                    random_seed,
                )
                record = performance_record(case_id, result)
                performance_results.append(record)
                all_passed &= record["correct"]
                if record["correct"]:
                    print(
                        f"{record['name']:<20} {'PASS':^8} {record['time_ms']:>12.3f} "
                        f"{record['reference_ms']:>11.3f} {record['speedup']:>9.3f}x "
                        f"{len(record['samples_ms']):>5}"
                    )
                else:
                    print(f"{record['name']:<20} {'FAIL':^8} {'-':>12} {'-':>11} {'-':>10} {'-':>5}")
                    print(" ", json.dumps(result, ensure_ascii=False))
                if result.get("stderr"):
                    print("  stderr:", result["stderr"])
            except subprocess.TimeoutExpired:
                all_passed = False
                performance_results.append(
                    {
                        "case_id": case_id,
                        "name": case["name"],
                        "status": "TIMEOUT",
                        "correct": False,
                    }
                )
                print(f"{case['name']:<20} {'TIMEOUT':^8} {'-':>12} {'-':>11} {'-':>10} {'-':>5}")
            except (RuntimeError, json.JSONDecodeError) as error:
                all_passed = False
                performance_results.append(
                    {
                        "case_id": case_id,
                        "name": case["name"],
                        "status": "ERROR",
                        "correct": False,
                        "error": str(error),
                    }
                )
                print(f"{case['name']:<20} {'ERROR':^8} {'-':>12} {'-':>11} {'-':>10} {'-':>5}")
                print(" ", error)
        print("-" * 76)

    print("overall:", "PASS" if all_passed else "FAIL")
    if args.json_output:
        report_path = args.json_output.resolve()
        report = {
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "solution_dir": str(solution_dir),
            "command": [sys.executable, *sys.argv],
            "environment": collect_environment(nvcc),
            "functional": {
                "status": (
                    "SKIPPED"
                    if not run_random
                    else "PASS"
                    if all(item["correct"] for item in functional_results)
                    else "FAIL"
                ),
                "seed": random_seed if run_random else None,
                "cases": functional_results,
            },
            "performance": performance_results,
            "overall": "PASS" if all_passed else "FAIL",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("json report:", report_path)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
