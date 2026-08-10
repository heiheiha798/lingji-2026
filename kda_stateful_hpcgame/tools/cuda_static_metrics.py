#!/usr/bin/env python3
"""Compile generated CUDA for SM89 and retain ptxas/SASS metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile one generated CUDA source and retain SM89 static metrics."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-compiler", type=Path, required=True)
    parser.add_argument("--tilelang-include", type=Path, required=True)
    parser.add_argument("--cutlass-include", type=Path, required=True)
    parser.add_argument("--dynamic-smem-bytes", type=int)
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.host_compiler.is_file():
        parser.error(f"host compiler does not exist: {args.host_compiler}")
    if not args.tilelang_include.is_dir():
        parser.error(f"TileLang include directory does not exist: {args.tilelang_include}")
    if not args.cutlass_include.is_dir():
        parser.error(f"CUTLASS include directory does not exist: {args.cutlass_include}")
    if args.dynamic_smem_bytes is not None and args.dynamic_smem_bytes < 0:
        parser.error("--dynamic-smem-bytes must be non-negative")

    cuda_home_value = os.environ.get("CUDA_HOME")
    if cuda_home_value is None:
        parser.error("CUDA_HOME must be set explicitly")
    cuda_home = Path(cuda_home_value)
    nvcc = cuda_home / "bin" / "nvcc"
    nvdisasm = cuda_home / "bin" / "nvdisasm"
    if not nvcc.is_file():
        parser.error(f"nvcc does not exist: {nvcc}")
    if not nvdisasm.is_file():
        parser.error(f"nvdisasm does not exist: {nvdisasm}")

    source = args.source.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True)
    source_path = args.output_dir / "kernel.cu"
    cubin_path = args.output_dir / "kernel.cubin"
    ptxas_path = args.output_dir / "ptxas.log"
    sass_path = args.output_dir / "kernel.sass"
    source_path.write_text(source, encoding="utf-8")

    command = [
        str(nvcc),
        f"-ccbin={args.host_compiler}",
        "--cubin",
        "-O3",
        "-lineinfo",
        "-arch=sm_89",
        "-std=c++20",
        f"-I{args.tilelang_include}",
        f"-I{args.cutlass_include}",
        "--use_fast_math",
        "--ptxas-options=--verbose",
        "-o",
        str(cubin_path),
        str(source_path),
    ]
    (args.output_dir / "nvcc.command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    nvcc_seconds = time.perf_counter() - started
    ptxas_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"NVCC resource compilation failed; see {ptxas_path}")

    started = time.perf_counter()
    completed = subprocess.run(
        [str(nvdisasm), str(cubin_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    nvdisasm_seconds = time.perf_counter() - started
    sass_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"nvdisasm failed; see {sass_path}")

    ptxas_output = ptxas_path.read_text(encoding="utf-8")
    report: dict[str, str | int | float] = {
        "target": "sm_89",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_bytes": len(source.encode("utf-8")),
        "source_lines": len(source.splitlines()),
        "cubin_sha256": hashlib.sha256(cubin_path.read_bytes()).hexdigest(),
        "cubin_bytes": cubin_path.stat().st_size,
        "nvcc_seconds": round(nvcc_seconds, 3),
        "nvdisasm_seconds": round(nvdisasm_seconds, 3),
    }
    for name, pattern in {
        "registers": r"Used (\d+) registers",
        "stack_frame_bytes": r"(\d+) bytes stack frame",
        "spill_store_bytes": r"(\d+) bytes spill stores",
        "spill_load_bytes": r"(\d+) bytes spill loads",
        "barriers": r"used (\d+) barriers",
    }.items():
        match = re.search(pattern, ptxas_output)
        if match is None:
            raise RuntimeError(f"PTXAS output did not contain {name!r}; see {ptxas_path}")
        report[name] = int(match.group(1))

    instruction_lines = [
        line
        for line in sass_path.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*/\*[0-9a-fA-F]+\*/", line)
    ]
    report["instructions"] = len(instruction_lines)
    for name, opcode in {
        "hmma": "HMMA",
        "mufu": "MUFU",
        "ffma": "FFMA",
        "branch": "BRA",
        "global_load": "LDG",
        "global_store": "STG",
        "shared_load": "LDS",
        "shared_store": "STS",
        "barrier_instructions": "BAR",
    }.items():
        report[name] = sum(
            bool(re.search(rf"\b{opcode}(?:\.|\b)", line))
            for line in instruction_lines
        )
    if args.dynamic_smem_bytes is not None:
        report["dynamic_smem_bytes"] = args.dynamic_smem_bytes

    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"registers={report['registers']} "
        f"spill={report['spill_store_bytes']}/{report['spill_load_bytes']} "
        f"instructions={report['instructions']} "
        f"cubin_bytes={report['cubin_bytes']}"
    )
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
