"""Frozen data loading and integrity verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .contract import CaseSpec
from .manifest import Manifest


class DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaseData:
    strings: np.ndarray
    lengths: np.ndarray
    golden_indices: np.ndarray


def expected_payloads(manifest: Manifest) -> tuple[str, ...]:
    return ("cases.json",) + tuple(
        f"{case.case_id}.npz" for case in manifest.cases
    )


def verify_sha256(manifest: Manifest, data_dir: Path) -> None:
    checksum_path = data_dir / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise DataError("cannot read data/SHA256SUMS") from error
    parsed = {}
    for line in lines:
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise DataError("malformed SHA256SUMS line")
        digest, relative = parts
        if relative in parsed or Path(relative).name != relative:
            raise DataError("SHA256SUMS contains duplicate or nested paths")
        parsed[relative] = digest.lower()
    expected = set(expected_payloads(manifest))
    if set(parsed) != expected:
        raise DataError("SHA256SUMS payload set does not match cases.json")
    for relative in sorted(expected):
        path = data_dir / relative
        try:
            actual = _sha256(path)
        except OSError as error:
            raise DataError(f"cannot read data/{relative}") from error
        if actual != parsed[relative]:
            raise DataError(f"SHA-256 mismatch for data/{relative}")


def load_case(data_dir: Path, case: CaseSpec) -> CaseData:
    filename = f"{case.case_id}.npz"
    path = data_dir / filename
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"strings", "lengths", "golden_indices"}:
                raise DataError(f"{filename} has unexpected arrays")
            strings = np.ascontiguousarray(archive["strings"])
            lengths = np.ascontiguousarray(archive["lengths"])
            golden = np.ascontiguousarray(archive["golden_indices"])
    except (OSError, ValueError) as error:
        raise DataError(f"cannot load data/{filename}") from error

    if strings.shape != (case.n, case.width) or strings.dtype != np.uint8:
        raise DataError(f"{case.case_id} strings shape or dtype mismatch")
    if lengths.shape != (case.n,) or lengths.dtype != np.int32:
        raise DataError(f"{case.case_id} lengths shape or dtype mismatch")
    if golden.shape != (case.n,) or golden.dtype != np.int32:
        raise DataError(f"{case.case_id} golden shape or dtype mismatch")
    if np.any(lengths < 0) or np.any(lengths > case.width):
        raise DataError(f"{case.case_id} has invalid string lengths")

    columns = np.arange(case.width, dtype=np.int32)[None, :]
    valid = columns < lengths[:, None]
    if np.any(strings[valid] < 33) or np.any(strings[valid] > 126):
        raise DataError(f"{case.case_id} valid bytes must be printable ASCII")
    if np.any(strings[~valid] != 0):
        raise DataError(f"{case.case_id} padding bytes must be zero")
    if not np.array_equal(np.sort(golden.astype(np.int64)), np.arange(case.n)):
        raise DataError(f"{case.case_id} golden output is not a permutation")
    return CaseData(strings=strings, lengths=lengths, golden_indices=golden)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CaseData",
    "DataError",
    "expected_payloads",
    "load_case",
    "verify_sha256",
]
