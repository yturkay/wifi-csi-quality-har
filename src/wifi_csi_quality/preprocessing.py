"""Canonical preprocessing and resumable cache creation for CSI samples.

Raw datasets are opened read-only. Each output is a compressed NumPy archive
containing one float32 tensor with shape ``[time, subcarrier]`` plus provenance
metadata sufficient to reproduce the transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .inventory import load_config, sha256_file
from .readers import CSISample, read_csi_bench_h5, read_csi_bench_mat, read_xrf55

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class PreprocessingConfig:
    target_sample_rate_hz: float = 100.0
    target_time_steps: int = 500
    target_subcarriers: int = 64
    link_reduction: str = "mean"
    amplitude_transform: str = "log1p"
    normalization: str = "per_sample_robust_median_iqr"
    iqr_epsilon: float = 1e-6

    def validate(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if self.target_time_steps < 2 or self.target_subcarriers < 2:
            raise ValueError("target dimensions must both be at least 2")
        if self.link_reduction != "mean":
            raise ValueError(f"Unsupported link reduction: {self.link_reduction}")
        if self.amplitude_transform != "log1p":
            raise ValueError(f"Unsupported amplitude transform: {self.amplitude_transform}")
        if self.normalization != "per_sample_robust_median_iqr":
            raise ValueError(f"Unsupported normalization: {self.normalization}")


def preprocessing_config(protocol: dict[str, Any]) -> PreprocessingConfig:
    values = protocol["preprocessing"]
    config = PreprocessingConfig(
        target_sample_rate_hz=float(values["target_sample_rate_hz"]),
        target_time_steps=int(values["target_time_steps"]),
        target_subcarriers=int(values["target_subcarriers"]),
        link_reduction=str(values["link_reduction"]),
        amplitude_transform=str(values["amplitude_transform"]),
        normalization=str(values["normalization"]),
    )
    config.validate()
    return config


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _strictly_increasing(values: Any, timestamps: Any) -> tuple[Any, Any]:
    import numpy as np

    times = np.asarray(timestamps, dtype=np.float64)
    data = np.asarray(values)
    finite = np.isfinite(times)
    if not finite.all():
        times, data = times[finite], data[finite]
    if times.size < 2:
        raise ValueError("At least two finite timestamps are required")
    order = np.argsort(times, kind="stable")
    times, data = times[order], data[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    if unique_times.size < 2:
        raise ValueError("At least two distinct timestamps are required")
    return data[unique_indices], unique_times


def transform_sample(
    sample: CSISample, config: PreprocessingConfig
) -> tuple[Any, dict[str, Any]]:
    """Convert one canonical reader sample to a normalized fixed-size tensor."""
    import numpy as np

    config.validate()
    amplitude = np.abs(np.asarray(sample.values))
    if not np.isfinite(amplitude).all():
        raise ValueError("CSI amplitude contains non-finite values")
    # Reduce links after magnitude conversion to avoid complex cancellation.
    reduced = amplitude.mean(axis=2)
    reduced, timestamps = _strictly_increasing(reduced, sample.timestamps)
    transformed = np.log1p(reduced)

    target_times = np.arange(config.target_time_steps, dtype=np.float64)
    target_times /= config.target_sample_rate_hz
    time_resampled = np.empty(
        (config.target_time_steps, transformed.shape[1]), dtype=np.float64
    )
    for subcarrier in range(transformed.shape[1]):
        time_resampled[:, subcarrier] = np.interp(
            target_times, timestamps, transformed[:, subcarrier]
        )

    source_axis = np.linspace(0.0, 1.0, time_resampled.shape[1])
    target_axis = np.linspace(0.0, 1.0, config.target_subcarriers)
    carrier_resampled = np.empty(
        (config.target_time_steps, config.target_subcarriers), dtype=np.float64
    )
    for time_index in range(config.target_time_steps):
        carrier_resampled[time_index] = np.interp(
            target_axis, source_axis, time_resampled[time_index]
        )

    # Robust normalization is fitted independently inside each sample and
    # subcarrier, so no statistics can leak between train/validation/test.
    median = np.median(carrier_resampled, axis=0, keepdims=True)
    q25, q75 = np.percentile(carrier_resampled, [25.0, 75.0], axis=0)
    iqr = np.maximum(q75 - q25, config.iqr_epsilon)[None, :]
    normalized = ((carrier_resampled - median) / iqr).astype(np.float32)

    target_end = float(target_times[-1])
    source_end = float(timestamps[-1])
    extrapolated = int(np.count_nonzero(target_times > source_end))
    diagnostics = {
        "source_time_steps": int(sample.values.shape[0]),
        "source_subcarriers": int(sample.values.shape[1]),
        "source_links": int(sample.values.shape[2]),
        "source_duration_seconds": source_end,
        "target_duration_seconds": target_end,
        "duplicate_or_invalid_timestamps_removed": int(
            sample.values.shape[0] - timestamps.size
        ),
        "right_edge_extrapolated_steps": extrapolated,
        "right_edge_extrapolated_fraction": extrapolated / config.target_time_steps,
    }
    return normalized, diagnostics


def _read_record(record: dict[str, Any], source_path: Path) -> CSISample:
    dataset = record["dataset"]
    if dataset == "CSI-Bench":
        if source_path.suffix.lower() in {".h5", ".hdf5"}:
            return read_csi_bench_h5(source_path)
        if source_path.suffix.lower() == ".mat":
            return read_csi_bench_mat(source_path)
    elif dataset == "XRF55":
        return read_xrf55(source_path)
    raise ValueError(f"Unsupported dataset or extension: {dataset}, {source_path.suffix}")


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _cache_relative_path(record: dict[str, Any], split: str) -> Path:
    digest = hashlib.sha256(record["sample_id"].encode("utf-8")).hexdigest()[:20]
    dataset = record["dataset"].lower().replace("-", "_")
    return Path(dataset) / split / record["label"] / f"{digest}.npz"


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_cache(
    manifest_path: Path,
    roots: dict[str, Path],
    output_dir: Path,
    config: PreprocessingConfig,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build or resume one manifest cache and return an audit summary."""
    import numpy as np

    split = manifest_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{split}_cache_index.jsonl"
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    processed = skipped = 0

    for position, record in enumerate(_records(manifest_path)):
        if limit is not None and position >= limit:
            break
        source_root = roots[record["dataset"]]
        source_path = source_root / Path(record["relative_path"])
        cache_relative = _cache_relative_path(record, split)
        cache_path = output_dir / cache_relative
        try:
            reuse_cache = False
            if cache_path.exists() and not overwrite:
                try:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        cached_tensor = cached["x"]
                        provenance = json.loads(str(cached["provenance_json"].item()))
                        diagnostics = json.loads(str(cached["diagnostics_json"].item()))
                    source_stat = source_path.stat()
                    reuse_cache = (
                        cached_tensor.shape
                        == (config.target_time_steps, config.target_subcarriers)
                        and cached_tensor.dtype == np.float32
                        and np.isfinite(cached_tensor).all()
                        and provenance.get("preprocessing") == asdict(config)
                        and provenance.get("source_size_bytes") == source_stat.st_size
                        and provenance.get("source_mtime_ns") == source_stat.st_mtime_ns
                    )
                    if reuse_cache:
                        shape = list(cached_tensor.shape)
                        skipped += 1
                except (KeyError, OSError, ValueError, json.JSONDecodeError):
                    reuse_cache = False
            if not reuse_cache:
                sample = _read_record(record, source_path)
                tensor, diagnostics = transform_sample(sample, config)
                provenance = {
                    "sample_id": record["sample_id"],
                    "dataset": record["dataset"],
                    "relative_path": record["relative_path"],
                    "source_size_bytes": source_path.stat().st_size,
                    "source_mtime_ns": source_path.stat().st_mtime_ns,
                    "preprocessing": asdict(config),
                }
                _atomic_save_npz(
                    cache_path,
                    x=tensor,
                    label=np.int64(record["label_index"]),
                    sample_id=np.asarray(record["sample_id"]),
                    diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
                    provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
                )
                shape = list(tensor.shape)
                processed += 1
            index_rows.append(
                {
                    **record,
                    "cache_relative_path": cache_relative.as_posix(),
                    "cache_sha256": sha256_file(cache_path),
                    "shape": shape,
                    "diagnostics": diagnostics,
                }
            )
        except Exception as exc:  # preserve progress and report every bad sample
            failures.append(
                {"sample_id": record.get("sample_id", ""), "error": repr(exc)}
            )

    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "preprocessing": asdict(config),
        "processed": processed,
        "skipped_existing": skipped,
        "successful": len(index_rows),
        "failed": len(failures),
        "failures": failures,
        "cache_index": str(index_path),
        "cache_index_sha256": sha256_file(index_path),
    }
    summary_path = output_dir / f"{split}_cache_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--protocol-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    csi_root, xrf_root = load_config(args.dataset_config)
    protocol = _load_toml(args.protocol_config)
    roots = {
        "CSI-Bench": csi_root,
        "XRF55": xrf_root,
    }
    summary = build_cache(
        manifest_path=args.manifest,
        roots=roots,
        output_dir=args.output_dir,
        config=preprocessing_config(protocol),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
