"""Run a lightweight real-data smoke test across supported CSI formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .corruptions import ImpairmentConfig, simulate_impairments
from .inventory import load_config
from .readers import (
    CSISample,
    read_csi_bench_h5,
    read_csi_bench_mat,
    read_xrf55_dat,
    read_xrf55_mat,
)


def _summarize(sample: CSISample) -> dict[str, Any]:
    import numpy as np

    intervals = np.diff(sample.timestamps)
    positive = intervals[intervals > 0]
    return {
        "shape": list(sample.values.shape),
        "dtype": str(sample.values.dtype),
        "duration_seconds": float(sample.timestamps[-1]) if sample.timestamps.size else 0.0,
        "median_delta_t_seconds": float(np.median(positive)) if positive.size else None,
        "finite_amplitude": bool(np.isfinite(sample.amplitude).all()),
        "format": sample.metadata.get("format"),
    }


def run_smoke(config_path: Path) -> dict[str, Any]:
    csi_root, xrf_root = load_config(config_path)
    candidates: list[tuple[str, Callable[[Path], CSISample], Path]] = [
        (
            "csi_bench_h5",
            read_csi_bench_h5,
            next((csi_root / "Multitask" / "sub_Human_h5").rglob("*.h5")),
        ),
        (
            "csi_bench_mat",
            read_csi_bench_mat,
            next((csi_root / "Multitask" / "sub_Human_mat").rglob("*.mat")),
        ),
        ("xrf55_dat", read_xrf55_dat, next(xrf_root.rglob("*.dat"))),
        ("xrf55_mat", read_xrf55_mat, next(xrf_root.rglob("*.mat"))),
    ]
    config = ImpairmentConfig(
        random_loss_probability=0.10,
        burst_start_probability=0.01,
        burst_length=3,
        jitter_std_seconds=0.0005,
        stale_start_probability=0.02,
        stale_length=3,
        seed=2026,
    )
    results: dict[str, Any] = {}
    for name, reader, path in candidates:
        sample = reader(path)
        impaired = simulate_impairments(sample, config)
        results[name] = {
            "path": str(path),
            **_summarize(sample),
            "observed_fraction": impaired.metadata["observed_fraction"],
            "stale_fraction": impaired.metadata["stale_fraction"],
            "quality_feature_shape": list(impaired.quality_features.shape),
        }
    return {"schema_version": 1, "impairment_config": config.__dict__, "samples": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_smoke(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

