"""Evaluate clean-trained CSI HAR checkpoints under reproducible impairments."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .corruptions import ImpairmentConfig, impair_regular_tensor
from .learning import CLASS_NAMES, create_model, metrics_from_outputs

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_impairment_scenarios(path: Path) -> dict[str, ImpairmentConfig]:
    with path.open("rb") as handle:
        sections = tomllib.load(handle)
    return {name: ImpairmentConfig(**values) for name, values in sections.items()}


def deterministic_sample_seed(sample_id: str, scenario: str, replicate_seed: int) -> int:
    payload = f"{replicate_seed}\0{scenario}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class ImpairedCacheDataset(Dataset):
    def __init__(
        self,
        index_path: Path,
        cache_root: Path,
        scenario_name: str,
        impairment: ImpairmentConfig,
        replicate_seed: int,
        sample_rate_hz: float = 100.0,
        limit: int | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.scenario_name = scenario_name
        self.impairment = impairment
        self.replicate_seed = replicate_seed
        self.sample_rate_hz = sample_rate_hz
        with Path(index_path).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            per_class = max(1, limit // len(CLASS_NAMES))
            balanced: list[dict[str, Any]] = []
            for label_index in range(len(CLASS_NAMES)):
                class_rows = [
                    row for row in self.rows if int(row["label_index"]) == label_index
                ]
                balanced.extend(class_rows[:per_class])
            self.rows = balanced

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.cache_root / Path(row["cache_relative_path"])
        with np.load(path, allow_pickle=False) as cached:
            values = np.asarray(cached["x"], dtype=np.float32)
            label = int(cached["label"].item())
        seed = deterministic_sample_seed(
            str(row["sample_id"]), self.scenario_name, self.replicate_seed
        )
        impaired_config = ImpairmentConfig(**{**asdict(self.impairment), "seed": seed})
        recovered, diagnostics = impair_regular_tensor(
            values, impaired_config, sample_rate_hz=self.sample_rate_hz
        )
        quality = torch.tensor(
            [
                diagnostics["observed_fraction"],
                diagnostics["stale_fraction"],
                diagnostics["mean_abs_timing_offset_seconds"],
                diagnostics["max_abs_timing_offset_seconds"],
            ],
            dtype=torch.float64,
        )
        return (
            torch.from_numpy(recovered),
            torch.tensor(label, dtype=torch.long),
            quality,
        )


def evaluate_scenarios(
    cache_root: Path,
    test_index: Path,
    checkpoint_path: Path,
    impairment_path: Path,
    output_path: Path,
    scenarios: list[str] | None = None,
    replicate_seed: int = 2026,
    sample_rate_hz: float = 100.0,
    batch_size: int = 64,
    num_workers: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = str(checkpoint["model_name"])
    dropout = float(checkpoint["training_config"]["dropout"])
    calibration_bins = int(checkpoint["training_config"]["calibration_bins"])
    model = create_model(model_name, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    all_scenarios = load_impairment_scenarios(impairment_path)
    selected_names = scenarios or list(all_scenarios)
    unknown = sorted(set(selected_names) - set(all_scenarios))
    if unknown:
        raise ValueError(f"Unknown impairment scenarios: {unknown}")

    results: dict[str, Any] = {}
    with torch.no_grad():
        for name in selected_names:
            dataset = ImpairedCacheDataset(
                test_index,
                cache_root,
                name,
                all_scenarios[name],
                replicate_seed,
                sample_rate_hz=sample_rate_hz,
                limit=limit,
            )
            loader = DataLoader(
                dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=False,
                num_workers=num_workers,
            )
            logits_parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []
            quality_parts: list[torch.Tensor] = []
            for values, labels, quality in loader:
                logits_parts.append(model(values).cpu())
                label_parts.append(labels)
                quality_parts.append(quality)
            quality = torch.cat(quality_parts).mean(dim=0)
            results[name] = {
                "impairment": asdict(all_scenarios[name]),
                "sample_count": len(dataset),
                "realized_quality_mean": {
                    "observed_fraction": float(quality[0]),
                    "stale_fraction": float(quality[1]),
                    "mean_abs_timing_offset_seconds": float(quality[2]),
                    "max_abs_timing_offset_seconds": float(quality[3]),
                },
                "metrics": metrics_from_outputs(
                    torch.cat(logits_parts),
                    torch.cat(label_parts),
                    bins=calibration_bins,
                ),
            }

    payload = {
        "schema_version": 1,
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "replicate_seed": replicate_seed,
        "sample_rate_hz": sample_rate_hz,
        "scenarios": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--impairments-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--replicate-seed", type=int, default=2026)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = evaluate_scenarios(
        args.cache_root,
        args.test_index,
        args.checkpoint,
        args.impairments_config,
        args.output,
        scenarios=args.scenarios,
        replicate_seed=args.replicate_seed,
        sample_rate_hz=args.sample_rate_hz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
