"""Evaluate augmented and quality-aware CNN-LSTM checkpoints under impairments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .learning import CNNLSTMBaseline, metrics_from_outputs
from .quality_learning import (
    QUALITY_FEATURE_NAMES,
    AugmentedCacheDataset,
    GatedQualityCNNLSTM,
    QualityAwareCNNLSTM,
    _forward,
)
from .robustness import load_impairment_scenarios


def _create_quality_model(model_name: str, dropout: float) -> tuple[nn.Module, bool]:
    if model_name == "quality_cnn_lstm":
        return QualityAwareCNNLSTM(dropout=dropout), True
    if model_name == "gated_quality_cnn_lstm":
        return GatedQualityCNNLSTM(dropout=dropout), True
    if model_name == "augmented_cnn_lstm":
        return CNNLSTMBaseline(dropout=dropout), False
    raise ValueError(f"Unsupported quality-trained model: {model_name}")


def evaluate_quality_scenarios(
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
    training_config = checkpoint["training_config"]
    model, uses_quality = _create_quality_model(
        model_name, dropout=float(training_config["dropout"])
    )
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
            dataset = AugmentedCacheDataset(
                test_index,
                cache_root,
                all_scenarios,
                [name],
                replicate_seed,
                "test",
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
            quality_sum = torch.zeros(len(QUALITY_FEATURE_NAMES), dtype=torch.float64)
            quality_count = 0
            for values, quality, labels in loader:
                logits_parts.append(_forward(model, values, quality, uses_quality).cpu())
                label_parts.append(labels)
                quality_sum += quality.to(torch.float64).sum(dim=(0, 1))
                quality_count += quality.shape[0] * quality.shape[1]
            quality_mean = quality_sum / quality_count
            results[name] = {
                "impairment": asdict(all_scenarios[name]),
                "sample_count": len(dataset),
                "quality_feature_mean": {
                    feature: float(quality_mean[index])
                    for index, feature in enumerate(QUALITY_FEATURE_NAMES)
                },
                "metrics": metrics_from_outputs(
                    torch.cat(logits_parts),
                    torch.cat(label_parts),
                    bins=int(training_config["calibration_bins"]),
                ),
            }

    payload = {
        "schema_version": 1,
        "model": model_name,
        "uses_quality": uses_quality,
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
    result = evaluate_quality_scenarios(
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
