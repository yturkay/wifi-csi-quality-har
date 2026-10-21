"""Train impairment-augmented and quality-aware CNN-LSTM models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .corruptions import ImpairmentConfig, impair_regular_tensor_with_quality
from .learning import (
    CLASS_NAMES,
    CNNLSTMBaseline,
    TrainingConfig,
    load_training_config,
    metrics_from_outputs,
    set_deterministic,
)
from .robustness import deterministic_sample_seed, load_impairment_scenarios


QUALITY_FEATURE_NAMES = (
    "observed",
    "stale",
    "delta_t_ratio",
    "measurement_age_ratio",
)


class QualityAwareCNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(64 + len(QUALITY_FEATURE_NAMES), 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.recurrent = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, values: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((values, quality), dim=2)
        sequence = self.temporal(combined.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(sequence)
        return self.classifier(sequence.mean(dim=1))


class GatedQualityCNNLSTM(nn.Module):
    """Fuse CSI and packet quality with bounded feature-wise modulation."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.csi_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.quality_encoder = nn.Sequential(
            nn.Conv1d(
                len(QUALITY_FEATURE_NAMES), 32, kernel_size=7, stride=2, padding=3
            ),
            nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=1),
        )
        self.recurrent = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, values: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        csi_features = self.csi_encoder(values.transpose(1, 2))
        gamma_logits, beta_logits = self.quality_encoder(
            quality.transpose(1, 2)
        ).chunk(2, dim=1)
        gamma = 1.0 + 0.5 * torch.tanh(gamma_logits)
        beta = 0.1 * torch.tanh(beta_logits)
        sequence = (csi_features * gamma + beta).transpose(1, 2)
        sequence, _ = self.recurrent(sequence)
        return self.classifier(sequence.mean(dim=1))


def _hash_index(text: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % size


class AugmentedCacheDataset(Dataset):
    def __init__(
        self,
        index_path: Path,
        cache_root: Path,
        scenarios: dict[str, ImpairmentConfig],
        scenario_names: list[str],
        base_seed: int,
        namespace: str,
        sample_rate_hz: float = 100.0,
        limit: int | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.scenarios = scenarios
        self.scenario_names = tuple(scenario_names)
        self.base_seed = base_seed
        self.namespace = namespace
        self.sample_rate_hz = sample_rate_hz
        self.epoch = 0
        if not self.scenario_names:
            raise ValueError("At least one training scenario is required")
        unknown = sorted(set(self.scenario_names) - set(self.scenarios))
        if unknown:
            raise ValueError(f"Unknown training scenarios: {unknown}")
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> dict[int, int]:
        counts = Counter(int(row["label_index"]) for row in self.rows)
        return {index: counts.get(index, 0) for index in range(len(CLASS_NAMES))}

    def scenario_for(self, sample_id: str) -> str:
        key = f"{self.base_seed}\0{self.namespace}\0{self.epoch}\0{sample_id}"
        return self.scenario_names[_hash_index(key, len(self.scenario_names))]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.cache_root / Path(row["cache_relative_path"])
        with np.load(path, allow_pickle=False) as cached:
            values = np.asarray(cached["x"], dtype=np.float32)
            label = int(cached["label"].item())
        sample_id = str(row["sample_id"])
        scenario_name = self.scenario_for(sample_id)
        seed = deterministic_sample_seed(
            sample_id,
            scenario_name,
            self.base_seed + self.epoch * 1_000_003,
        )
        config = ImpairmentConfig(
            **{**asdict(self.scenarios[scenario_name]), "seed": seed}
        )
        recovered, quality, _ = impair_regular_tensor_with_quality(
            values, config, sample_rate_hz=self.sample_rate_hz
        )
        return (
            torch.from_numpy(recovered),
            torch.from_numpy(quality),
            torch.tensor(label, dtype=torch.long),
        )


def _class_weights(dataset: AugmentedCacheDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASS_NAMES) * max(counts[index], 1)) for index in range(4)],
        dtype=torch.float32,
    )


def _forward(
    model: nn.Module,
    values: torch.Tensor,
    quality: torch.Tensor,
    uses_quality: bool,
) -> torch.Tensor:
    if uses_quality:
        return model(values, quality)
    return model(values)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    uses_quality: bool,
    bins: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for values, quality, labels in loader:
            values = values.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            logits_parts.append(_forward(model, values, quality, uses_quality).cpu())
            label_parts.append(labels)
    return metrics_from_outputs(torch.cat(logits_parts), torch.cat(label_parts), bins=bins)


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_quality_model(
    cache_root: Path,
    train_index: Path,
    val_index: Path,
    impairment_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    model_name: str,
    scenario_names: list[str],
    sample_rate_hz: float = 100.0,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if model_name not in {
        "augmented_cnn_lstm",
        "quality_cnn_lstm",
        "gated_quality_cnn_lstm",
    }:
        raise ValueError(f"Unsupported quality-training model: {model_name}")
    set_deterministic(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_impairment_scenarios(impairment_path)
    limit = 128 if smoke else None
    train_set = AugmentedCacheDataset(
        train_index,
        cache_root,
        scenarios,
        scenario_names,
        config.seed,
        "train",
        sample_rate_hz=sample_rate_hz,
        limit=limit,
    )
    val_set = AugmentedCacheDataset(
        val_index,
        cache_root,
        scenarios,
        scenario_names,
        config.seed,
        "validation",
        sample_rate_hz=sample_rate_hz,
        limit=limit,
    )
    loader_args = {
        "batch_size": min(config.batch_size, len(train_set)),
        "num_workers": 0 if smoke else config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    uses_quality = model_name != "augmented_cnn_lstm"
    model: nn.Module
    if model_name == "quality_cnn_lstm":
        model = QualityAwareCNNLSTM(dropout=config.dropout)
    elif model_name == "gated_quality_cnn_lstm":
        model = GatedQualityCNNLSTM(dropout=config.dropout)
    else:
        model = CNNLSTMBaseline(dropout=config.dropout)
    model = model.to(device)
    weights = _class_weights(train_set) if config.class_weighting else None
    if weights is not None:
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_macro_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    epochs = 1 if smoke else config.epochs
    best_path = output_dir / f"{model_name}_best.pt"
    last_path = output_dir / f"{model_name}_last.pt"
    start_epoch = 1
    generator = torch.Generator().manual_seed(config.seed)

    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_macro_f1 = float(state["best_macro_f1"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        generator.set_state(state["loader_generator_state"])

    for epoch in range(start_epoch, epochs + 1):
        train_set.set_epoch(epoch)
        train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
        model.train()
        loss_total = 0.0
        seen = 0
        for values, quality, labels in train_loader:
            values = values.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, values, quality, uses_quality)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.shape[0]
            seen += labels.shape[0]
        validation = _evaluate(
            model, val_loader, uses_quality, config.calibration_bins, device
        )
        history.append(
            {"epoch": epoch, "train_loss": loss_total / seen, "validation": validation}
        )
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            _atomic_save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "training_config": asdict(config),
                    "class_names": CLASS_NAMES,
                    "quality_feature_names": QUALITY_FEATURE_NAMES,
                    "uses_quality": uses_quality,
                    "training_scenarios": scenario_names,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        _atomic_save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_config": asdict(config),
                "quality_feature_names": QUALITY_FEATURE_NAMES,
                "uses_quality": uses_quality,
                "training_scenarios": scenario_names,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "stale_epochs": stale_epochs,
                "history": history,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "loader_generator_state": generator.get_state(),
            },
            last_path,
        )
        if stale_epochs >= config.early_stopping_patience:
            break

    result = {
        "schema_version": 1,
        "model": model_name,
        "uses_quality": uses_quality,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "training_config": asdict(config),
        "training_scenarios": scenario_names,
        "dataset_sizes": {"train": len(train_set), "validation": len(val_set)},
        "best_epoch": best_epoch,
        "history": history,
        "smoke": smoke,
    }
    result_path = output_dir / f"{model_name}_results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--impairments-config", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "augmented_cnn_lstm",
            "quality_cnn_lstm",
            "gated_quality_cnn_lstm",
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("clean", "mixed_mild", "mixed_moderate", "mixed_severe"),
    )
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.training_config)
    result = train_quality_model(
        args.cache_root,
        args.train_index,
        args.val_index,
        args.impairments_config,
        args.output_dir,
        config,
        args.model,
        list(args.scenarios),
        sample_rate_hz=args.sample_rate_hz,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
