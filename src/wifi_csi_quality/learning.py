"""PyTorch datasets, clean-CSI baselines, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


CLASS_NAMES = ("jumping", "running", "walking", "wavinghand")


class CacheDataset(Dataset):
    """Lazy reader for cache entries referenced by a generated JSONL index."""

    def __init__(
        self,
        index_path: Path,
        cache_root: Path,
        limit: int | None = None,
        balanced_limit: bool = False,
    ):
        self.index_path = Path(index_path)
        self.cache_root = Path(cache_root)
        with self.index_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None and balanced_limit:
            per_class = limit // len(CLASS_NAMES)
            selected = []
            for label_index in range(len(CLASS_NAMES)):
                class_rows = [
                    row for row in self.rows if int(row["label_index"]) == label_index
                ]
                selected.extend(class_rows[:per_class])
            self.rows = selected
        elif limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Cache index is empty: {self.index_path}")
        labels = {int(row["label_index"]) for row in self.rows}
        if not labels.issubset(range(len(CLASS_NAMES))):
            raise ValueError(f"Unexpected label indices in {self.index_path}: {labels}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.cache_root / Path(row["cache_relative_path"])
        with np.load(path, allow_pickle=False) as cached:
            values = np.asarray(cached["x"], dtype=np.float32)
            label = int(cached["label"].item())
        if values.shape != (500, 64):
            raise ValueError(f"Unexpected cache tensor shape {values.shape} in {path}")
        return torch.from_numpy(values), torch.tensor(label, dtype=torch.long)

    def class_counts(self) -> dict[int, int]:
        counts = Counter(int(row["label_index"]) for row in self.rows)
        return {index: counts.get(index, 0) for index in range(len(CLASS_NAMES))}


class MLPBaseline(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(500 * 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN1DBaseline(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.transpose(1, 2)))


class CNNLSTMBaseline(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(sequence)
        return self.classifier(sequence.mean(dim=1))


def create_model(name: str, num_classes: int = 4, dropout: float = 0.3) -> nn.Module:
    models = {
        "mlp": MLPBaseline,
        "cnn1d": CNN1DBaseline,
        "cnn_lstm": CNNLSTMBaseline,
    }
    try:
        return models[name](num_classes=num_classes, dropout=dropout)
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(models)}") from exc


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "cnn1d"
    seed: int = 2026
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stopping_patience: int = 6
    num_workers: int = 2
    calibration_bins: int = 15
    class_weighting: bool = True


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("rb") as handle:
        values = tomllib.load(handle)["training"]
    return TrainingConfig(**values)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor, classes: int = 4) -> torch.Tensor:
    encoded = labels.to(torch.int64) * classes + predictions.to(torch.int64)
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def metrics_from_outputs(
    logits: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    matrix = confusion_matrix(labels, predictions, probabilities.shape[1]).to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(recall),
    )
    confidence, _ = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    ece = torch.tensor(0.0, dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            gap = correct[selected].float().mean() - confidence[selected].mean()
            ece += selected.float().mean().double() * gap.abs().double()
    nll = nn.functional.cross_entropy(logits, labels).item()
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "ece": float(ece),
        "nll": float(nll),
        "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(4)},
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(4)},
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, bins: int) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for values, labels in loader:
            logits_parts.append(model(values.to(device)).cpu())
            label_parts.append(labels)
    return metrics_from_outputs(torch.cat(logits_parts), torch.cat(label_parts), bins=bins)


def _class_weights(dataset: CacheDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASS_NAMES) * max(counts[index], 1)) for index in range(4)],
        dtype=torch.float32,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so an interruption cannot corrupt it."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train(
    cache_root: Path,
    train_index: Path,
    val_index: Path,
    test_index: Path,
    output_dir: Path,
    config: TrainingConfig,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    set_deterministic(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = 128 if smoke else None
    train_set = CacheDataset(train_index, cache_root, limit=limit, balanced_limit=smoke)
    val_set = CacheDataset(val_index, cache_root, limit=limit, balanced_limit=smoke)
    test_set = CacheDataset(test_index, cache_root, limit=limit, balanced_limit=smoke)
    generator = torch.Generator().manual_seed(config.seed)
    loader_args = {
        "batch_size": min(config.batch_size, len(train_set)),
        "num_workers": 0 if smoke else config.num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    device = torch.device("cpu")
    model = create_model(config.model, dropout=config.dropout).to(device)
    weights = _class_weights(train_set).to(device) if config.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_macro_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    epochs = 1 if smoke else config.epochs
    checkpoint_path = output_dir / f"{config.model}_best.pt"
    last_checkpoint_path = output_dir / f"{config.model}_last.pt"
    start_epoch = 1

    if resume and last_checkpoint_path.exists():
        state = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        if state["model_name"] != config.model:
            raise ValueError(
                f"Resume checkpoint model {state['model_name']!r} does not match {config.model!r}"
            )
        if state["training_config"] != asdict(config):
            raise ValueError("Resume checkpoint training configuration does not match")
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
        model.train()
        loss_total = 0.0
        seen = 0
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.shape[0]
            seen += labels.shape[0]
        validation = _evaluate(model, val_loader, device, config.calibration_bins)
        history.append({"epoch": epoch, "train_loss": loss_total / seen, "validation": validation})
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "model_name": config.model,
                    "model_state_dict": model.state_dict(),
                    "training_config": asdict(config),
                    "class_names": CLASS_NAMES,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        _atomic_torch_save(
            {
                "model_name": config.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_config": asdict(config),
                "class_names": CLASS_NAMES,
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
            last_checkpoint_path,
        )
        if stale_epochs >= config.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = {
        "schema_version": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": config.model,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_config": asdict(config),
        "dataset_sizes": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "class_counts": train_set.class_counts(),
        "best_epoch": best_epoch,
        "history": history,
        "test": _evaluate(model, test_loader, device, config.calibration_bins),
        "smoke": smoke,
    }
    result_path = output_dir / f"{config.model}_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=("mlp", "cnn1d", "cnn_lstm"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.training_config)
    if args.model:
        config = TrainingConfig(**{**asdict(config), "model": args.model})
    result = train(
        args.cache_root,
        args.train_index,
        args.val_index,
        args.test_index,
        args.output_dir,
        config,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
