"""Kaggle-friendly Optuna search for the gated quality-aware CSI model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import optuna

from .learning import TrainingConfig
from .quality_learning import train_quality_model


def _balanced_subset(source: Path, target: Path, per_class: int) -> Path:
    selected: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label_index"])
            if label in selected and len(selected[label]) < per_class:
                selected[label].append(row)
    missing = {label: per_class - len(rows) for label, rows in selected.items() if len(rows) < per_class}
    if missing:
        raise ValueError(f"Not enough rows for balanced subset: {missing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for label in sorted(selected):
            for row in selected[label]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--impairments-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_index = _balanced_subset(
        args.train_index,
        args.output_dir / "balanced_train_subset.jsonl",
        args.train_per_class,
    )
    scenarios = ["clean", "mixed_mild", "mixed_moderate", "mixed_severe"]

    def objective(trial: optuna.Trial) -> float:
        config = TrainingConfig(
            model="cnn_lstm",
            seed=5000 + trial.number,
            epochs=args.epochs,
            batch_size=trial.suggest_categorical("batch_size", [64, 96, 128]),
            learning_rate=trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.15, 0.50, step=0.05),
            early_stopping_patience=3,
            num_workers=2,
            calibration_bins=15,
            class_weighting=trial.suggest_categorical("class_weighting", [True, False]),
        )
        result = train_quality_model(
            cache_root=args.cache_root,
            train_index=subset_index,
            val_index=args.val_index,
            impairment_path=args.impairments_config,
            output_dir=args.output_dir / f"trial_{trial.number:03d}",
            config=config,
            model_name="gated_quality_cnn_lstm",
            scenario_names=scenarios,
        )
        best = max(float(item["validation"]["macro_f1"]) for item in result["history"])
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        return best

    storage = f"sqlite:///{(args.output_dir / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name="gated_quality_cnn_lstm",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    summary = {
        "best_macro_f1": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_epoch": study.best_trial.user_attrs.get("best_epoch"),
        "completed_trials": len(study.trials),
    }
    (args.output_dir / "optuna_best.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    study.trials_dataframe().to_csv(args.output_dir / "optuna_trials.csv", index=False)

    best = study.best_params
    final_config = TrainingConfig(
        model="cnn_lstm",
        seed=2029,
        epochs=30,
        batch_size=int(best["batch_size"]),
        learning_rate=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        dropout=float(best["dropout"]),
        early_stopping_patience=6,
        num_workers=2,
        calibration_bins=15,
        class_weighting=bool(best["class_weighting"]),
    )
    values = asdict(final_config)
    toml_lines = ["[training]"]
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        toml_lines.append(f"{key} = {rendered}")
    (args.output_dir / "best_training.toml").write_text(
        "\n".join(toml_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
