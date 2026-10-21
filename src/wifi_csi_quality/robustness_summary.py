"""Aggregate repeated robustness evaluations into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll")
SCENARIO_LABELS = {
    "clean": "Temiz",
    "single_random_loss_10": "Rastgele %10 kayıp",
    "single_burst_moderate": "Orta burst kaybı",
    "single_irregular_moderate": "Orta düzensiz zamanlama",
    "single_stale_moderate": "Orta eski paket",
    "mixed_mild": "Birleşik hafif",
    "mixed_moderate": "Birleşik orta",
    "mixed_severe": "Birleşik ağır",
}


def aggregate_model(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("At least two replicate files are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    model = payloads[0]["model"]
    scenarios = list(payloads[0]["scenarios"])
    for payload in payloads[1:]:
        if payload["model"] != model or list(payload["scenarios"]) != scenarios:
            raise ValueError("Replicate files do not describe the same model and scenarios")
    result: dict[str, Any] = {
        "model": model,
        "replicate_seeds": [int(payload["replicate_seed"]) for payload in payloads],
        "scenarios": {},
    }
    for scenario in scenarios:
        sample_counts = {
            int(payload["scenarios"][scenario]["sample_count"]) for payload in payloads
        }
        if len(sample_counts) != 1:
            raise ValueError(f"Sample count differs across replicates for {scenario}")
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = np.asarray(
                [payload["scenarios"][scenario]["metrics"][metric] for payload in payloads],
                dtype=np.float64,
            )
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
        result["scenarios"][scenario] = {
            "sample_count": sample_counts.pop(),
            "metrics": metrics,
        }
    return result


def _percent_cell(summary: dict[str, Any], metric: str = "macro_f1") -> str:
    values = summary["metrics"][metric]
    cell = f"%{100 * values['mean']:.2f} ± {100 * values['std']:.2f}"
    return cell.replace(".", ",")


def markdown_report(models: dict[str, dict[str, Any]]) -> str:
    names = list(models)
    scenarios = list(models[names[0]]["scenarios"])
    lines = [
        "# Çoklu Tekrarlı CSI Kalite Dayanıklılığı Özeti",
        "",
        "Sonuçlar üç eşleştirilmiş bozulma tohumu için ortalama ± örnek standart",
        "sapma biçimindedir. Model eğitim tohumu tüm koşullarda 2026'dır.",
        "",
        "## Macro-F1",
        "",
        "| Senaryo | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for scenario in scenarios:
        cells = [
            _percent_cell(models[name]["scenarios"][scenario]) for name in names
        ]
        lines.append(f"| {SCENARIO_LABELS.get(scenario, scenario)} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Dengeli doğruluk",
            "",
            "| Senaryo | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
    )
    for scenario in scenarios:
        cells = [
            _percent_cell(models[name]["scenarios"][scenario], "balanced_accuracy")
            for name in names
        ]
        lines.append(f"| {SCENARIO_LABELS.get(scenario, scenario)} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Not: Bu standart sapmalar yalnızca sentetik bozulma örüntüsünün",
            "değişkenliğini gösterir; farklı model eğitim tohumlarının belirsizliğini",
            "içermez.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_group(value: str) -> tuple[str, list[Path]]:
    try:
        name, path_text = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use MODEL=DIRECTORY") from exc
    paths = sorted(Path(path_text).glob("replicate_*.json"))
    if not paths:
        raise argparse.ArgumentTypeError(f"No replicate JSON files found in {path_text}")
    return name, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, type=_parse_group)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    models = {name: aggregate_model(paths) for name, paths in args.group}
    payload = {"schema_version": 1, "models": models}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown_report(models), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
