"""Summarize paired robustness replicates for quality-aware CSI models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll")


def _load_replicates(directory: Path) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("replicate_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["replicate_seed"])
        if seed in payloads:
            raise ValueError(f"Duplicate replicate seed {seed} in {directory}")
        payloads[seed] = payload
    if not payloads:
        raise ValueError(f"No replicate JSON files found in {directory}")
    return payloads


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_paired_replicates(
    quality_dir: Path,
    augmented_dir: Path,
) -> dict[str, Any]:
    quality = _load_replicates(quality_dir)
    augmented = _load_replicates(augmented_dir)
    if set(quality) != set(augmented):
        raise ValueError("Quality-aware and augmented replicate seeds do not match")
    seeds = sorted(quality)
    quality_model = str(quality[seeds[0]]["model"])
    augmented_model = str(augmented[seeds[0]]["model"])
    delta_key = f"paired_delta_{quality_model}_minus_{augmented_model}"
    scenarios = list(quality[seeds[0]]["scenarios"])
    for seed in seeds:
        if list(quality[seed]["scenarios"]) != scenarios:
            raise ValueError(f"Quality-aware scenario mismatch for seed {seed}")
        if list(augmented[seed]["scenarios"]) != scenarios:
            raise ValueError(f"Augmented scenario mismatch for seed {seed}")

    summary: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_summary: dict[str, Any] = {}
        for model_key, payloads in (
            (quality_model, quality),
            (augmented_model, augmented),
        ):
            scenario_summary[model_key] = {
                metric: _mean_sd(
                    [
                        float(payloads[seed]["scenarios"][scenario]["metrics"][metric])
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }
        scenario_summary[delta_key] = {
            metric: _mean_sd(
                [
                    float(quality[seed]["scenarios"][scenario]["metrics"][metric])
                    - float(augmented[seed]["scenarios"][scenario]["metrics"][metric])
                    for seed in seeds
                ]
            )
            for metric in METRICS
        }
        summary[scenario] = scenario_summary

    return {
        "schema_version": 1,
        "replicate_seeds": seeds,
        "replicate_count": len(seeds),
        "quality_model": quality_model,
        "augmented_model": augmented_model,
        "quality_best_epoch": quality[seeds[0]]["best_epoch"],
        "augmented_best_epoch": augmented[seeds[0]]["best_epoch"],
        "scenarios": summary,
    }


def _percent(stat: dict[str, float]) -> str:
    return f"{100 * stat['mean']:.2f} ± {100 * stat['sample_sd']:.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    quality_model = payload["quality_model"]
    augmented_model = payload["augmented_model"]
    delta_key = f"paired_delta_{quality_model}_minus_{augmented_model}"
    lines = [
        "# Quality-aware versus quality-blind robustness",
        "",
        f"Paired impairment seeds: {', '.join(map(str, payload['replicate_seeds']))}. ",
        "Values are mean ± sample SD across impairment replicates. Model-training "
        "uncertainty is not represented because both models currently use one training seed.",
        "",
        "| Scenario | Quality macro-F1 (%) | Blind macro-F1 (%) | Paired delta (pp) | Quality accuracy (%) | Blind accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    wins = 0
    for scenario, values in payload["scenarios"].items():
        quality = values[quality_model]
        augmented = values[augmented_model]
        delta = values[delta_key]["macro_f1"]
        wins += delta["mean"] > 0
        lines.append(
            f"| {scenario} | {_percent(quality['macro_f1'])} | "
            f"{_percent(augmented['macro_f1'])} | "
            f"{100 * delta['mean']:+.2f} ± {100 * delta['sample_sd']:.2f} | "
            f"{_percent(quality['accuracy'])} | {_percent(augmented['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            f"Quality-aware macro-F1 is higher in {wins}/{len(payload['scenarios'])} scenarios.",
            "A positive paired delta favors the quality-aware model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", required=True, type=Path)
    parser.add_argument("--augmented-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize_paired_replicates(args.quality_dir, args.augmented_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
