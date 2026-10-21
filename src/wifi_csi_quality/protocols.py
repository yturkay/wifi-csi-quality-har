"""Build portable, leakage-aware manifests for the common four-class study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .inventory import load_config, sha256_file


COMMON_LABELS = ("jumping", "running", "walking", "wavinghand")
COMMON_LABEL_TO_INDEX = {label: index for index, label in enumerate(COMMON_LABELS)}
XRF55_ACTION_TO_COMMON_LABEL = {
    31: "wavinghand",
    34: "jumping",
    35: "running",
    39: "walking",
}
XRF_NAME_RE = re.compile(
    r"^(?P<volunteer>\d+)_(?P<action>\d+)_(?P<repetition>\d+)\.(?P<format>dat|mat)$",
    re.IGNORECASE,
)

CSI_BENCH_SPLITS = (
    "train_id",
    "val_id",
    "test_id",
    "test_cross_device",
    "test_cross_env",
    "test_cross_user",
    "test_cross_user_env",
)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in records).items()))


def build_csi_bench_manifests(root: Path, output_dir: Path) -> dict[str, Any]:
    har = root / "Multitask" / "HumanActivityRecognition"
    metadata_dir = har / "metadata"
    with (metadata_dir / "sample_metadata.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row["label"]
        if label not in COMMON_LABEL_TO_INDEX:
            continue
        absolute = os.path.normpath(os.path.join(metadata_dir, row["file_path"]))
        relative = os.path.relpath(absolute, root).replace(os.sep, "/")
        selected[row["id"]] = {
            "dataset": "CSI-Bench",
            "sample_id": row["id"],
            "relative_path": relative,
            "label": label,
            "label_index": COMMON_LABEL_TO_INDEX[label],
            "user": row["user"],
            "environment": row["environment"],
            "device": row["device"],
            "frequency_band": row["c_freq"],
            "num_subcarriers": int(row["num_sub"]),
        }

    summary: dict[str, Any] = {"selected_metadata_rows": len(selected), "splits": {}}
    split_id_sets: dict[str, set[str]] = {}
    for split_name in CSI_BENCH_SPLITS:
        ids = json.loads((har / "splits" / f"{split_name}.json").read_text(encoding="utf-8"))
        records = [selected[sample_id] for sample_id in ids if sample_id in selected]
        records.sort(key=lambda row: row["sample_id"])
        sample_ids = {row["sample_id"] for row in records}
        if len(sample_ids) != len(records):
            raise ValueError(f"Duplicate CSI-Bench sample IDs in {split_name}")
        split_id_sets[split_name] = sample_ids
        filename = f"csi_bench_{split_name}.jsonl"
        _write_jsonl(output_dir / filename, records)
        summary["splits"][split_name] = {
            "file": filename,
            "count": len(records),
            "class_counts": _counts(records),
        }
    id_core = ("train_id", "val_id", "test_id")
    overlaps = {
        f"{left}__{right}": len(split_id_sets[left] & split_id_sets[right])
        for index, left in enumerate(id_core)
        for right in id_core[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"CSI-Bench in-domain split leakage detected: {overlaps}")
    summary["validation"] = {
        "in_domain_split_overlaps": overlaps,
        "passed": True,
    }
    return summary


def scan_xrf55_common(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".dat", ".mat"}:
            continue
        match = XRF_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        action_id = int(match.group("action"))
        if action_id not in XRF55_ACTION_TO_COMMON_LABEL:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            raise ValueError(f"Unexpected XRF55 path layout: {relative}")
        scene, receiver, subject_folder, _ = relative.parts
        volunteer = match.group("volunteer")
        if subject_folder != volunteer:
            raise ValueError(
                f"Volunteer mismatch in {relative}: folder={subject_folder}, file={volunteer}"
            )
        label = XRF55_ACTION_TO_COMMON_LABEL[action_id]
        records.append(
            {
                "dataset": "XRF55",
                "sample_id": f"{scene}:{receiver}:{path.stem}",
                "relative_path": relative.as_posix(),
                "label": label,
                "label_index": COMMON_LABEL_TO_INDEX[label],
                "action_id": action_id,
                "volunteer": volunteer,
                "scene": scene,
                "receiver": receiver,
                "repetition": int(match.group("repetition")),
                "format": match.group("format").lower(),
            }
        )
    return sorted(
        records,
        key=lambda row: (
            row["scene"],
            row["receiver"],
            row["volunteer"],
            row["action_id"],
            row["repetition"],
        ),
    )


def build_xrf55_manifests(root: Path, output_dir: Path) -> dict[str, Any]:
    records = scan_xrf55_common(root)
    unique_paths = {row["relative_path"] for row in records}
    unique_ids = {row["sample_id"] for row in records}
    if len(unique_paths) != len(records) or len(unique_ids) != len(records):
        raise ValueError("Duplicate XRF55 paths or sample IDs in common-class selection")
    split_sets = {
        "external_test": records,
        "repeat_train": [row for row in records if 1 <= row["repetition"] <= 14],
        "repeat_val": [row for row in records if 15 <= row["repetition"] <= 17],
        "repeat_test": [row for row in records if 18 <= row["repetition"] <= 20],
        "cross_scene_train": [
            row
            for row in records
            if row["scene"] == "Scene_1" and 1 <= row["repetition"] <= 14
        ],
        "cross_scene_val": [
            row
            for row in records
            if row["scene"] == "Scene_1" and 15 <= row["repetition"] <= 20
        ],
        "cross_scene_test": [row for row in records if row["scene"] != "Scene_1"],
    }
    summary: dict[str, Any] = {"selected_files": len(records), "splits": {}}
    for split_name, split_records in split_sets.items():
        filename = f"xrf55_{split_name}.jsonl"
        _write_jsonl(output_dir / filename, split_records)
        summary["splits"][split_name] = {
            "file": filename,
            "count": len(split_records),
            "class_counts": _counts(split_records),
        }
    split_paths = {
        name: {row["relative_path"] for row in rows} for name, rows in split_sets.items()
    }
    disjoint_groups = {
        "repeat_train__repeat_val": len(
            split_paths["repeat_train"] & split_paths["repeat_val"]
        ),
        "repeat_train__repeat_test": len(
            split_paths["repeat_train"] & split_paths["repeat_test"]
        ),
        "repeat_val__repeat_test": len(
            split_paths["repeat_val"] & split_paths["repeat_test"]
        ),
        "cross_scene_train__cross_scene_val": len(
            split_paths["cross_scene_train"] & split_paths["cross_scene_val"]
        ),
        "cross_scene_train__cross_scene_test": len(
            split_paths["cross_scene_train"] & split_paths["cross_scene_test"]
        ),
        "cross_scene_val__cross_scene_test": len(
            split_paths["cross_scene_val"] & split_paths["cross_scene_test"]
        ),
    }
    if any(disjoint_groups.values()):
        raise ValueError(f"XRF55 split leakage detected: {disjoint_groups}")
    summary["validation"] = {
        "unique_relative_paths": len(unique_paths),
        "unique_sample_ids": len(unique_ids),
        "split_overlaps": disjoint_groups,
        "passed": True,
    }
    return summary


def build_protocol(config_path: Path, output_dir: Path) -> dict[str, Any]:
    csi_root, xrf_root = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "protocol": "common4_quality_v1",
        "classes": list(COMMON_LABELS),
        "label_to_index": COMMON_LABEL_TO_INDEX,
        "csi_bench": build_csi_bench_manifests(csi_root, output_dir),
        "xrf55": build_xrf55_manifests(xrf_root, output_dir),
    }
    summary["manifest_sha256"] = {
        path.name: sha256_file(path) for path in sorted(output_dir.glob("*.jsonl"))
    }
    (output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "protocol_summary.md").write_text(
        render_protocol_markdown(summary), encoding="utf-8"
    )
    return summary


def render_protocol_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Common Four-Class Quality-Aware Protocol v1",
        "",
        "## Shared labels",
        "",
        "| Index | Canonical label | XRF55 action ID |",
        "|---:|---|---:|",
    ]
    reverse_xrf = {label: action for action, label in XRF55_ACTION_TO_COMMON_LABEL.items()}
    for label, index in COMMON_LABEL_TO_INDEX.items():
        lines.append(f"| {index} | {label} | {reverse_xrf[label]} |")
    lines.extend(
        [
            "",
            "## CSI-Bench splits",
            "",
            "| Split | Samples | Jumping | Running | Walking | Waving hand |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, info in summary["csi_bench"]["splits"].items():
        counts = info["class_counts"]
        lines.append(
            f"| {name} | {info['count']} | {counts.get('jumping', 0)} | "
            f"{counts.get('running', 0)} | {counts.get('walking', 0)} | "
            f"{counts.get('wavinghand', 0)} |"
        )
    lines.extend(
        [
            "",
            "## XRF55 splits",
            "",
            "| Split | Samples | Per class |",
            "|---|---:|---:|",
        ]
    )
    for name, info in summary["xrf55"]["splits"].items():
        per_class = sorted(set(info["class_counts"].values()))
        display = str(per_class[0]) if len(per_class) == 1 else "imbalanced"
        lines.append(f"| {name} | {info['count']} | {display} |")
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"- CSI-Bench in-domain splits: **{'PASS' if summary['csi_bench']['validation']['passed'] else 'FAIL'}**",
            f"- XRF55 repetition and cross-scene splits: **{'PASS' if summary['xrf55']['validation']['passed'] else 'FAIL'}**",
            f"- Unique XRF55 paths and IDs: **{summary['xrf55']['validation']['unique_relative_paths']}**",
            "",
            "## Usage policy",
            "",
            "CSI-Bench official splits remain unchanged except for filtering to the four shared labels. "
            "XRF55 `external_test` is reserved for cross-dataset evaluation and must not be used "
            "for model selection. Manifest files contain paths and labels only; no CSI payload is redistributed.",
            "",
            "Every JSONL manifest is covered by a SHA-256 value in `protocol_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_protocol(args.config, args.output_dir)
    print(args.output_dir / "protocol_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

