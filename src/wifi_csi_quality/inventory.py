"""Create a lightweight, reproducible inventory of local CSI datasets.

The scanner reads metadata and directory entries only. It never changes raw
dataset files and deliberately avoids hashing multi-gigabyte CSI payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CORE_SPLITS = (
    "train_id",
    "val_id",
    "test_id",
    "test_cross_device",
    "test_cross_env",
    "test_cross_user",
    "test_cross_user_env",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_values(rows: Iterable[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row[column] for row in rows).items()))


def inventory_csi_bench(root: Path) -> dict[str, Any]:
    har_root = root / "Multitask" / "HumanActivityRecognition"
    metadata_dir = har_root / "metadata"
    splits_dir = har_root / "splits"
    metadata_path = metadata_dir / "sample_metadata.csv"

    required = (
        root / "README.md",
        metadata_path,
        metadata_dir / "label_mapping.json",
        metadata_dir / "device_mapping.json",
        root / "Multitask" / "sub_Human_h5",
    )
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError("Missing CSI-Bench paths: " + "; ".join(missing_required))

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    split_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int | dict[str, Any]] = {}
    split_hashes: dict[str, str] = {}
    for path in sorted(splits_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_hashes[path.name] = sha256_file(path)
        if isinstance(payload, list):
            split_sets[path.stem] = set(payload)
            split_counts[path.stem] = len(payload)
        else:
            split_counts[path.stem] = {"type": "object", "keys": sorted(payload)}

    # Build the set once. Calling Path.resolve()/is_file() for every metadata
    # row is disproportionately slow on Windows volumes mounted in WSL.
    h5_root = root / "Multitask" / "sub_Human_h5"
    available_h5 = {
        os.path.normcase(os.path.normpath(str(path))) for path in h5_root.rglob("*.h5")
    }
    missing_sample_paths = 0
    for row in rows:
        # file_path values are relative to the metadata directory.
        sample_path = os.path.normcase(
            os.path.normpath(os.path.join(metadata_dir, row["file_path"]))
        )
        if sample_path not in available_h5:
            missing_sample_paths += 1

    overlaps: dict[str, int] = {}
    available_core = [name for name in CORE_SPLITS if name in split_sets]
    for index, left in enumerate(available_core):
        for right in available_core[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(split_sets[left] & split_sets[right])

    metadata_hashes = {
        path.name: sha256_file(path)
        for path in sorted(metadata_dir.iterdir())
        if path.is_file()
    }

    return {
        "root": str(root),
        "sample_metadata_rows": len(rows),
        "columns": list(rows[0]) if rows else [],
        "distributions": {
            column: count_values(rows, column)
            for column in (
                "activity",
                "user",
                "environment",
                "device",
                "c_freq",
                "num_sub",
                "distance",
                "is_multi_subject",
            )
        },
        "split_counts": split_counts,
        "core_split_overlaps": overlaps,
        "missing_sample_paths": missing_sample_paths,
        "metadata_sha256": metadata_hashes,
        "split_sha256": split_hashes,
    }


XRF_FILE_RE = re.compile(
    r"^(?P<t1>\d+)_(?P<t2>\d+)_(?P<t3>\d+)\.(?P<extension>dat|mat)$",
    re.IGNORECASE,
)


def inventory_xrf55(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing XRF55 Wi-Fi root: {root}")

    by_scene: Counter[str] = Counter()
    by_receiver: Counter[str] = Counter()
    by_scene_receiver: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    subjects_by_scene: dict[str, set[str]] = defaultdict(set)
    filename_token_1: Counter[str] = Counter()
    malformed: list[str] = []
    total_bytes = 0
    file_count = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".dat", ".mat"}:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        file_count += 1
        total_bytes += path.stat().st_size
        if len(parts) != 4:
            malformed.append(str(relative))
            continue
        scene, receiver, subject, filename = parts
        match = XRF_FILE_RE.fullmatch(filename)
        if match is None:
            malformed.append(str(relative))
            continue
        by_scene[scene] += 1
        by_receiver[receiver] += 1
        by_scene_receiver[f"{scene}/{receiver}"] += 1
        subjects_by_scene[scene].add(subject)
        filename_token_1[match.group("t1")] += 1
        by_extension[match.group("extension").lower()] += 1

    return {
        "root": str(root),
        "file_count": file_count,
        "size_bytes": total_bytes,
        "size_gib": round(total_bytes / (1024**3), 3),
        "files_by_scene": dict(sorted(by_scene.items())),
        "files_by_receiver": dict(sorted(by_receiver.items())),
        "files_by_scene_receiver": dict(sorted(by_scene_receiver.items())),
        "files_by_extension": dict(sorted(by_extension.items())),
        "subjects_by_scene": {
            scene: sorted(subjects) for scene, subjects in sorted(subjects_by_scene.items())
        },
        "first_filename_token_counts": dict(sorted(filename_token_1.items())),
        "malformed_path_count": len(malformed),
        "malformed_path_examples": malformed[:20],
    }


def render_markdown(inventory: dict[str, Any]) -> str:
    csi = inventory["datasets"]["csi_bench"]
    xrf = inventory["datasets"]["xrf55"]
    distributions = csi["distributions"]
    core_splits = {
        key: csi["split_counts"][key]
        for key in CORE_SPLITS
        if key in csi["split_counts"]
    }
    return "\n".join(
        (
            "# Dataset Inventory",
            "",
            f"Generated: `{inventory['generated_at_utc']}`",
            "",
            "## CSI-Bench HAR",
            "",
            f"- Metadata rows: **{csi['sample_metadata_rows']:,}**",
            f"- Referenced H5 files missing: **{csi['missing_sample_paths']:,}**",
            f"- Activities: `{json.dumps(distributions['activity'], ensure_ascii=False)}`",
            f"- Users: **{len(distributions['user'])}**",
            f"- Environments: **{len(distributions['environment'])}**",
            f"- Devices: **{len(distributions['device'])}**",
            f"- Core split sizes: `{json.dumps(core_splits, ensure_ascii=False)}`",
            "",
            "## XRF55 Wi-Fi",
            "",
            f"- CSI sample files: **{xrf['file_count']:,}**",
            f"- Size: **{xrf['size_gib']:.3f} GiB**",
            f"- Files by extension: `{json.dumps(xrf['files_by_extension'])}`",
            f"- Files by scene: `{json.dumps(xrf['files_by_scene'])}`",
            f"- Files by receiver: `{json.dumps(xrf['files_by_receiver'])}`",
            f"- Malformed paths: **{xrf['malformed_path_count']:,}**",
            "",
            "## Integrity scope",
            "",
            "SHA-256 values cover CSI-Bench metadata and split JSON files only. ",
            "Large CSI payload files are counted and path-validated but are not hashed by this scan.",
            "",
        )
    )


def load_config(path: Path) -> tuple[Path, Path]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    datasets = config["datasets"]
    return Path(datasets["csi_bench_root"]), Path(datasets["xrf55_wifi_root"])


def build_inventory(config_path: Path) -> dict[str, Any]:
    csi_root, xrf_root = load_config(config_path)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "csi_bench": inventory_csi_bench(csi_root),
            "xrf55": inventory_xrf55(xrf_root),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = build_inventory(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dataset_inventory.json"
    markdown_path = args.output_dir / "dataset_inventory.md"
    json_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(inventory), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

