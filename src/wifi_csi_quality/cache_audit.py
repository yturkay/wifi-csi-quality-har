"""Audit generated cache indexes without modifying cache or raw CSI files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import sha256_file


CORE_INDEX_NAMES = (
    "csi_bench_train_id",
    "csi_bench_val_id",
    "csi_bench_test_id",
    "xrf55_external_test",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_positions(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def audit_index(index_path: Path, cache_root: Path, deep_samples: int = 32) -> dict[str, Any]:
    rows = _read_jsonl(index_path)
    sample_ids = [row["sample_id"] for row in rows]
    cache_paths = [row["cache_relative_path"] for row in rows]
    missing = [relative for relative in cache_paths if not (cache_root / relative).is_file()]
    deep_failures: list[dict[str, str]] = []
    for position in _sample_positions(len(rows), deep_samples):
        row = rows[position]
        path = cache_root / row["cache_relative_path"]
        try:
            if sha256_file(path) != row["cache_sha256"]:
                raise ValueError("SHA-256 mismatch")
            with np.load(path, allow_pickle=False) as cached:
                values = cached["x"]
                label = int(cached["label"].item())
                provenance = json.loads(str(cached["provenance_json"].item()))
            if values.shape != (500, 64) or values.dtype != np.float32:
                raise ValueError(f"unexpected tensor {values.shape} {values.dtype}")
            if not np.isfinite(values).all():
                raise ValueError("non-finite tensor values")
            if label != int(row["label_index"]):
                raise ValueError("label mismatch")
            if provenance["sample_id"] != row["sample_id"]:
                raise ValueError("provenance sample ID mismatch")
        except Exception as exc:
            deep_failures.append({"sample_id": row["sample_id"], "error": repr(exc)})
    duplicate_sample_ids = len(sample_ids) - len(set(sample_ids))
    duplicate_cache_paths = len(cache_paths) - len(set(cache_paths))
    result = {
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "count": len(rows),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_cache_paths": duplicate_cache_paths,
        "missing_cache_files": len(missing),
        "deep_samples_checked": len(_sample_positions(len(rows), deep_samples)),
        "deep_failures": deep_failures,
    }
    result["passed"] = not (
        duplicate_sample_ids or duplicate_cache_paths or missing or deep_failures
    )
    return result


def audit_core(cache_root: Path, output_dir: Path, deep_samples: int = 32) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    indexes = {
        name: audit_index(cache_root / f"{name}_cache_index.jsonl", cache_root, deep_samples)
        for name in CORE_INDEX_NAMES
    }
    csi_ids = {
        name: {row["sample_id"] for row in _read_jsonl(cache_root / f"{name}_cache_index.jsonl")}
        for name in CORE_INDEX_NAMES[:3]
    }
    overlaps = {
        "train__validation": len(csi_ids[CORE_INDEX_NAMES[0]] & csi_ids[CORE_INDEX_NAMES[1]]),
        "train__test": len(csi_ids[CORE_INDEX_NAMES[0]] & csi_ids[CORE_INDEX_NAMES[2]]),
        "validation__test": len(csi_ids[CORE_INDEX_NAMES[1]] & csi_ids[CORE_INDEX_NAMES[2]]),
    }
    passed = all(item["passed"] for item in indexes.values()) and not any(overlaps.values())
    report = {
        "schema_version": 1,
        "cache_root": str(cache_root),
        "indexes": indexes,
        "csi_bench_split_overlaps": overlaps,
        "passed": passed,
    }
    json_path = output_dir / "core_cache_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Çekirdek Önbellek Bütünlük Raporu",
        "",
        f"Genel sonuç: **{'BAŞARILI' if passed else 'BAŞARISIZ'}**",
        "",
        "| Küme | Örnek | Derin kontrol | Eksik | Tekrar | Sonuç |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in indexes.items():
        duplicates = item["duplicate_sample_ids"] + item["duplicate_cache_paths"]
        lines.append(
            f"| {name} | {item['count']} | {item['deep_samples_checked']} | "
            f"{item['missing_cache_files']} | {duplicates} | "
            f"{'Başarılı' if item['passed'] else 'Başarısız'} |"
        )
    lines.extend(
        [
            "",
            "## CSI-Bench bölme çakışmaları",
            "",
            *[f"- {name}: {count}" for name, count in overlaps.items()],
            "",
            "Derin kontrolde seçilmiş dosyaların SHA-256 değeri, 500×64 float32 tensör "
            "biçimi, sonlu değerleri, etiketi ve kaynak kimliği doğrulanmıştır.",
        ]
    )
    (output_dir / "core_cache_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deep-samples", type=int, default=32)
    args = parser.parse_args()
    report = audit_core(args.cache_root, args.output_dir, args.deep_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
