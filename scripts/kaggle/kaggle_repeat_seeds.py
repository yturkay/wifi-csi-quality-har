"""Kaggle notebook cell for fixed-hyperparameter repeat training.

Run this after the notebook setup cells have defined CODE, CACHE_ROOT,
TRAIN_INDEX, VAL_INDEX, WORK, sys, subprocess, json, and shutil.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

# The cell is self-contained except for the already extracted cache. Recreate
# notebook variables after a kernel restart without unpacking 4.1 GiB again.
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/wifi_csi_search")
WORK.mkdir(parents=True, exist_ok=True)

code_candidates = list(INPUT.rglob("pyproject.toml"))
assert code_candidates, "Kod paketi bulunamadı."
CODE = code_candidates[0].parent
SRC = CODE / "src"
assert SRC.exists(), f"src klasörü bulunamadı: {SRC}"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

train_indexes = list((WORK / "cache").rglob("csi_bench_train_id_cache_index.jsonl"))
assert train_indexes, "Açılmış önbellek bulunamadı; önce .bin önbellek hücresini çalıştırın."
TRAIN_INDEX = train_indexes[0]
CACHE_ROOT = TRAIN_INDEX.parent
VAL_INDEX = CACHE_ROOT / "csi_bench_val_id_cache_index.jsonl"
assert VAL_INDEX.exists(), "Doğrulama indeks dosyası bulunamadı."

print("Kod kökü:", CODE)
print("Cache kökü:", CACHE_ROOT)


SEEDS = [2030, 2031]
repeat_rows = []

for seed in SEEDS:
    config_path = WORK / f"best_training_seed_{seed}.toml"
    config_path.write_text(
        f"""[training]
model = "cnn_lstm"
seed = {seed}
epochs = 30
batch_size = 96
learning_rate = 0.0006662034366198896
weight_decay = 7.66276127696697e-05
dropout = 0.30
early_stopping_patience = 6
num_workers = 2
calibration_bins = 15
class_weighting = false
""",
        encoding="utf-8",
    )

    output_dir = WORK / f"final_seed_{seed}"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "wifi_csi_quality.quality_learning",
        "--cache-root",
        str(CACHE_ROOT),
        "--train-index",
        str(TRAIN_INDEX),
        "--val-index",
        str(VAL_INDEX),
        "--impairments-config",
        str(CODE / "configs/impairments_v1.toml"),
        "--training-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        "gated_quality_cnn_lstm",
        "--resume",
    ]

    print(f"\n===== SEED {seed} BAŞLIYOR =====")
    subprocess.run(cmd, check=True)

    result_path = output_dir / "gated_quality_cnn_lstm_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    best_row = max(
        result["history"],
        key=lambda row: row["validation"]["macro_f1"],
    )
    validation = best_row["validation"]
    row = {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "accuracy": float(validation["accuracy"]),
        "balanced_accuracy": float(validation["balanced_accuracy"]),
        "macro_f1": float(validation["macro_f1"]),
        "ece": float(validation["ece"]),
    }
    repeat_rows.append(row)
    print(
        f"SEED {seed} TAMAMLANDI | "
        f"epoch={row['best_epoch']} | "
        f"acc={100 * row['accuracy']:.2f}% | "
        f"bal_acc={100 * row['balanced_accuracy']:.2f}% | "
        f"macro_f1={100 * row['macro_f1']:.2f}% | "
        f"ece={row['ece']:.4f}"
    )

summary_path = WORK / "repeat_summary_2030_2031.json"
summary_path.write_text(
    json.dumps(repeat_rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

export_dir = Path("/kaggle/working/wifi_csi_repeats_export")
if export_dir.exists():
    shutil.rmtree(export_dir)
export_dir.mkdir(parents=True)

for seed in SEEDS:
    shutil.copytree(
        WORK / f"final_seed_{seed}",
        export_dir / f"final_seed_{seed}",
    )
    shutil.copy2(
        WORK / f"best_training_seed_{seed}.toml",
        export_dir / f"best_training_seed_{seed}.toml",
    )
shutil.copy2(summary_path, export_dir / summary_path.name)

archive = shutil.make_archive(
    "/kaggle/working/wifi_csi_repeats_2030_2031",
    "zip",
    export_dir,
)
print("\nİki tekrar tamamlandı.")
print("İndirilecek arşiv:", archive)
