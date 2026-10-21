"""Leave-one-user-out / leave-one-environment-out training on Kaggle.

Paste this whole file into ONE Kaggle notebook cell (or upload it and run
`%run KAGGLE_LOO.py`). Required inputs (Add Input):
  * wifi-csi-quality-code        (code package with GPU-enabled training)
  * wifi-csi-preprocessed-cache  (core cache: train/val/test_id + XRF55)
  * wifi-csi-ood-cache           (CSI-Bench OOD cache, wifi_csi_ood_cache.bin)

For every held-out user (and every held-out environment) the gated CNN-LSTM is
retrained from scratch with the manuscript's fixed hyperparameters on all other
domains and evaluated on the held-out domain (clean and mixed_severe).
Resumable: finished folds are skipped; results are zipped after every fold.
"""

# ============================ SETTINGS ============================
SMOKE = False            # True = 2-minute pipeline test (tiny data, 1 epoch)
import os as _os
SMOKE = SMOKE or _os.environ.get("LOO_SMOKE") == "1"
FACTORS = ("user", "environment")
SEEDS = [2029]           # add 2030, 2031 later if quota allows
MAX_TRAIN = 18810        # match the original training-set size
MAX_VAL = 3988
VAL_FRACTION = 0.15
EVAL_SCENARIOS = ("clean", "mixed_severe")
NUM_WORKERS = 4
# =================================================================

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INPUT = Path(os.environ.get("LOO_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("LOO_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("LOO_TMP", "/tmp/wifi_csi_loo_cache"))  # not saved as notebook output
OUT = WORK / ("loo_smoke" if SMOKE else "loo")
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------- locate inputs
def first(paths, what):
    paths = sorted(paths, key=lambda p: len(str(p)))
    assert paths, f"Bulunamadı: {what}"
    return paths[0]


def find_code():
    hits = [p.parent for p in INPUT.rglob("pyproject.toml") if (p.parent / "src" / "wifi_csi_quality").exists()]
    return first(hits, "kod paketi (wifi-csi-quality-code)")


def extract_if_needed(index_name, want_ood, target):
    for root in (TMP, INPUT):
        hits = list(root.rglob(index_name))
        if hits:
            return first(hits, index_name).parent
    archives = [p for p in INPUT.rglob("*") if p.is_file() and "cache" in p.name.lower()
                and ("ood" in p.name.lower()) == want_ood
                and (p.suffix in {".bin", ".tar", ".zip"} or p.name.endswith(".tar.gz"))]
    archive = first(archives, f"{index_name} veya önbellek arşivi")
    target.mkdir(parents=True, exist_ok=True)
    log("Arşiv açılıyor:", archive.name, "->", target)
    if archive.suffix == ".zip":
        shutil.unpack_archive(str(archive), str(target))
    else:
        with tarfile.open(archive, "r:*") as handle:
            try:
                handle.extractall(target, filter="data")
            except TypeError:
                handle.extractall(target)
    return first(list(target.rglob(index_name)), index_name).parent


CODE = find_code()
SRC = CODE / "src"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
CORE = extract_if_needed("csi_bench_train_id_cache_index.jsonl", False, TMP / "core")
OOD = extract_if_needed("csi_bench_test_cross_user_cache_index.jsonl", True, TMP / "ood")
IMPAIRMENTS = CODE / "configs" / "impairments_v1.toml"
log("Kod:", CODE)
log("Çekirdek önbellek:", CORE)
log("OOD önbellek:", OOD)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from wifi_csi_quality.learning import metrics_from_outputs  # noqa: E402
from wifi_csi_quality.quality_learning import AugmentedCacheDataset, GatedQualityCNNLSTM  # noqa: E402
from wifi_csi_quality.robustness import load_impairment_scenarios  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("Cihaz:", DEVICE)


# ---------------------------------------------------------------- pool all CSI-Bench HAR examples
def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


pool, seen_ids = [], set()
sources = [CORE / f"csi_bench_{s}_cache_index.jsonl" for s in ("train_id", "val_id", "test_id")]
sources += [OOD / f"csi_bench_test_{s}_cache_index.jsonl" for s in ("cross_user", "cross_env", "cross_device", "cross_user_env")]
for index in sources:
    assert index.exists(), f"Eksik indeks: {index}"
    for row in read_rows(index):
        if row["sample_id"] in seen_ids:
            continue
        seen_ids.add(row["sample_id"])
        row = dict(row)
        # absolute path: Path(cache_root) / absolute_path == absolute_path
        row["cache_relative_path"] = str((index.parent / row["cache_relative_path"]).resolve())
        pool.append(row)
log(f"Havuz: {len(pool)} tekil CSI-Bench örneği")
for k in ("user", "environment", "device"):
    log(f"  {k}: {dict(sorted(Counter(r[k] for r in pool).items()))}")


# ---------------------------------------------------------------- training / evaluation helpers
def training_config(seed, path):
    path.write_text(f"""[training]
model = "cnn_lstm"
seed = {seed}
epochs = 30
batch_size = 96
learning_rate = 0.0006662034366198896
weight_decay = 7.66276127696697e-05
dropout = 0.30
early_stopping_patience = 6
num_workers = {NUM_WORKERS}
calibration_bins = 15
class_weighting = false
""", encoding="utf-8")
    return path


def evaluate(checkpoint, index, scenario, limit=None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GatedQualityCNNLSTM(dropout=float(ckpt["training_config"]["dropout"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    scenarios = load_impairment_scenarios(IMPAIRMENTS)
    dataset = AugmentedCacheDataset(index, Path("/"), scenarios, [scenario], 2026, "test", limit=limit)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    logits, labels = [], []
    with torch.no_grad():
        for values, quality, y in loader:
            logits.append(model(values.to(DEVICE), quality.to(DEVICE)).float().cpu())
            labels.append(y)
    return metrics_from_outputs(torch.cat(logits), torch.cat(labels), bins=15), len(dataset)


def summarize():
    rows = []
    for path in sorted(OUT.glob("*/seed_*/fold_result.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    (OUT / "all_fold_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with open(OUT / "fold_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", "held_out", "seed", "scenario", "n_test", "n_train", "accuracy", "balanced_accuracy",
                    "macro_f1", "ece", "nll", "best_epoch"])
        for r in rows:
            w.writerow([r["factor"], r["held_out"], r["seed"], r["scenario"], r["n_test"], r["n_train"],
                        f"{100*r['accuracy']:.2f}", f"{100*r['balanced_accuracy']:.2f}", f"{100*r['macro_f1']:.2f}",
                        f"{r['ece']:.4f}", f"{r['nll']:.4f}", r["best_epoch"]])
    agg = defaultdict(list)
    for r in rows:
        agg[(r["factor"], r["scenario"])].append(r)
    with open(OUT / "factor_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", "scenario", "folds", "macro_f1_mean", "macro_f1_sd", "macro_f1_min", "macro_f1_max",
                    "balanced_accuracy_mean", "accuracy_mean", "ece_mean"])
        for (factor, scenario), rs in sorted(agg.items()):
            f1 = np.array([100 * r["macro_f1"] for r in rs])
            w.writerow([factor, scenario, len(rs), f"{f1.mean():.2f}", f"{f1.std(ddof=1):.2f}" if len(rs) > 1 else "",
                        f"{f1.min():.2f}", f"{f1.max():.2f}",
                        f"{np.mean([100*r['balanced_accuracy'] for r in rs]):.2f}",
                        f"{np.mean([100*r['accuracy'] for r in rs]):.2f}", f"{np.mean([r['ece'] for r in rs]):.4f}"])
    archive = shutil.make_archive(str(WORK / ("wifi_csi_loo_SMOKE_results" if SMOKE else "wifi_csi_loo_results")), "zip", OUT)
    return archive, agg


# ---------------------------------------------------------------- folds
folds = []
for factor in FACTORS:
    for value in sorted(set(r[factor] for r in pool)):
        folds.append((factor, value))
if SMOKE:
    folds = folds[:2]
log("Katlar:", ", ".join(f"{f}={v}" for f, v in folds))

# In CSI-Bench several users live in exactly one home, so some held-out-user and
# held-out-environment folds have IDENTICAL test sets. Train those only once.
import hashlib
def signature(rows):
    return hashlib.sha256("\n".join(sorted(r["sample_id"] for r in rows)).encode()).hexdigest()
first_with_signature = {}
for factor, value in folds:
    first_with_signature.setdefault(signature([r for r in pool if r[factor] == value]), (factor, value))
n_unique = len(first_with_signature)
log(f"Benzersiz test kümesi: {n_unique} (toplam kat {len(folds)}); aynı test kümeleri bir kez eğitilir")

start = time.time()
for seed in SEEDS:
    for number, (factor, value) in enumerate(folds, 1):
        fold_dir = OUT / f"{factor}_{value}" / f"seed_{seed}"
        result_path = fold_dir / "fold_result.json"
        if result_path.exists():
            log(f"[{number}/{len(folds)}] {factor}={value} seed={seed}: zaten tamamlanmış, atlanıyor")
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_rows = [r for r in pool if r[factor] == value]
        origin = first_with_signature[signature(test_rows)]
        if origin != (factor, value):
            origin_path = OUT / f"{origin[0]}_{origin[1]}" / f"seed_{seed}" / "fold_result.json"
            copied = json.loads(origin_path.read_text(encoding="utf-8"))
            for row in copied:
                row.update({"factor": factor, "held_out": value, "identical_test_set_as": f"{origin[0]}={origin[1]}"})
            result_path.write_text(json.dumps(copied, indent=2) + "\n", encoding="utf-8")
            log(f"[{number}/{len(folds)}] {factor}={value}: test kümesi {origin[0]}={origin[1]} ile aynı -> sonuç yeniden kullanıldı")
            summarize()
            continue
        rest = [r for r in pool if r[factor] != value]
        rng = random.Random(f"{seed}-{factor}-{value}")
        rng.shuffle(rest)
        n_val = min(MAX_VAL, int(round(VAL_FRACTION * len(rest))))
        val_rows, train_rows = rest[:n_val], rest[n_val:][:MAX_TRAIN]
        train_index = write_rows(train_rows, fold_dir / "train_index.jsonl")
        val_index = write_rows(val_rows, fold_dir / "val_index.jsonl")
        test_index = write_rows(test_rows, fold_dir / "test_index.jsonl")
        log(f"[{number}/{len(folds)}] {factor}={value} seed={seed}: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

        cmd = [sys.executable, "-u", "-m", "wifi_csi_quality.quality_learning",
               "--cache-root", "/", "--train-index", str(train_index), "--val-index", str(val_index),
               "--impairments-config", str(IMPAIRMENTS),
               "--training-config", str(training_config(seed, fold_dir / "training.toml")),
               "--output-dir", str(fold_dir), "--model", "gated_quality_cnn_lstm", "--resume"]
        if SMOKE:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        train_result = json.loads((fold_dir / "gated_quality_cnn_lstm_results.json").read_text(encoding="utf-8"))
        checkpoint = fold_dir / "gated_quality_cnn_lstm_best.pt"

        results = []
        for scenario in EVAL_SCENARIOS:
            metrics, n_test = evaluate(checkpoint, test_index, scenario, limit=256 if SMOKE else None)
            results.append({
                "factor": factor, "held_out": value, "seed": seed, "scenario": scenario,
                "n_test": n_test, "n_train": len(train_rows), "n_val": len(val_rows),
                "best_epoch": int(train_result["best_epoch"]),
                "test_class_counts": dict(Counter(r["label"] for r in test_rows)),
                "test_domains": {k: sorted(set(r[k] for r in test_rows)) for k in ("user", "environment", "device")},
                **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll",
                                           "per_class_f1", "per_class_recall", "confusion_matrix")},
            })
            log(f"      {scenario:13s} macro-F1={100*metrics['macro_f1']:.2f}  bal-acc={100*metrics['balanced_accuracy']:.2f}"
                f"  acc={100*metrics['accuracy']:.2f}  ECE={metrics['ece']:.4f}")
        result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        for name in ("gated_quality_cnn_lstm_last.pt",):  # keep disk usage low; best checkpoint is kept
            (fold_dir / name).unlink(missing_ok=True)
        archive, _ = summarize()
        log(f"      kaydedildi -> {archive}  (geçen süre {(time.time()-start)/60:.1f} dk)")

archive, agg = summarize()
log("BİTTİ. İndirilecek dosya:", archive)
for (factor, scenario), rs in sorted(agg.items()):
    f1 = [100 * r["macro_f1"] for r in rs]
    log(f"  {factor:12s} {scenario:13s} folds={len(rs)}  macro-F1 = {np.mean(f1):.2f} ± {np.std(f1, ddof=1) if len(f1) > 1 else 0:.2f}"
        f"  (min {min(f1):.2f}, max {max(f1):.2f})")
