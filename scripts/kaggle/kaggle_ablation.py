"""Controlled ablation of the quality gate on Kaggle.

Paste this whole file into ONE Kaggle notebook cell. Required inputs (Add Input):
  * wifi-csi-quality-code         (code package, GPU-enabled training)
  * wifi-csi-preprocessed-cache   (core cache: train/val/test_id)
  * wifi-csi-ood-cache            (CSI-Bench OOD cache, for the pure unseen-factor subsets)
  * wifi-csi-trained-checkpoints  (the three gated checkpoints of the manuscript; optional:
                                   if missing, the gated model is retrained with the same protocol)

What it does
  1. Trains two ablation variants with exactly the manuscript protocol (official
     in-distribution train/val split, four-scenario impairment augmentation, fixed
     hyperparameters, seeds 2029-2031):
       - augmented_cnn_lstm : same CNN-LSTM backbone, NO quality channels (quality-blind)
       - quality_cnn_lstm   : quality channels simply concatenated to the CSI input
  2. Evaluates gated, concatenation and quality-blind models on the in-distribution
     test set (clean, mild, moderate, severe) and on the pure unseen-user,
     unseen-environment and unseen-device subsets (clean, severe).
  3. Test-time quality-channel interventions for the two quality-aware models
     (no retraining):
       - nominal_quality  : impaired CSI, but the quality channels of the CLEAN version
                            of the same example (the model is told nothing is wrong)
       - shuffled_quality : quality channels taken from other examples of the same batch
  4. Paired bootstrap (resampling test examples, averaging over seeds) of macro-F1
     differences between variants and between interventions.
Resumable: finished trainings and evaluations are skipped; results are zipped regularly.
"""

# ============================ SETTINGS ============================
SMOKE = False            # True = short pipeline test (tiny data, 1 epoch)
import os as _os
SMOKE = SMOKE or _os.environ.get("ABL_SMOKE") == "1"
SEEDS = [2029, 2030, 2031]
TRAIN_VARIANTS = ("augmented_cnn_lstm", "quality_cnn_lstm")
REFERENCE = "gated_quality_cnn_lstm"
ID_SCENARIOS = ("clean", "mixed_mild", "mixed_moderate", "mixed_severe")
OOD_SCENARIOS = ("clean", "mixed_severe")
INTERVENTIONS = ("nominal_quality", "shuffled_quality")
REPLICATE_SEED = 2026    # must match the manuscript's test evaluation
TRAIN_WORKERS = 2        # as in the original training runs
EVAL_WORKERS = 4
BOOTSTRAP = 2000
EVAL_LIMIT = 64 if SMOKE else None
# =================================================================

import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

INPUT = Path(os.environ.get("ABL_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ABL_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("ABL_TMP", "/tmp/wifi_csi_abl_cache"))  # not saved as notebook output
OUT = WORK / ("ablation_smoke" if SMOKE else "ablation")
ZIP_BASE = WORK / ("wifi_csi_ablation_SMOKE_results" if SMOKE else "wifi_csi_ablation_results")
for sub in ("train", "records", "predictions", "_indexes"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)


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
try:
    OOD = extract_if_needed("csi_bench_test_cross_user_cache_index.jsonl", True, TMP / "ood")
except AssertionError:
    OOD = None
IMPAIRMENTS = CODE / "configs" / "impairments_v1.toml"
TRAIN_INDEX = CORE / "csi_bench_train_id_cache_index.jsonl"
VAL_INDEX = CORE / "csi_bench_val_id_cache_index.jsonl"
TEST_INDEX = CORE / "csi_bench_test_id_cache_index.jsonl"
for p in (IMPAIRMENTS, TRAIN_INDEX, VAL_INDEX, TEST_INDEX):
    assert p.exists(), f"Eksik dosya: {p}"
log("Kod:", CODE)
log("Çekirdek önbellek:", CORE)
log("OOD önbellek:", OOD if OOD else "YOK (saf OOD alt kümeleri atlanacak)")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from wifi_csi_quality.learning import CNNLSTMBaseline, metrics_from_outputs  # noqa: E402
from wifi_csi_quality.quality_learning import (  # noqa: E402
    AugmentedCacheDataset, GatedQualityCNNLSTM, QualityAwareCNNLSTM)
from wifi_csi_quality.robustness import load_impairment_scenarios  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIOS = load_impairment_scenarios(IMPAIRMENTS)
log("Cihaz:", DEVICE)


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def make_zip():
    archive = shutil.make_archive(str(ZIP_BASE), "zip", OUT)
    log("Arşiv güncellendi:", archive)


# ---------------------------------------------------------------- 1. training
TOML = """[training]
model = "cnn_lstm"
seed = {seed}
epochs = 30
batch_size = 96
learning_rate = 0.0006662034366198896
weight_decay = 7.66276127696697e-05
dropout = 0.30
early_stopping_patience = 6
num_workers = {workers}
calibration_bins = 15
class_weighting = false
"""


def train(model_name, seed):
    out_dir = OUT / "train" / model_name / f"seed_{seed}"
    result = out_dir / f"{model_name}_results.json"
    best = out_dir / f"{model_name}_best.pt"
    if result.exists() and best.exists():
        log(f"Atlandı (zaten eğitildi): {model_name} seed {seed}")
        return best
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "training.toml"
    cfg.write_text(TOML.format(seed=seed, workers=TRAIN_WORKERS), encoding="utf-8")
    cmd = [sys.executable, "-u", "-m", "wifi_csi_quality.quality_learning",
           "--cache-root", str(CORE), "--train-index", str(TRAIN_INDEX), "--val-index", str(VAL_INDEX),
           "--impairments-config", str(IMPAIRMENTS), "--training-config", str(cfg),
           "--output-dir", str(out_dir), "--model", model_name, "--resume"]
    if SMOKE:
        cmd.append("--smoke")
    log(f"===== EĞİTİM: {model_name} seed {seed} =====")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "train_log.txt").write_text(proc.stdout[-20000:] + "\n" + proc.stderr[-20000:], encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"Eğitim başarısız: {model_name} seed {seed}")
    res = json.loads(result.read_text(encoding="utf-8"))
    bv = max(res["history"], key=lambda r: r["validation"]["macro_f1"])["validation"]
    log(f"Bitti {model_name} seed {seed}: {(time.time()-t0)/60:.1f} dk, best epoch {res['best_epoch']}, "
        f"val macro-F1 {100*bv['macro_f1']:.2f}, parametre {res['parameter_count']}")
    lastp = out_dir / f"{model_name}_last.pt"
    if lastp.exists():
        lastp.unlink()   # only needed for resuming an interrupted run
    make_zip()
    return best


def _ckpt_seed(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return int(ckpt["training_config"]["seed"])
    except Exception as exc:  # unreadable file: ignore
        log(f"Uyarı: {path} okunamadı ({exc})")
        return None


INPUT_CKPTS = {}
for _p in sorted(INPUT.rglob(f"{REFERENCE}_best.pt"), key=lambda p: len(str(p))):
    _s = _ckpt_seed(_p)          # seed is read from the checkpoint itself, not from folder names
    if _s is not None and _s not in INPUT_CKPTS:
        INPUT_CKPTS[_s] = _p
log("Girdilerde bulunan kapılı checkpoint'ler:", {s: str(p) for s, p in INPUT_CKPTS.items()} or "YOK")


def reference_checkpoint(seed):
    if seed in INPUT_CKPTS and not SMOKE:
        return INPUT_CKPTS[seed], "manuscript"
    if not SMOKE:
        log(f"UYARI: seed {seed} için makale checkpoint'i bulunamadı; kapılı model yeniden eğitilecek.")
    return train(REFERENCE, seed), "retrained"


CHECKPOINTS = {}
for seed in SEEDS:
    CHECKPOINTS[(REFERENCE, seed)] = reference_checkpoint(seed)
    log(f"Kapılı model seed {seed}: {CHECKPOINTS[(REFERENCE, seed)][1]} -> {CHECKPOINTS[(REFERENCE, seed)][0]}")
for name in TRAIN_VARIANTS:
    for seed in SEEDS:
        CHECKPOINTS[(name, seed)] = (train(name, seed), "trained_here")


# ---------------------------------------------------------------- 2. evaluation splits
SPLITS = {"in_distribution_test": (TEST_INDEX, CORE, ID_SCENARIOS)}
if OOD is not None:
    seen = {"user": set(), "environment": set(), "device": set()}
    for r in read_rows(TRAIN_INDEX):
        for k in seen:
            seen[k].add(r[k])
    for split, factor in (("csi_bench_test_cross_user", "user"),
                          ("csi_bench_test_cross_env", "environment"),
                          ("csi_bench_test_cross_device", "device")):
        idx = OOD / f"{split}_cache_index.jsonl"
        if not idx.exists():
            continue
        rows = [r for r in read_rows(idx) if r[factor] not in seen[factor]
                and all(r[o] in seen[o] for o in seen if o != factor)]
        if rows:
            name = f"pure_unseen_{factor}"
            SPLITS[name] = (write_rows(rows, OUT / "_indexes" / f"{name}.jsonl"), OOD, OOD_SCENARIOS)
for name, (idx, _r, sc) in SPLITS.items():
    log(f"Bölüm {name:28s} n={len(read_rows(idx)):6d} senaryolar={list(sc)}")


def load_model(name, seed):
    path, _ = CHECKPOINTS[(name, seed)]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["model_name"] == name, f"{path}: beklenmeyen model {ckpt['model_name']}"
    dropout = float(ckpt["training_config"]["dropout"])
    model = {"augmented_cnn_lstm": CNNLSTMBaseline, "quality_cnn_lstm": QualityAwareCNNLSTM,
             "gated_quality_cnn_lstm": GatedQualityCNNLSTM}[name](dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval(), name != "augmented_cnn_lstm", int(ckpt["training_config"]["calibration_bins"])


def loader(index, root, scenario):
    ds = AugmentedCacheDataset(index, root, SCENARIOS, [scenario], REPLICATE_SEED, "test",
                               sample_rate_hz=100.0, limit=EVAL_LIMIT)
    return DataLoader(ds, batch_size=min(256, len(ds)), shuffle=False, num_workers=EVAL_WORKERS)


@torch.no_grad()
def predict(model, uses_quality, index, root, scenario, mode):
    logits, labels = [], []
    main = loader(index, root, scenario)
    clean = loader(index, root, "clean") if mode == "nominal_quality" else None
    gen = torch.Generator().manual_seed(12345)
    batches = zip(main, clean) if clean is not None else ((b, None) for b in main)
    for (values, quality, y), other in batches:
        if mode == "nominal_quality":
            _cv, quality, cy = other
            assert torch.equal(cy, y), "temiz ve bozulmuş yükleyiciler hizalı değil"
        elif mode == "shuffled_quality":
            quality = quality[torch.randperm(quality.shape[0], generator=gen)]
        values = values.to(DEVICE)
        out = model(values, quality.to(DEVICE)) if uses_quality else model(values)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


# ---------------------------------------------------------------- 3. run evaluations
MODELS = (REFERENCE, "quality_cnn_lstm", "augmented_cnn_lstm")
records = []
t0 = time.time()
for name in MODELS:
    for seed in SEEDS:
        model, uses_q, bins = load_model(name, seed)
        modes = ("standard",) + (INTERVENTIONS if uses_q else ())
        for split, (idx, root, scs) in SPLITS.items():
            for scenario in scs:
                for mode in modes:
                    if mode != "standard" and scenario == "clean" and split != "in_distribution_test":
                        continue  # interventions are uninformative on clean OOD data; keep ID clean as control
                    key = f"{name}__{mode}__{split}__{scenario}__seed{seed}"
                    rec_path = OUT / "records" / f"{key}.json"
                    if rec_path.exists():
                        records.append(json.loads(rec_path.read_text(encoding="utf-8")))
                        continue
                    lg, y = predict(model, uses_q, idx, root, scenario, mode)
                    m = metrics_from_outputs(lg, y, bins=bins)
                    rec = {"model": name, "quality_mode": mode, "split": split, "scenario": scenario,
                           "seed": seed, "n": int(y.numel()), "checkpoint_source": CHECKPOINTS[(name, seed)][1],
                           **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll")},
                           "per_class_recall": m["per_class_recall"], "confusion_matrix": m["confusion_matrix"]}
                    np.savez_compressed(OUT / "predictions" / f"{key}.npz",
                                        y=y.numpy().astype(np.int8), pred=lg.argmax(1).numpy().astype(np.int8))
                    rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
                    records.append(rec)
                    log(f"{name:24s} s{seed} {mode:16s} {split:28s} {scenario:15s} "
                        f"F1={100*rec['macro_f1']:6.2f} bal={100*rec['balanced_accuracy']:6.2f} ECE={rec['ece']:.4f}")
        make_zip()
log(f"Değerlendirme süresi: {(time.time()-t0)/60:.1f} dk")


# ---------------------------------------------------------------- 4. summaries
(OUT / "all_results.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
agg = defaultdict(list)
for r in records:
    agg[(r["model"], r["quality_mode"], r["split"], r["scenario"])].append(r)
with open(OUT / "summary_mean_sd.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "quality_mode", "split", "scenario", "n", "seeds",
                "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "balanced_accuracy_sd",
                "macro_f1_mean", "macro_f1_sd", "ece_mean", "ece_sd", "nll_mean", "nll_sd"])
    for (model, mode, split, scenario), rs in agg.items():
        row = [model, mode, split, scenario, rs[0]["n"], len(rs)]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll"):
            v = np.array([r[k] for r in rs], float) * (1 if k in ("ece", "nll") else 100)
            row += [f"{v.mean():.4f}", f"{v.std(ddof=1):.4f}" if len(v) > 1 else ""]
        w.writerow(row)

# robustness: drop from clean to each impaired scenario, per model (standard inputs)
with open(OUT / "robustness_drop.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "split", "scenario", "macro_f1_mean", "drop_vs_clean_pp_mean", "drop_vs_clean_pp_sd"])
    for model in MODELS:
        for split, (_i, _r, scs) in SPLITS.items():
            clean = {r["seed"]: r["macro_f1"] for r in agg[(model, "standard", split, "clean")]}
            for sc in scs:
                cur = {r["seed"]: r["macro_f1"] for r in agg[(model, "standard", split, sc)]}
                d = np.array([100 * (cur[s] - clean[s]) for s in cur if s in clean])
                w.writerow([model, split, sc, f"{100*np.mean(list(cur.values())):.2f}",
                            f"{d.mean():.2f}", f"{d.std(ddof=1):.2f}" if len(d) > 1 else ""])


def mf1(y, p):
    cm = np.bincount(y.astype(np.int64) * 4 + p.astype(np.int64), minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return f1.mean()


def load_pred(model, mode, split, scenario, seed):
    d = np.load(OUT / "predictions" / f"{model}__{mode}__{split}__{scenario}__seed{seed}.npz")
    return d["y"], d["pred"]


COMPARISONS = [  # (A, B): difference A - B
    ((REFERENCE, "standard"), ("augmented_cnn_lstm", "standard")),
    ((REFERENCE, "standard"), ("quality_cnn_lstm", "standard")),
    (("quality_cnn_lstm", "standard"), ("augmented_cnn_lstm", "standard")),
    ((REFERENCE, "standard"), (REFERENCE, "nominal_quality")),
    ((REFERENCE, "standard"), (REFERENCE, "shuffled_quality")),
    (("quality_cnn_lstm", "standard"), ("quality_cnn_lstm", "nominal_quality")),
    (("quality_cnn_lstm", "standard"), ("quality_cnn_lstm", "shuffled_quality")),
]
rng = np.random.default_rng(0)
with open(OUT / "paired_differences.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["A_model", "A_mode", "B_model", "B_mode", "split", "scenario", "n", "seeds",
                "delta_macro_f1_pp_mean", "delta_sd_over_seeds", "ci95_low_pp", "ci95_high_pp", "p_one_sided_A_le_B"])
    for (am, amode), (bm, bmode) in COMPARISONS:
        for split, (_i, _r, scs) in SPLITS.items():
            for sc in scs:
                pairs = []
                for s in SEEDS:
                    try:
                        ya, pa = load_pred(am, amode, split, sc, s)
                        yb, pb = load_pred(bm, bmode, split, sc, s)
                    except FileNotFoundError:
                        continue
                    assert np.array_equal(ya, yb)
                    pairs.append((ya, pa, pb))
                if not pairs:
                    continue
                deltas = np.array([100 * (mf1(y, a) - mf1(y, b)) for y, a, b in pairs])
                n = len(pairs[0][0])
                boot = np.empty(BOOTSTRAP)
                for i in range(BOOTSTRAP):
                    idx = rng.integers(0, n, n)
                    boot[i] = np.mean([100 * (mf1(y[idx], a[idx]) - mf1(y[idx], b[idx])) for y, a, b in pairs])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                w.writerow([am, amode, bm, bmode, split, sc, n, len(pairs), f"{deltas.mean():.2f}",
                            f"{deltas.std(ddof=1):.2f}" if len(deltas) > 1 else "", f"{lo:.2f}", f"{hi:.2f}",
                            f"{np.mean(boot <= 0):.4f}"])
log("Eşleştirilmiş farklar hesaplandı.")

# ---------------------------------------------------------------- 5. sanity check and report
ref = [r["macro_f1"] for r in agg[(REFERENCE, "standard", "in_distribution_test", "clean")]]
if ref and not SMOKE:
    m = 100 * np.mean(ref)
    src = {CHECKPOINTS[(REFERENCE, s)][1] for s in SEEDS}
    log(f"Kontrol: kapılı model ID temiz macro-F1 = {m:.2f} (makale: 74.16; kaynak: {src})")
print("\nÖZET (macro-F1 %, ortalama ± SS, standart girdiler)")
for split, (_i, _r, scs) in SPLITS.items():
    for sc in scs:
        cells = []
        for model in MODELS:
            v = 100 * np.array([r["macro_f1"] for r in agg[(model, "standard", split, sc)]])
            cells.append(f"{model.split('_')[0]:9s} {v.mean():6.2f}±{v.std(ddof=1) if len(v) > 1 else 0:4.2f}")
        print(f"  {split:28s} {sc:15s} | " + " | ".join(cells))
make_zip()
log("BİTTİ.")
