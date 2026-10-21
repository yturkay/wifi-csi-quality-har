"""Out-of-distribution and adaptation evaluation for the quality-gated CNN-LSTM.

Evaluates the three trained checkpoints (seeds 2029-2031) WITHOUT retraining on:

  * CSI-Bench in-distribution test (sanity check: must reproduce Table 6/7)
  * CSI-Bench official OOD splits: cross_user, cross_env, cross_device,
    cross_user_env, plus "pure" subsets in which only one factor is unseen
  * XRF55 zero-shot: all repetitions (reproduces Table 8) and repetitions 18-20
  * XRF55 receiver fusion: probabilities averaged over the 3 receiver files of
    one action instance
  * XRF55 AdaBN: BatchNorm statistics re-estimated on UNLABELLED XRF55
    repetitions 1-14, evaluated on repetitions 18-20

Works on Kaggle (paths auto-discovered under /kaggle/input and
/kaggle/working) and locally (pass paths explicitly). Uses the project's own
model, impairment and metric code so that numbers are directly comparable.

Example (local):
  python scripts/evaluate_ood.py \
     --code-root . \
     --checkpoint-root /path/to/checkpoints \
     --core-cache /path/to/cache/core \
     --ood-cache /path/to/cache/ood \
     --output-dir outputs/ood_evaluation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SEEDS_DEFAULT = (2029, 2030, 2031)
OOD_SPLITS = (
    "csi_bench_test_cross_user",
    "csi_bench_test_cross_env",
    "csi_bench_test_cross_device",
    "csi_bench_test_cross_user_env",
)
SCENARIOS_ALL = ("clean", "mixed_mild", "mixed_moderate", "mixed_severe")


# --------------------------------------------------------------------------- paths
def _first(paths: list[Path], what: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Bulunamadı / not found: {what}")
    return sorted(paths, key=lambda p: len(str(p)))[0]


def _extract(archive: Path, target: Path) -> Path:
    import shutil
    import tarfile
    target.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {target}", flush=True)
    if archive.suffix == ".bin" or archive.name.endswith(".tar"):
        with tarfile.open(archive, "r:*") as handle:
            try:
                handle.extractall(target, filter="data")
            except TypeError:  # Python < 3.12
                handle.extractall(target)
    else:
        shutil.unpack_archive(str(archive), str(target))
    return target


def _archives(root: Path, want_ood: bool) -> list[Path]:
    hits = []
    for p in root.rglob("*"):
        name = p.name.lower()
        if p.is_file() and "cache" in name and (p.suffix in {".zip", ".tar", ".bin"} or name.endswith(".tar.gz")):
            if ("ood" in name) == want_ood:
                hits.append(p)
    return hits


def discover(args: argparse.Namespace) -> argparse.Namespace:
    kaggle_in, kaggle_work = Path("/kaggle/input"), Path("/kaggle/working")
    roots = [p for p in (kaggle_in, kaggle_work) if p.exists()]
    if kaggle_in.exists():
        # unpack archived caches once (Kaggle input folders are read-only)
        if args.core_cache is None and not any(next(r.rglob("csi_bench_test_id_cache_index.jsonl"), None) for r in roots):
            arch = _archives(kaggle_in, want_ood=False)
            if arch:
                _extract(arch[0], kaggle_work / "wifi_csi_search" / "cache")
        if args.ood_cache is None and not any(next(r.rglob("csi_bench_test_cross_user_cache_index.jsonl"), None) for r in roots):
            arch = _archives(kaggle_in, want_ood=True)
            if arch:
                _extract(arch[0], kaggle_work / "ood_cache")
    if args.code_root is None:
        args.code_root = _first(
            [p.parent for r in roots for p in r.rglob("pyproject.toml") if (p.parent / "src" / "wifi_csi_quality").exists()],
            "code package (pyproject.toml + src/wifi_csi_quality)",
        )
    if args.core_cache is None:
        args.core_cache = _first(
            [p.parent for r in roots for p in r.rglob("csi_bench_test_id_cache_index.jsonl")],
            "core cache (csi_bench_test_id_cache_index.jsonl)",
        )
    if args.ood_cache is None:
        hits = [p.parent for r in roots for p in r.rglob("csi_bench_test_cross_user_cache_index.jsonl")]
        args.ood_cache = _first(hits, "OOD cache") if hits else None
    if args.checkpoint_root is None:
        cps = [p for r in roots for p in r.rglob("gated_quality_cnn_lstm_best.pt")]
        args.checkpoint_root = _first([p.parent.parent for p in cps], "checkpoints (gated_quality_cnn_lstm_best.pt)")
    if args.output_dir is None:
        base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        args.output_dir = base / "ood_evaluation"
    if args.impairments_config is None:
        args.impairments_config = Path(args.code_root) / "configs" / "impairments_v1.toml"
    return args


def find_checkpoint(root: Path, seed: int) -> Path:
    hits = [p for p in Path(root).rglob("gated_quality_cnn_lstm_best.pt") if str(seed) in str(p.parent)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Seed {seed}: expected 1 checkpoint under {root}, found {hits}")
    return hits[0]


# --------------------------------------------------------------------------- index helpers
def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def xrf_instance(row: dict[str, Any]) -> str:
    return f"{row['scene']}|{row['volunteer']}|{int(row['action_id'])}|{int(row['repetition'])}"


# --------------------------------------------------------------------------- evaluation core
class Runner:
    def __init__(self, args: argparse.Namespace):
        sys.path.insert(0, str(Path(args.code_root) / "src"))
        import torch  # noqa: WPS433
        from torch.utils.data import DataLoader

        from wifi_csi_quality.learning import metrics_from_outputs
        from wifi_csi_quality.quality_learning import AugmentedCacheDataset, GatedQualityCNNLSTM
        from wifi_csi_quality.robustness import load_impairment_scenarios

        self.torch = torch
        self.DataLoader = DataLoader
        self.metrics_from_outputs = metrics_from_outputs
        self.Dataset = AugmentedCacheDataset
        self.Model = GatedQualityCNNLSTM
        self.scenarios = load_impairment_scenarios(Path(args.impairments_config))
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.models: dict[int, Any] = {}
        self.bins: dict[int, int] = {}
        for seed in args.seeds:
            path = find_checkpoint(Path(args.checkpoint_root), seed)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if ckpt["model_name"] != "gated_quality_cnn_lstm":
                raise ValueError(f"{path}: unexpected model {ckpt['model_name']}")
            model = self.Model(dropout=float(ckpt["training_config"]["dropout"]))
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            self.models[seed] = model
            self.bins[seed] = int(ckpt["training_config"]["calibration_bins"])
            print(f"[load] seed {seed}: {path} (best epoch {ckpt['best_epoch']})", flush=True)

    def _loader(self, index: Path, root: Path, scenario: str):
        dataset = self.Dataset(
            index, root, self.scenarios, [scenario],
            self.args.replicate_seed, "test", sample_rate_hz=self.args.sample_rate_hz,
            limit=self.args.limit,
        )
        loader = self.DataLoader(
            dataset, batch_size=min(self.args.batch_size, len(dataset)), shuffle=False,
            num_workers=self.args.num_workers,
        )
        return dataset, loader

    def predict(self, model, index: Path, root: Path, scenario: str):
        """Return (logits, labels, rows) in index order."""
        torch = self.torch
        dataset, loader = self._loader(index, root, scenario)
        logits, labels = [], []
        with torch.no_grad():
            for values, quality, label in loader:
                logits.append(model(values.to(self.device), quality.to(self.device)).float().cpu())
                labels.append(label)
        return torch.cat(logits), torch.cat(labels), dataset.rows

    def adabn(self, model, index: Path, root: Path):
        """Copy of model whose BatchNorm statistics are re-estimated on unlabelled data."""
        torch = self.torch
        adapted = copy.deepcopy(model)
        for module in adapted.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None  # cumulative average over all adaptation batches
        adapted.eval()
        for module in adapted.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
        _, loader = self._loader(index, root, "clean")
        with torch.no_grad():
            for values, quality, _label in loader:  # labels are never used
                adapted(values.to(self.device), quality.to(self.device))
        adapted.eval()
        return adapted


# --------------------------------------------------------------------------- bootstrap
def macro_f1_np(y: np.ndarray, p: np.ndarray, k: int = 4) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def grouped_bootstrap(y, pred, groups, reps: int, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of macro-F1, resampling whole groups (e.g. XRF55 action instances)."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        stats.append(macro_f1_np(y[idx], pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-root", type=Path)
    ap.add_argument("--checkpoint-root", type=Path)
    ap.add_argument("--core-cache", type=Path, help="folder with csi_bench_test_id_cache_index.jsonl and xrf55_external_test_cache_index.jsonl")
    ap.add_argument("--ood-cache", type=Path, help="folder with csi_bench_test_cross_*_cache_index.jsonl")
    ap.add_argument("--impairments-config", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--ood-scenarios", nargs="+", default=["clean", "mixed_severe"],
                    help="impairment scenarios for OOD and XRF55 splits (in-distribution test always uses all four)")
    ap.add_argument("--replicate-seed", type=int, default=2026, help="must match the original test evaluation (default 2026)")
    ap.add_argument("--sample-rate-hz", type=float, default=100.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int, help="smoke test: per-split balanced subset size")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-ood", action="store_true", help="only in-distribution + XRF55")
    args = discover(ap.parse_args())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    work = out / "_indexes"
    for name in ("code_root", "checkpoint_root", "core_cache", "ood_cache", "impairments_config"):
        print(f"[path] {name:18s} = {getattr(args, name)}")

    runner = Runner(args)
    print(f"[device] {runner.device}", flush=True)
    core = Path(args.core_cache)

    # ---------------- build evaluation index files
    train_rows_path = None
    for candidate in (core / "csi_bench_train_id_cache_index.jsonl",):
        if candidate.exists():
            train_rows_path = candidate
    splits: dict[str, tuple[Path, Path, tuple[str, ...]]] = {}
    splits["csi_bench_test_id"] = (core / "csi_bench_test_id_cache_index.jsonl", core, SCENARIOS_ALL)

    if not args.skip_ood:
        if args.ood_cache is None:
            print("[warn] OOD cache not found -> CSI-Bench OOD splits skipped. Run run_ood_preprocessing.sh first.")
        else:
            ood = Path(args.ood_cache)
            seen = {"user": set(), "environment": set(), "device": set()}
            if train_rows_path is not None:
                for r in read_rows(train_rows_path):
                    for k in seen:
                        seen[k].add(r[k])
            else:
                print("[warn] train index not found -> 'pure' OOD subsets skipped")
            for split in OOD_SPLITS:
                idx = ood / f"{split}_cache_index.jsonl"
                if not idx.exists():
                    print(f"[warn] missing {idx}; skipped")
                    continue
                splits[split] = (idx, ood, tuple(args.ood_scenarios))
            if train_rows_path is not None:
                # pure subsets: exactly one factor unseen, the other two seen in training
                for split, factor in (("csi_bench_test_cross_user", "user"),
                                      ("csi_bench_test_cross_env", "environment"),
                                      ("csi_bench_test_cross_device", "device")):
                    idx = ood / f"{split}_cache_index.jsonl"
                    if not idx.exists():
                        continue
                    rows = [r for r in read_rows(idx)
                            if r[factor] not in seen[factor]
                            and all(r[o] in seen[o] for o in seen if o != factor)]
                    if rows:
                        name = f"{split}__pure_{factor}"
                        splits[name] = (write_rows(rows, work / f"{name}.jsonl"), ood, tuple(args.ood_scenarios))
                    else:
                        print(f"[info] {split}: no pure-{factor} subset")
    xrf_idx = core / "xrf55_external_test_cache_index.jsonl"
    xrf_rows = read_rows(xrf_idx) if xrf_idx.exists() else []
    adapt_index = None
    if xrf_rows:
        splits["xrf55_all_reps"] = (xrf_idx, core, tuple(args.ood_scenarios))
        test_rows = [r for r in xrf_rows if 18 <= int(r["repetition"]) <= 20]
        adapt_rows = [r for r in xrf_rows if 1 <= int(r["repetition"]) <= 14]
        splits["xrf55_reps_18_20"] = (write_rows(test_rows, work / "xrf55_reps_18_20.jsonl"), core, tuple(args.ood_scenarios))
        adapt_index = write_rows(adapt_rows, work / "xrf55_reps_01_14_unlabelled.jsonl")
    else:
        print("[warn] XRF55 external cache index not found -> XRF55 skipped")

    for name, (idx, _root, sc) in splits.items():
        print(f"[split] {name:42s} n={sum(1 for _ in open(idx, encoding='utf-8')):6d} scenarios={list(sc)}")

    # ---------------- evaluate
    records: list[dict[str, Any]] = []
    pred_dir = out / "predictions"; pred_dir.mkdir(exist_ok=True)

    def record(seed, split, scenario, variant, logits, labels, rows):
        torch = runner.torch
        m = runner.metrics_from_outputs(logits, labels, bins=runner.bins[seed])
        rec = {"seed": seed, "split": split, "scenario": scenario, "variant": variant, "n": int(labels.numel()), **{
            k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll")},
            "per_class_f1": m["per_class_f1"], "per_class_recall": m["per_class_recall"],
            "confusion_matrix": m["confusion_matrix"]}
        y = labels.numpy(); pred = logits.argmax(1).numpy()
        groups = None
        if rows and "repetition" in rows[0] and variant.endswith("per_receiver"):
            groups = np.array([xrf_instance(r) for r in rows])   # 3 correlated receiver files per instance
        elif variant.endswith("receiver_fusion"):
            groups = np.array([r["sample_id"] for r in rows])     # one independent unit per instance
        if groups is not None:
            rec["macro_f1_ci95_instance_bootstrap"] = grouped_bootstrap(y, pred, groups, args.bootstrap, seed)
            rec["independent_units"] = int(len(set(groups)))
        records.append(rec)
        probs = torch.softmax(logits, 1).numpy()
        meta = [k for k in ("user", "environment", "device", "scene", "volunteer", "action_id", "repetition", "receiver")
                if rows and k in rows[0]]
        with (pred_dir / f"{split}__{scenario}__{variant}__seed{seed}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", "label", "pred", "p_jumping", "p_running", "p_walking", "p_wavinghand", *meta])
            for i in range(len(y)):
                r = rows[i] if rows else None
                sid = r["sample_id"] if r else f"instance_{i}"
                extra = [r[k] for k in meta] if r else []
                w.writerow([sid, int(y[i]), int(pred[i]), *[f"{v:.6f}" for v in probs[i]], *extra])
        print(f"  seed {seed} | {split:40s} | {scenario:15s} | {variant:16s} | n={rec['n']:6d} "
              f"| acc={100*rec['accuracy']:6.2f} bal={100*rec['balanced_accuracy']:6.2f} "
              f"F1={100*rec['macro_f1']:6.2f} ECE={rec['ece']:.4f}", flush=True)

    def fuse(logits, labels, rows):
        torch = runner.torch
        probs = torch.softmax(logits, 1)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[xrf_instance(r)].append(i)
        keys = sorted(groups)
        fused = torch.stack([probs[groups[k]].mean(0) for k in keys])
        fused_labels = torch.stack([labels[groups[k][0]] for k in keys])
        return torch.log(fused.clamp_min(1e-12)), fused_labels, [{"sample_id": k} for k in keys]

    t0 = time.time()
    for seed, model in runner.models.items():
        print(f"\n===== SEED {seed} =====", flush=True)
        for split, (idx, root, scs) in splits.items():
            for scenario in scs:
                logits, labels, rows = runner.predict(model, idx, root, scenario)
                record(seed, split, scenario, "per_receiver" if split.startswith("xrf55") else "standard", logits, labels, rows)
                if split.startswith("xrf55"):
                    fl, fy, frows = fuse(logits, labels, rows)
                    record(seed, split, scenario, "receiver_fusion", fl, fy, frows)
        if adapt_index is not None:
            adapted = runner.adabn(model, adapt_index, core)
            for scenario in args.ood_scenarios:
                logits, labels, rows = runner.predict(adapted, splits["xrf55_reps_18_20"][0], core, scenario)
                record(seed, "xrf55_reps_18_20", scenario, "adabn_per_receiver", logits, labels, rows)
                fl, fy, frows = fuse(logits, labels, rows)
                record(seed, "xrf55_reps_18_20", scenario, "adabn_receiver_fusion", fl, fy, frows)
    print(f"\n[time] {time.time()-t0:.1f} s")

    # ---------------- aggregate
    (out / "all_results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    agg = defaultdict(list)
    for r in records:
        agg[(r["split"], r["scenario"], r["variant"])].append(r)
    with (out / "summary_mean_sd.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "scenario", "variant", "n", "seeds", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean",
                    "balanced_accuracy_sd", "macro_f1_mean", "macro_f1_sd", "ece_mean", "ece_sd", "nll_mean", "nll_sd"])
        for (split, scenario, variant), rs in agg.items():
            row = [split, scenario, variant, rs[0]["n"], len(rs)]
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll"):
                v = np.array([r[k] for r in rs], dtype=float)
                scale = 1.0 if k in ("ece", "nll") else 100.0
                row += [f"{scale*v.mean():.4f}", f"{scale*v.std(ddof=1):.4f}" if len(v) > 1 else ""]
            w.writerow(row)
    print(f"[done] {out/'summary_mean_sd.csv'}")

    # sanity check against the manuscript (Table 6: 74.16 +- 0.84 macro-F1 on clean test_id)
    ref = [r["macro_f1"] for r in records
           if r["split"] == "csi_bench_test_id" and r["scenario"] == "clean" and r["n"] == 4040]
    if ref:
        mean = 100 * float(np.mean(ref))
        status = "OK" if abs(mean - 74.16) < 0.10 else "MISMATCH - check replicate seed / cache / checkpoints"
        print(f"[sanity] clean in-distribution macro-F1 = {mean:.2f} (manuscript 74.16) -> {status}")

    import shutil
    archive = shutil.make_archive(str(out), "zip", out)
    print(f"[archive] {archive}")


if __name__ == "__main__":
    main()
