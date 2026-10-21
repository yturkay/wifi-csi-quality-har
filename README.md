# wifi-csi-quality-har

Code, configurations, and aggregate results for the study

> **Quality-Aware Gated CNN–LSTM for Robust Wi-Fi CSI Human Activity Recognition: Cross-Dataset Evidence from CSI-Bench and XRF55**
> Yavuz Türkay (manuscript under review, 2026).

The repository contains everything needed to reproduce the experiments from the
public raw datasets:

- a harmonized four-class protocol (jumping, running, walking, hand waving) that
  preserves the official CSI-Bench in-distribution splits and keeps XRF55 as an
  untouched external test set;
- deterministic simulation of acquisition impairments (random loss, burst loss,
  timing jitter, stale samples) with four per-step quality channels
  (observed, stale, Δt/Δnom, sample age);
- the quality-gated CNN–LSTM (101,092 parameters) and the two ablation variants
  (quality-blind and concatenation);
- scripts for the in-distribution, out-of-distribution, leave-one-user-out /
  leave-one-environment-out, XRF55 (zero-shot, receiver fusion, AdaBN), and
  ablation experiments;
- the aggregate results behind every table of the paper.

The raw datasets, the preprocessed caches, and the trained weights are **not**
included (see *Data and licenses*).

## Repository layout

```
src/wifi_csi_quality/    library: readers, protocol, preprocessing, impairments, models, training, evaluation
configs/                 protocol, impairment scenarios, training configurations (paper: training_gated_seed*.toml)
scripts/run_preprocessing.sh       build manifests and caches from the raw datasets
scripts/evaluate_ood.py            OOD, XRF55, receiver-fusion and AdaBN evaluation (Sections 3.4–3.6)
scripts/kaggle/                    notebooks/scripts used on Kaggle (Optuna + training, repeat seeds, LOO, ablation)
results/                           aggregate results reported in the paper
tests/                             unit tests
```

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest            # unit tests
```

The experiments in the paper were run with PyTorch 2.11 and CUDA 12.8 on one
NVIDIA Tesla T4 GPU (Kaggle). CPU execution works but is slow for training.

## Data and licenses

Download the datasets from their providers and keep them read-only:

| Dataset | Reference | Distribution used | License |
|---|---|---|---|
| CSI-Bench | Zhu *et al.*, NeurIPS 2025, vol. 38, doi:10.52202/085713-5648 | https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench | CC BY-NC-ND 4.0 |
| XRF55 (Wi-Fi raw data) | Wang *et al.*, Proc. ACM IMWUT 8(1), 2024, doi:10.1145/3643543 | https://www.kaggle.com/datasets/xrfdataset/xrf55-rawdata (v1) | CC BY-NC 4.0 |

Because CSI-Bench is licensed CC BY-NC-ND 4.0, derived files (preprocessed
caches and trained model weights) are not redistributed here. They can be
regenerated deterministically with the code below; the trained checkpoints are
available from the author on reasonable request for non-commercial research.

## Reproducing the paper

### 1. Protocol manifests and caches (Section 2.3)

```bash
cp configs/datasets.example.toml configs/datasets.local.toml   # set the two dataset paths
bash scripts/run_preprocessing.sh outputs
```

This writes the manifests (`outputs/manifests`), the core cache with the
CSI-Bench in-distribution splits and the 9,360 XRF55 external examples
(`outputs/cache/core`), the CSI-Bench OOD cache (`outputs/cache/ood`), and a
cache audit. Each example is a 500 × 64 float32 tensor (5 s at 100 Hz,
64 subcarriers).

### 2. Hyperparameter search and training (Sections 2.6, 3.1)

The Optuna search (12 trials) and the final three-seed training were run on
Kaggle with `scripts/kaggle/kaggle_optuna_and_training.ipynb` and
`scripts/kaggle/kaggle_repeat_seeds.py`. The equivalent local command for one
seed is

```bash
python -m wifi_csi_quality.quality_learning \
  --cache-root outputs/cache/core \
  --train-index outputs/cache/core/csi_bench_train_id_cache_index.jsonl \
  --val-index   outputs/cache/core/csi_bench_val_id_cache_index.jsonl \
  --impairments-config configs/impairments_v1.toml \
  --training-config configs/training_gated_seed2029.toml \
  --model gated_quality_cnn_lstm \
  --output-dir outputs/train/gated/seed_2029 --resume
```

Repeat with seeds 2030 and 2031. Training uses all four impairment scenarios
(clean, mild, moderate, severe) as augmentation.

### 3. In-distribution robustness (Section 3.3, Table 7)

```bash
python -m wifi_csi_quality.quality_robustness \
  --cache-root outputs/cache/core \
  --test-index outputs/cache/core/csi_bench_test_id_cache_index.jsonl \
  --checkpoint outputs/train/gated/seed_2029/gated_quality_cnn_lstm_best.pt \
  --impairments-config configs/impairments_v1.toml \
  --scenarios clean mixed_mild mixed_moderate mixed_severe \
  --output outputs/eval/gated_seed2029_id.json
```

Impairments are generated deterministically from the example identifier and the
replicate seed (default 2026), so every model sees exactly the same corrupted inputs.

### 4. OOD, XRF55, receiver fusion, AdaBN (Sections 3.4–3.6, Tables 8–10)

```bash
python scripts/evaluate_ood.py \
  --code-root . --checkpoint-root outputs/train/gated \
  --core-cache outputs/cache/core --ood-cache outputs/cache/ood \
  --output-dir outputs/ood_evaluation
```

On the in-distribution test set this script reproduces the macro-F1 of 74.16% (Table 7).

### 5. Leave-one-user-out / leave-one-environment-out (Section 3.7, Table 11)

`scripts/kaggle/kaggle_leave_one_out.py` (one Kaggle notebook cell; set
`SMOKE = True` for a short pipeline test). It pools all four-class CSI-Bench
examples, builds the 14 folds (10 unique test sets), trains one model per unique
fold with the paper configuration (seed 2029), and evaluates each held-out domain.

### 6. Ablation of quality conditioning (Section 3.8, Tables 12–13)

`scripts/kaggle/kaggle_ablation.py` trains the quality-blind
(`augmented_cnn_lstm`, 95,940 parameters) and concatenation
(`quality_cnn_lstm`, 97,732 parameters) variants with seeds 2029–2031, evaluates
them together with the gated checkpoints, runs the nominal- and shuffled-quality
test-time interventions, and computes paired bootstrap confidence intervals.
Expected inputs and run times are documented in the script header.

Note: the Kaggle scripts print progress messages partly in Turkish; the code and
outputs are otherwise language-independent.

## Results in this repository

| Folder | Content | Paper |
|---|---|---|
| `results/main_training/` | Training histories and configurations of the three gated models | Section 3.1, Table 5 |
| `results/ood_and_xrf55/` | In-distribution, OOD, pure-subset, XRF55, receiver-fusion and AdaBN metrics (mean ± SD and per seed) | Tables 7–10, Figures 5–8 |
| `results/leave_one_out/` | Per-fold and per-factor leave-one-out metrics | Table 11, Figure 9 |
| `results/ablation/` | Variant and intervention metrics, severity drops, paired bootstrap differences, training histories | Tables 12–13, Figure 10 |

## Citation

If you use this code, please cite the paper (reference to be updated on
publication) and this software release (see `CITATION.cff`; the archived version
has a Zenodo DOI).

## License

Code: MIT (see `LICENSE`). Datasets: see the licenses of their providers above.
