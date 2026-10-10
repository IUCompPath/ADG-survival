# AI-Based Prognostic Risk Stratification of Adult-Type Diffuse Glioma Patients from H&E-Stained Slides

Code for Innani et al., *Cancers* 2026 — a systematic benchmark of pathology
foundation models (FMs) and multiple-instance-learning (MIL) methods for
overall-survival risk stratification in adult-type diffuse glioma (ADG) under
the WHO 2021 classification.

**84 encoder-MIL configurations** (11 FMs x 7 MIL methods, plus seven ResNet-50
baseline configurations) evaluated on **817 TCGA subjects** with patient-level
10-fold cross-validation, together with a perturbation-based assessment of
attention faithfulness.

Headline result: UNI2 + MambaMIL reached the highest cross-validated
concordance index (**C-index 0.77**), with several combinations performing
comparably. Risk stratification *within* individual WHO 2021 types was not
significant, and the score's discrimination largely reflects differences
*between* WHO 2021 types.

📄 [Paper](https://www.mdpi.com/2072-6694/18/20/3259) ·
[doi:10.3390/cancers18203259](https://doi.org/10.3390/cancers18203259)

The MIL backbone and Mamba CUDA kernels are adapted from
[MambaMIL](https://github.com/isyangshu/MambaMIL) (MICCAI 2024); patching and
feature-extraction utilities follow [CLAM](https://github.com/mahmoodlab/CLAM).

---

## Overview

| Stage | Entry point |
|-------|-------------|
| 1. Data and cross-validation folds | `dataset_csv/`, `splits/`, `tools/splitting.py` |
| 2. Patch extraction and FM features | `pipeline/extract_features.py` (see *Feature extraction*) |
| 3. MIL training (10-fold CV) | `pipeline/main_survival.py` |
| 4. Evaluation | `pipeline/eval_survival.py` |
| 5. Aggregation, c-index, CIs, statistics | `analysis/cindex_stats.py`, `analysis/survival_results_all_.py` |
| 6. Kaplan-Meier figures and median-OS tables | `analysis/plot_km_curves.py`, `analysis/median_survival_tables.py` |
| 7. Attention faithfulness | `analysis/robust_att.py`, `analysis/robustness_fm.py` |

---

## Environment

* Linux (tested on Ubuntu 22.04 / WSL2), NVIDIA GPU, CUDA 11.8
* Python 3.10, PyTorch 2.0.1

```bash
conda create -n adg_survival python=3.10 -y
conda activate adg_survival
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install packaging causal-conv1d==1.1.1
pip install -r requirements.txt

# Mamba CUDA kernels (bundled, compiled in place)
cd mamba && pip install . && cd ..
```

---

## 1. Data

**Cohort.** Diagnostic H&E WSIs for TCGA-GBM and TCGA-LGG from the
[GDC Data Portal](https://portal.gdc.cancer.gov/). Overall-survival endpoints
come from the TCGA clinical files; WHO CNS5 histology labels are in
`dataset_csv/df_2021_histology_labels.csv`.

After filtering to patients with both a diagnostic slide and survival data, the
analysis cohort is **817 patients**.

| File | Contents |
|------|----------|
| `dataset_csv/gbmlgg_processed_fixed.csv` | `case_id`, `slide_id`, `survival_months`, `censorship` |
| `dataset_csv/df_2021_histology_labels.csv` | WHO CNS5 subtype (oligodendroglioma / astrocytoma / glioblastoma) |
| `splits/tcga_gbmlgg_survival_kfold/` | 10 patient-level train/val/test folds |

Survival times are discretised into 4 quantile bins over uncensored patients
(`n_bins=4`), as in the standard TCGA survival-MIL setup.

> **Survival labels.** `gbmlgg_processed_fixed.csv` is the authoritative label
> file. An earlier version of this table carried two defects: `survival_months`
> disagreed with the day count for ~42% of patients (one case stored 117 months
> as 4.7), and `censorship` was inverted relative to the convention used by
> `nll_surv` and every c-index call. The corrected file was cross-checked
> against [PORPOISE](https://github.com/mahmoodlab/PORPOISE), LinkedOmics and
> [PathomicFusion](https://github.com/mahmoodlab/PathomicFusion), agreeing
> 735/735 on the overlapping patients. `censorship` is **1 = censored**.
>
> Cohort sizes: **817** patients have survival; the **656** with WHO 2021
> labels are exactly the 656 with extracted features. The 161 excluded
> patients are enriched for events (132/161, 82%) relative to the retained
> cohort (44%), so the exclusion is not random.

Folds are regenerated with:

```bash
python tools/splitting.py --task tcga_gbmlgg_survival --k 10 --seed 1
```

Splitting is **patient-level**: all slides from one patient stay in the same
fold, so no patient appears in both training and test.

---

## 2. Feature extraction

Tissue segmentation and patching follow CLAM (256x256 px at 20x). Each patch is
then embedded with one of the pathology foundation models (11 domain-specific
FMs plus an ImageNet-pretrained ResNet-50 baseline):

ResNet50 (ImageNet) · Lunit · GigaPath · CTransPath · Hibou · H-Optimus-1 ·
Virchow2 · CONCHv1.5 · UNI2 · CHIEF · PRISM · TITAN

Several are gated on Hugging Face. Request access, then:

```bash
export HF_TOKEN=<your_hugging_face_token>
```

Tokens are never stored in this repository.

Features are written as one `.pt` per slide under
`features/<fm_name>/pt_files/`, which is git-ignored.

---

## 3. Training

Ten-fold cross-validation, one run per (foundation model x MIL aggregator):

```bash
python pipeline/main_survival.py \
    --task          tcga_gbmlgg_survival \
    --data_root_dir features/uni_v2/pt_files \
    --split_dir     splits/tcga_gbmlgg_survival_kfold \
    --results_dir   results/tcga_gbmlgg_survival \
    --model_type    mamba_mil \
    --bag_loss      nll_surv \
    --in_dim        1024 \
    --k             10 \
    --seed          1
```

`--model_type` accepts `mean_mil`, `max_mil`, `att_mil`, `dsmil`, `trans_mil`,
`rrtmil`, `mamba_mil`. Set `--in_dim` to the feature dimension of the chosen
foundation model.

Each fold writes `s_<k>_checkpoint.pt` and `split_<k>_results.pkl`
(per-patient risk score, survival time, censorship).

---

## 4. Evaluation

```bash
python pipeline/eval_survival.py \
    --task            tcga_gbmlgg_survival \
    --data_root_dir   features/uni_v2/pt_files \
    --models_exp_code results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1 \
    --splits_dir      splits/tcga_gbmlgg_survival_kfold \
    --model_type      mamba_mil \
    --in_dim          1024 \
    --k               10 \
    --seed            1
```

> **Note.** `--data_root_dir`, `--seed` and the label column must match the
> training run exactly. Mismatches here silently change the c-index rather than
> raising an error.

---

## 5. Statistics

Pooled and fold-wise c-index with confidence intervals, per-fold variability,
and pairwise tests between the leading configurations:

```bash
python analysis/cindex_stats.py
```

Outputs (under `results/.../stats/`):

| File | Contents |
|------|----------|
| `cindex_with_ci.csv` | Pooled c-index + bootstrap CI; fold mean, SD, t-CI |
| `per_fold_cindex.csv` | Per-fold c-index for every configuration |
| `table_pooled_with_ci.csv` | FM x MIL grid, `c [95% CI]` |
| `table_foldwise_with_ci.csv` | FM x MIL grid, fold-wise convention |
| `leading_model_comparisons.csv` | Pairwise deltas, bootstrap/t/Wilcoxon p, Holm-adjusted |

Two conventions are reported, and they are **not interchangeable**:

* **Pooled** — concatenate the 10 disjoint test folds, compute one c-index over
  all 817 patients. CI by patient-level bootstrap (2000 resamples).
* **Fold-wise** — c-index within each fold, then mean +/- SD across folds.
  CI is t-based on the 10 fold values. *This is the convention used in the paper.*

---

## 6. Figures and tables

```bash
python analysis/plot_km_curves.py          # KM curves: all / oligo / astro / GBM
python analysis/median_survival_tables.py \
    --results results/.../combined_results.csv
```

`plot_km_curves.py` writes 12 figures (4 cohorts x {median split, tertile,
quartile}) as PNG and TIFF, with at-risk/censored tables beneath each panel.

---

## 7. Attention faithfulness

Tests whether attention weights identify patches the model actually relies on:
patches are ranked by attention, the top-X% progressively removed, and the
c-index recomputed. A faithful model degrades faster under attention-guided
removal than under random removal.

```bash
python analysis/robust_att.py      # attention-guided vs random patch ablation
python analysis/robustness_fm.py   # same sweep across foundation models
```

---

## Citation

If you use this code, please cite the paper and the upstream work it builds on:

```bibtex
@article{innani2026adg,
  author        = {Innani, Shubham and Pitarch-Abaigar, Carla and
                   Majeed, Marwan M. and Harmsen, Hannah and Bell, W. Robert and
                   Makris, Dimitrios and Bakas, Spyridon},
  title         = {{AI-Based Prognostic Risk Stratification of Adult-Type Diffuse
                   Glioma Patients from H\&E-Stained Slides}},
  journal       = {Cancers},
  volume        = {18},
  number        = {20},
  article-number = {3259},
  year          = {2026},
  issn          = {2072-6694},
  url           = {https://www.mdpi.com/2072-6694/18/20/3259},
  doi           = {10.3390/cancers18203259}
}
```

```bibtex
@inproceedings{yang2024mambamil,
  title     = {MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology},
  author    = {Yang, Shu and Wang, Yihui and Chen, Hao},
  booktitle = {MICCAI},
  year      = {2024}
}
```

## Acknowledgements

Built on [MambaMIL](https://github.com/isyangshu/MambaMIL) and
[CLAM](https://github.com/mahmoodlab/CLAM).

## License

MIT (inherited from MambaMIL).
