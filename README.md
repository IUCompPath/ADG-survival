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

## Pipeline

| # | Stage | Entry point |
|---|-------|-------------|
| 1 | Data and survival labels | `dataset_csv/` |
| 2 | Tissue segmentation and patching | [CLAM](https://github.com/mahmoodlab/CLAM) / [TRIDENT](https://github.com/mahmoodlab/TRIDENT) (external) |
| 3 | Patch-level feature extraction (11 FMs) | CLAM / TRIDENT (external) |
| 3b | Patient-level cross-validation folds | `tools/make_splits.py` |
| 4 | MIL training (7 aggregators) | `pipeline/main_survival.py` |
| 4b | Evaluation | `pipeline/eval_survival.py` |
| 5 | C-index, confidence intervals, statistics | `analysis/cindex_stats.py`, `analysis/survival_results_all_.py` |
| 6 | Kaplan-Meier curves and median-OS tables | `analysis/plot_km_curves.py`, `analysis/median_survival_tables.py` |
| 7 | UMAP of slide embeddings | `analysis/umap_embeddings.py` |
| 8 | Attention faithfulness (patch flipping) | `analysis/patch_flipping.py` |

Steps 2 and 3 are run with the upstream CLAM/TRIDENT tooling rather than code in
this repository; the sections below give the settings used in the paper.

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
python tools/make_splits.py \
    --csv       dataset_csv/gbmlgg_processed_fixed.csv \
    --histology dataset_csv/df_2021_histology_labels.csv \
    --features  features/<BACKBONE>/pt_files \
    --out       splits/tcga_gbmlgg_survival_kfold \
    --k 10 --seed 1
```

Splitting is **patient-level**: all slides from one patient stay in the same
fold, so no patient appears in both training and test.

---

## 2. Patching and feature extraction (CLAM / TRIDENT)

Slides are organised per dataset:

```
data/wsi/<DATASET>/
    ├── patient_1_slide_a.svs
    ├── patient_1_slide_b.svs
    └── ...
```

### 2a. Tissue segmentation and patching

Non-overlapping **256 x 256 px patches at 20x**, with background removed by
Otsu thresholding on the downsampled slide. TCGA mixes 20x- and 40x-native
slides, so the pyramid level is resolved per slide from its native objective
power; a slide with no level at the target is read at the finer level with a
proportionally larger window and resized, so every patch covers the same tissue
area.

```bash
python create_patches_fp.py \
    --source      data/wsi/<DATASET> \
    --save_dir    data/patches/<DATASET>/20x \
    --patch_size 256 --step_size 256 \
    --seg --patch --stitch
```

```
data/patches/<DATASET>/20x/
    ├── masks/        # segmentation overlays (QC)
    ├── patches/      # per-slide .h5 of patch coordinates
    ├── stitches/     # downsampled reconstructions (QC)
    └── slides_processed.csv
```

An optional cleanup pass removes low-quality tiles: patches with >85% white
space, patches failing HED colour-deconvolution stain detection, and blurry or
out-of-focus regions flagged by HSV thresholds.

### 2b. Patch-level feature extraction

Each patch is embedded with one of the **11 pathology foundation models**
benchmarked in the paper, plus an ImageNet-pretrained ResNet-50 baseline.

```bash
python extract_features_fp.py \
    --data_h5_dir   data/patches/<DATASET>/20x \
    --data_slide_dir data/wsi/<DATASET> \
    --csv_path      data/patches/<DATASET>/20x/slides_processed.csv \
    --feat_dir      features/<BACKBONE> \
    --model_name    <BACKBONE> --batch_size 256
```

```
features/<BACKBONE>/
    ├── h5_files/     # patch features + coordinates
    └── pt_files/     # serialised tensors, used for training
```

Training reads `pt_files/`. **`--in_dim` must match the backbone**, or the
state dict will not load:

| Foundation model | `--in_dim` | Source |
|------------------|-----------|--------|
| Lunit ViT   | 384  | [lunit-io/benchmark-ssl-pathology](https://github.com/lunit-io/benchmark-ssl-pathology) |
| CONCH v1    | 512  | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) |
| CTransPath  | 768  | [Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath) |
| CHIEF       | 768  | [hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF) |
| CONCH v1.5  | 768  | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) |
| TITAN       | 768  | [mahmoodlab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |
| ResNet-50   | 1024 | ImageNet pretrained (baseline) |
| Hibou       | 1024 | [HistAI/hibou](https://github.com/HistAI/hibou) |
| PRISM       | 1280 | [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) |
| GigaPath    | 1536 | [prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) |
| H-Optimus-1 | 1536 | [bioptimus/h-optimus](https://github.com/bioptimus/releases/tree/main/models/h-optimus) |
| UNI2        | 1536 | [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI) |
| Virchow2    | 2560 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |

Several are gated. Request access on each model page, then export a token
before extraction - tokens are never stored in this repository:

```bash
export HF_TOKEN=<your_hugging_face_token>
```

> Patching and feature extraction use the upstream
> [CLAM](https://github.com/mahmoodlab/CLAM) tooling.
> [TRIDENT](https://github.com/mahmoodlab/TRIDENT) is its successor and offers a
> more scalable pipeline for the same two stages; either can produce the
> `pt_files/` this repository consumes.

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

`--model_type` selects one of the **7 MIL aggregators** benchmarked in the paper:

| `--model_type` | Aggregator |
|----------------|------------|
| `mean_mil`  | Mean pooling |
| `max_mil`   | Max pooling |
| `att_mil`   | ABMIL (gated attention) |
| `dsmil`     | DSMIL |
| `trans_mil` | TransMIL |
| `rrtmil`    | RRT-MIL |
| `mamba_mil` | MambaMIL (SR-Mamba) |

Set `--in_dim` to the feature dimension of the chosen foundation model (see the
table in section 2).

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

## 6. Kaplan-Meier curves and median-OS tables

```bash
python analysis/plot_km_curves.py          # KM curves: all / oligo / astro / GBM
python analysis/median_survival_tables.py \
    --results results/.../combined_results.csv
```

`plot_km_curves.py` writes 12 figures (4 cohorts x {median split, tertile,
quartile}) as PNG and TIFF, with at-risk/censored tables beneath each panel.

---

## 7. UMAP of slide embeddings

Projects pooled slide embeddings to 2-D and colours them twice - once by
predicted risk group, once by WHO 2021 type. Comparing the panels shows whether
the embedding separates risk independently of subtype, or whether the apparent
risk structure is mostly subtype structure.

```bash
python analysis/umap_embeddings.py \
    --features  features/uni_v2/pt_files \
    --results   results/.../combined_results.csv \
    --histology dataset_csv/df_2021_histology_labels.csv \
    --in-dim    1536 \
    --pooling   mean \
    --out       results/umap
```

`--pooling gated` applies gated-attention pooling instead of an unweighted mean.
Requires `umap-learn`.

---

## 8. Attention faithfulness (patch flipping)

Tests whether attention weights identify the patches the model actually relies
on. Patches are ranked by attention and progressively removed, and the cohort
c-index is recomputed at each removal step. Three orderings are compared:

| Order | Patches removed first |
|-------|----------------------|
| `morf` | most relevant first (highest attention) |
| `morl` | most relevant last (lowest attention) |
| `random` | random baseline |

A faithful model degrades fastest under `morf`, with `random` in between.
Running `morf` alone cannot separate attention quality from the effect of simply
having a smaller bag, so all three arms are reported.

```bash
python analysis/patch_flipping.py \
    --features  features/uni_v2/pt_files \
    --ckpt-dir  results/.../uni_v2_s1 \
    --survival  results/.../combined_results.csv \
    --in-dim    1536 \
    --out       results/faithfulness
```

Outputs `patch_flipping_cindex.csv` and a curve per ordering. The protocol
follows [xMIL](https://github.com/bifold-pathomics/xMIL) (NeurIPS 2024), adapted
from classification to survival: the tracked quantity is the cohort c-index
rather than a target-class probability.

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
  title     = {{MambaMIL}: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology},
  author    = {Yang, Shu and Wang, Yihui and Chen, Hao},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2024}
}

@article{lu2021clam,
  title   = {Data-efficient and weakly supervised computational pathology on whole-slide images},
  author  = {Lu, Ming Y. and Williamson, Drew F. K. and Chen, Tiffany Y. and
             Chen, Richard J. and Barbieri, Matteo and Mahmood, Faisal},
  journal = {Nature Biomedical Engineering},
  volume  = {5},
  number  = {6},
  pages   = {555--570},
  year    = {2021},
  doi     = {10.1038/s41551-020-00682-w}
}

@article{zhang2025trident,
  title   = {Accelerating Data Processing and Benchmarking of {AI} Models for Pathology},
  author  = {Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and
             Ding, Tong and Mahmood, Faisal},
  journal = {arXiv preprint arXiv:2502.06750},
  year    = {2025},
  url     = {https://github.com/mahmoodlab/TRIDENT}
}

@inproceedings{hense2024xmil,
  title     = {{xMIL}: Insightful Explanations for Multiple Instance Learning in Histopathology},
  author    = {Hense, Julius and Jamshidi Idaji, Mina and Eberle, Oliver and
               Schnake, Thomas and Dippel, Jonas and Ciernik, Laure and
               Buchstab, Oliver and Mock, Andreas and Klauschen, Frederick and
               M\"{u}ller, Klaus-Robert},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  pages     = {8300--8328},
  year      = {2024}
}
```

## Acknowledgements

The MIL backbone and bundled Mamba CUDA kernels are adapted from
[MambaMIL](https://github.com/isyangshu/MambaMIL) (MICCAI 2024). Patching and
feature-extraction tooling follows [CLAM](https://github.com/mahmoodlab/CLAM),
with [TRIDENT](https://github.com/mahmoodlab/TRIDENT) as its successor for
high-throughput WSI processing. The faithfulness protocol follows
[xMIL](https://github.com/bifold-pathomics/xMIL) (NeurIPS 2024). Survival
labels were cross-checked against
[PORPOISE](https://github.com/mahmoodlab/PORPOISE),
[PathomicFusion](https://github.com/mahmoodlab/PathomicFusion) and
[LinkedOmics](https://linkedomics.org/).

We thank the authors of each pathology foundation model listed in section 2 for
releasing their weights.

## License

This repository builds on, and incorporates code from, several upstream
projects. **Refer to the original repositories for their licence terms:**

- [CLAM](https://github.com/mahmoodlab/CLAM)
- [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
- [MambaMIL](https://github.com/isyangshu/MambaMIL)
- [xMIL](https://github.com/bifold-pathomics/xMIL)
- the pathology foundation models listed in section 2

**Research use only.** This code is provided for academic research and is not a
medical device. Nothing here is validated or approved for clinical use,
diagnosis or treatment decisions.

TCGA data use is governed by the
[NIH Genomic Data Sharing Policy](https://gdc.cancer.gov/about-data/publications/publicationguidelines).
