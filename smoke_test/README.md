# Smoke test

A 24-patient subset of the study cohort, bundled so the pipeline can be run end
to end without downloading TCGA or extracting features. Use it to confirm an
installation works before committing to a full run.

```bash
bash smoke_test/run_smoke_test.sh            # ~5 min on a GPU
```

Ends with `SMOKE TEST PASSED`. Results are written to `smoke_test/results/`
(git-ignored), or to a directory given as the first argument.

---

## What is bundled

| Path | Contents |
|------|----------|
| `features/pt_files/` | 24 GigaPath 20x feature files, one slide per patient |
| `dataset_csv/smoke_processed.csv` | survival time and censorship |
| `dataset_csv/smoke_histology_labels.csv` | WHO 2021 subtype |
| `splits/smoke_3fold/` | 3 patient-level folds |

24 patients, balanced 8 per WHO 2021 subtype and 12 events / 12 censored, so
every fold has events to fit on. Patients were chosen from those with a single
slide and the smallest feature files, then subsampled to at most 400 patches per
slide to keep the bundle under 55 MB. The features are otherwise unmodified
GigaPath embeddings.

Patching and feature extraction are **not** exercised - those run with
CLAM/TRIDENT on whole-slide images, which are too large to bundle. See section 2
of the main README.

---

## The four stages

Run the script, or step through it manually as below. `$S` is `smoke_test/`.

### 1. Train (3 folds)

```bash
python pipeline/main_survival.py \
    --task smoke_survival \
    --csv_path      $S/dataset_csv/smoke_processed.csv \
    --data_root_dir $S/features/pt_files \
    --split_dir     $S/splits/smoke_3fold \
    --results_dir   $S/results \
    --exp_code smoke --model_type att_mil --in_dim 1536 \
    --k 3 --seed 1 --bag_loss nll_surv \
    --preloading yes --early_stopping --max_epochs 20
```

Writes `s_0..s_2_checkpoint.pt` and `split_*_results.pkl` to
`$S/results/smoke_s1/`. Each fold prints a validation c-index.

### 2. Aggregate out-of-fold predictions

Concatenates the three `split_*_results.pkl` into one `combined_results.csv`
(24 rows, one per patient). Every later stage reads this file.

### 3. UMAP

```bash
python analysis/umap_embeddings.py \
    --features  $S/features/pt_files \
    --results   $S/results/smoke_s1/combined_results.csv \
    --histology $S/dataset_csv/smoke_histology_labels.csv \
    --in-dim 1536 --pooling mean --out $S/results/umap
```

Writes `umap_mean.png`, `.tiff` and a coordinates CSV.

### 4. Attention faithfulness

```bash
python analysis/patch_flipping.py \
    --features   $S/features/pt_files \
    --ckpt-dir   $S/results/smoke_s1 \
    --survival   $S/results/smoke_s1/combined_results.csv \
    --in-dim 1536 --n-folds 3 --model-type att_mil \
    --percents 0 10 25 50 75 100 --n-random 2 \
    --out $S/results/faithfulness
```

Writes `patch_flipping_cindex.csv` and a curve.

---

## What passing means

Every stage ran, the shapes line up, and patch flipping reproduces the model's
own out-of-fold c-index at 0% removal - the one number in this run that must
match exactly.

**It says nothing about model quality.** A c-index on 24 patients is noise, and
the faithfulness curve here is not interpretable: with 8 patients per test fold,
a single pair changing order moves the c-index by several points, so the curve
can easily run the wrong way. Use the full cohort for any claim.

---

## Provenance

Derived from TCGA-GBM/LGG diagnostic slides (open access, subject to the
[NIH Genomic Data Sharing Policy](https://gdc.cancer.gov/about-data/publications/publicationguidelines))
embedded with [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)
(Apache-2.0, research use only). Patient barcodes are retained, so these rows
trace back to the public TCGA records.
