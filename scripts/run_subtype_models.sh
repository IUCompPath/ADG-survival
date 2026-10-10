#!/usr/bin/env bash
# GBM-only then IDH-mutant survival models, sequentially on one GPU.
# --preloading yes keeps every feature bag in RAM (shared across folds).
set -u
R=/mnt/e/ssl/adg-survival
F=/mnt/e/ssl/WHOclassifier/pt_features/20x/pt_files

run () {  # name split_dir
  echo "=== $1 : $(date) ==="
  python "$R/pipeline/main_survival.py" \
    --task tcga_gbmlgg_survival \
    --csv_path "$R/dataset_csv/gbmlgg_processed_fixed.csv" \
    --data_root_dir "$F" \
    --split_dir "$R/splits/$2" \
    --results_dir "$R/results" \
    --exp_code "$1" \
    --model_type att_mil --in_dim 1536 --k 5 --seed 1 \
    --bag_loss nll_surv --preloading yes --early_stopping --max_epochs 200
  echo "=== $1 done : $(date) ==="
}

run gbm_only_att_mil_5fold     gbm_only_5fold
run idhmut_att_mil_5fold       idhmut_5fold
