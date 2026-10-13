#!/usr/bin/env bash
# End-to-end check of the pipeline on the bundled 24-patient subset.
# Trains 3 folds, aggregates predictions, then runs the UMAP and
# patch-flipping analyses. Expect a few minutes on a GPU.
#
#   bash smoke_test/run_smoke_test.sh [OUT_DIR]
#
# The c-index on 24 patients is not meaningful; this only checks that every
# stage runs and that patch flipping reproduces the model's own predictions.
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S="$R/smoke_test"
OUT="${1:-$S/results}"
mkdir -p "$OUT"

echo "== 1/4  training (3 folds) =="
python "$R/pipeline/main_survival.py" \
    --task smoke_survival \
    --csv_path       "$S/dataset_csv/smoke_processed.csv" \
    --data_root_dir  "$S/features/pt_files" \
    --split_dir      "$S/splits/smoke_3fold" \
    --results_dir    "$OUT" \
    --exp_code smoke --model_type att_mil --in_dim 1536 \
    --k 3 --seed 1 --bag_loss nll_surv \
    --preloading yes --early_stopping --max_epochs 20

EXP="$OUT/smoke_s1"

echo "== 2/4  aggregating out-of-fold predictions =="
python - "$EXP" <<'PY'
import sys, glob, os
import pandas as pd, numpy as np
d = sys.argv[1]
fs = sorted(glob.glob(d + '/split_*_results.pkl'),
            key=lambda f: int(os.path.basename(f).split('_')[1]))
c = pd.concat([pd.DataFrame(pd.read_pickle(f)).T for f in fs])
c['risk'] = c['risk'].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
c.reset_index(drop=True).to_csv(d + '/combined_results.csv', index=False)
print(f'  {len(c)} patients -> combined_results.csv')
PY

echo "== 3/4  UMAP =="
python "$R/analysis/umap_embeddings.py" \
    --features  "$S/features/pt_files" \
    --results   "$EXP/combined_results.csv" \
    --histology "$S/dataset_csv/smoke_histology_labels.csv" \
    --in-dim 1536 --pooling mean --out "$OUT/umap"

echo "== 4/4  attention faithfulness =="
python "$R/analysis/patch_flipping.py" \
    --features   "$S/features/pt_files" \
    --ckpt-dir   "$EXP" \
    --survival   "$EXP/combined_results.csv" \
    --in-dim 1536 --n-folds 3 --model-type att_mil \
    --percents 0 10 25 50 75 100 --n-random 2 \
    --out "$OUT/faithfulness"

echo
echo "SMOKE TEST PASSED -> $OUT"
