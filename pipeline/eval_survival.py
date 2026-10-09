from __future__ import print_function

import numpy as np
import argparse
import torch
import os
import pandas as pd

# Internal imports
from utils.utils import *
from utils.eval_utils import eval_survival # <-- Corrected import based on reference
from dataset.dataset_survival import Generic_MIL_Survival_Dataset


# --- Evaluation settings ---
parser = argparse.ArgumentParser(description='CLAM Survival Evaluation Script')
#... (all parser arguments remain the same as the previous version) ...
parser.add_argument('--data_root_dir', type=str, default='features/pt_files',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default='tcga_gbmlgg_survival_eval_results',
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default='splits/tcga_gbmlgg_survival_kfold',
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, default='mamba_mil',
                    help='type of model (default: mambamil)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['tcga_gbmlgg_survival', 'tcga_luad_survival', 'other_survival_task'],
                    help='name of survival task')
parser.add_argument('--drop_out', type=float, default=0.0, help='dropout')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--seed', type=int, default=1, help='must match the training seed')

# MambaMIL specific arguments
parser.add_argument('--mambamil_rate',type=int, default=5, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')
parser.add_argument('--backbone', type=str, default='tcga_20x')
parser.add_argument('--patch_size', type=str, default=256)
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup Directories ---
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
args.embed_dim=args.in_dim
os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)
args.drop_out = 0.0
# --- Log Settings ---
settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(os.path.join(args.save_dir, 'eval_experiment_{}.txt'.format(args.save_exp_code)), 'w') as f:
    print(settings, file=f)

print(settings)

# --- Load Survival Dataset ---
if 'survival' in args.task.lower():
    args.n_classes = 4 # Number of survival bins used during training
    study = '_'.join(args.task.split('_')[:2]) # e.g., "tcga_gbmlgg"
    dataset = Generic_MIL_Survival_Dataset(
        mode = args.mode,
        apply_sig = args.apply_sig,
        csv_path = f'dataset_csv/{study}_processed.csv',
        data_dir = args.data_root_dir,
        shuffle = False,
        seed = args.seed,
        print_info = True,
        patient_strat = False,
        n_bins = 4,
        label_col = 'Survival_months',
        ignore = []
    )
else:
    raise NotImplementedError(f"Task {args.task} is not a survival task.")

# --- Fold Logic ---
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

# --- Main Evaluation Loop ---
if __name__ == "__main__":
    all_cindex = []

    for ckpt_idx, fold_num in enumerate(folds):
        if not os.path.exists(ckpt_paths[ckpt_idx]):
            print(f"Checkpoint path {ckpt_paths[ckpt_idx]} does not exist, skipping fold {fold_num}")
            continue

        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, fold_num)
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path,backbone=args.backbone)
            split_dataset = datasets[datasets_id[args.split]]
        
        # This is where the core logic is updated to align with your reference script
        # Note: We assume eval_model can handle a single dataset for evaluation
        # and returns a results dictionary and the c-index.
        patient_results_dict, c_index = eval_survival(split_dataset, args, ckpt_paths[ckpt_idx])
        all_cindex.append(c_index)

        # Save patient-level results for the fold
        patient_results_df = pd.DataFrame(patient_results_dict)
        patient_results_df.to_csv(os.path.join(args.save_dir, 'patient_results_fold_{}.csv'.format(fold_num)), index=False)


    # --- Summarize and Save Results ---
    final_df = pd.DataFrame({'folds': folds, 'test_c_index': all_cindex})

    if len(folds) > 1:
        mean_cindex = final_df['test_c_index'].mean()
        std_cindex = final_df['test_c_index'].std()

        summary_stats = pd.DataFrame({
            'folds': ['mean', 'std'],
            'test_c_index': [mean_cindex, std_cindex]
        })
        final_df = pd.concat([final_df, summary_stats], ignore_index=True)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'

    final_df.to_csv(os.path.join(args.save_dir, save_name), index=False)
    print("\nEvaluation Finished!")
    print(f"Results saved to {os.path.join(args.save_dir, save_name)}")