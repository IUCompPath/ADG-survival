import torch
import os
import pandas as pd
from tqdm import tqdm
from models.MambaMIL import MambaMIL
import numpy as np
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = MambaMIL(
    in_dim=1024,
    n_classes=4,
    dropout=0.0,
    act='gelu',
    survival=True,
    layer=2,
    rate=5,
    type='SRMamba'
).to(device)

# Folder containing feature .pt files
feature_dir = os.environ.get('ADG_FEATURES', 'features/uni_v2/pt_files') + os.sep
ckpt_base = os.environ.get('ADG_CKPT', 'results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1') + os.sep

# Get all pt files
pt_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]

# Results list
results = []

# Iterate through each .pt file
for pt_file in tqdm(pt_files, desc="Processing slides"):
    slide_path = os.path.join(feature_dir, pt_file)
    slide_name = pt_file.replace('.pt', '')

    try:
        features = torch.load(slide_path, map_location=device)
        if isinstance(features, dict):
            features = features['features']
        features = features.float().unsqueeze(0).to(device)

        risks = []

        # Run through 10 checkpoints
        for i in range(10):
            ckpt_path = os.path.join(ckpt_base, f's_{i}_checkpoint.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)

            # Check if checkpoint is full dict or just model_state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            with torch.no_grad():
                output = model(features)
                _, survival, _, _, _ = output
                risk = -torch.sum(survival, dim=1).item()
                risks.append(risk)

        result = {'slide_name': slide_name}
        for i, risk in enumerate(risks):
            result[f'risk_s_{i}'] = risk
        result['mean_risk'] = sum(risks) / len(risks)
        results.append(result)

    except Exception as e:
        print(f"Error processing {slide_name}: {e}")
        continue

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(ckpt_base, 'risk_scores_all_folds.csv'), index=False)
print("Saved results to mamba_risk_scores.csv")

df_risk_scores = pd.read_csv(os.path.join(ckpt_base, 'risk_scores_all_folds.csv'))
df_combined = pd.read_csv(os.path.join(ckpt_base, 'combined_results_histology_with_risk_groups.csv'))

df_risk_scores['slide_id'] = df_risk_scores['slide_name']
df_combined['slide_id'] = df_combined['slide_id_y']  # or 'slide_id_y' if needed

df_merged = df_combined.merge(df_risk_scores, on='slide_id', how='left')

risk_cols = [col for col in df_risk_scores.columns if col.startswith('risk_s_')]
def find_closest_col(row):
    if pd.isna(row['risk']):
        return None
    diffs = [abs(row['risk'] - row[col]) for col in risk_cols]
    min_idx = int(np.argmin(diffs))
    return risk_cols[min_idx]

df_merged['closest_risk_model'] = df_merged.apply(find_closest_col, axis=1)

df_merged.to_csv(os.path.join(ckpt_base, 'risk_scores_1699.csv'), index=False)

combined_df = df_merged
