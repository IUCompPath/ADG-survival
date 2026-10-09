import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.MambaMIL import MambaMIL
from lifelines.utils import concordance_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path):
    model = MambaMIL(
        in_dim=1024, n_classes=4, dropout=0.0,
        act='gelu', survival=True, layer=2,
        rate=5, type='SRMamba'
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    return model

def run_model(model, features):
    with torch.no_grad():
        hazards, S, Y_hat, A_raw, _ = model(features)
        risk = -torch.sum(S, dim=1).item()
        attn = A_raw.squeeze().detach().cpu().numpy()  # shape: (n_patches,)
    return risk, attn

# Load survival info (slide_name, time, event)
survival_df = pd.read_csv("survival_data.csv")  # Must contain slide_name, time, event
slide_dir = os.environ.get('ADG_FEATURES', 'features/uni_v2/pt_files') + os.sep
ckpt_base = os.environ.get('ADG_CKPT', 'results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1') + os.sep

removal_steps = list(range(0, 21, 1))  # 0% to 20% top-attn removal
results = {r: [] for r in removal_steps}  # r: [(risk, time, event)]

for pt_file in tqdm(os.listdir(slide_dir)):
    if not pt_file.endswith('.pt'):
        continue
    slide_name = pt_file.replace('.pt', '')
    slide_path = os.path.join(slide_dir, pt_file)
    slide_data = survival_df[survival_df['slide_name'] == slide_name]
    if slide_data.empty:
        continue

    time = slide_data['time'].values[0]
    event = slide_data['event'].values[0]

    features = torch.load(slide_path, map_location=device)
    if isinstance(features, dict):
        features = features['features']
    features = features.float().unsqueeze(0).to(device)  # [1, n_patches, 1024]
    n_patches = features.shape[1]

    # Get base attention from model 0 (for patch ranking)
    base_model = load_model(os.path.join(ckpt_base, 's_0_checkpoint.pt'))
    _, base_attn = run_model(base_model, features)
    attn_rank = np.argsort(-base_attn)  # descending sort

    for percent in removal_steps:
        n_remove = int((percent / 100) * n_patches)
        keep_idx = np.sort(attn_rank[n_remove:])  # keep remaining

        f_subset = features[:, keep_idx, :]  # [1, n_kept, 1024]

        risks = []
        for fold in range(10):
            model = load_model(os.path.join(ckpt_base, f's_{fold}_checkpoint.pt'))
            try:
                risk, _ = run_model(model, f_subset)
                risks.append(risk)
            except Exception as e:
                print(f"Error with {slide_name}, fold {fold}, %={percent}: {e}")
                continue

        mean_risk = np.mean(risks)
        results[percent].append((mean_risk, time, event))

# Compute C-index for each drop %
cindex_results = {}
for percent, values in results.items():
    if not values:
        continue
    risks, times, events = zip(*values)
    cindex = concordance_index(times, -np.array(risks), events)
    cindex_results[percent] = cindex
    print(f"Top-{percent}% removed -> C-index: {cindex:.4f}")

# Save to CSV
pd.DataFrame.from_dict(cindex_results, orient='index', columns=['c_index']).to_csv("cindex_robustness_curve.csv")

import matplotlib.pyplot as plt
df = pd.read_csv("cindex_robustness_curve.csv")
plt.plot(df['percent_removed'], df['c_index'], marker='o')
plt.xlabel("Top-% Attention Patches Removed")
plt.ylabel("C-Index")
plt.title("Attention Robustness Evaluation")
plt.grid(True)
plt.show()






import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.MambaMIL import MambaMIL
from lifelines.utils import concordance_index

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
feature_dir = os.environ.get('ADG_FEATURES', 'features/uni_v2/pt_files') + os.sep
ckpt_base = os.environ.get('ADG_CKPT', 'results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1') + os.sep
survival_csv = os.path.join(ckpt_base, 'combined_results.csv')  
# must have columns: slide_name, time, event

# Removal steps (0% to 20%)
removal_steps = list(range(0, 100))  # Removes 0% to 99% in 1% steps
results_top = {r: [] for r in removal_steps}
results_bottom = {r: [] for r in removal_steps}

# Load survival metadata
survival_df = pd.read_csv(survival_csv)

# Helper: load model
def load_model(ckpt_path):
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
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    return model

# Helper: run model and get risk + attention
def run_model(model, features):
    with torch.no_grad():
        hazards, S, Y_hat, A_raw, _ = model(features)
        risk = -torch.sum(S, dim=1).item()
        attn = A_raw.squeeze().detach().cpu().numpy()  # shape: (n_patches,)
    return risk, attn

# Iterate over slides
for pt_file in tqdm(os.listdir(feature_dir), desc="Evaluating slides"):
    if not pt_file.endswith('.pt'):
        continue
    slide_name = pt_file.replace('.pt', '')
    slide_path = os.path.join(feature_dir, pt_file)

    # Get survival info
    row = survival_df[survival_df['slide_name'] == slide_name]
    if row.empty:
        print(f"Missing survival data for {slide_name}")
        continue

    time = row['time'].values[0]
    event = row['event'].values[0]

    # Load features
    features = torch.load(slide_path, map_location=device)
    if isinstance(features, dict):
        features = features['features']
    features = features.float().unsqueeze(0).to(device)  # [1, n_patches, 1024]
    n_patches = features.shape[1]

    # Get attention ranking from model s_0
    try:
        model_base = load_model(os.path.join(ckpt_base, 's_5_checkpoint.pt'))
        _, base_attn = run_model(model_base, features)
    except Exception as e:
        print(f"Error getting base attention for {slide_name}: {e}")
        continue

    attn_rank = np.argsort(-base_attn)  # high to low

    for percent in tqdm(removal_steps):
        n_remove = int((percent / 100) * n_patches)
        for mode in ['top', 'bottom']:
            if n_remove >= n_patches:
                continue  # skip if all patches would be removed

            if mode == 'top':
                keep_idx = np.sort(attn_rank[n_remove:])  # remove top
            else:  # bottom
                keep_idx = np.sort(attn_rank[:n_patches - n_remove])  # remove bottom

            if len(keep_idx) == 0:
                continue

            f_subset = features[:, keep_idx, :]  # [1, remaining, 1024]
            risks = []

            
            model = load_model(os.path.join(ckpt_base, f's_5_checkpoint.pt'))
            risk, _ = run_model(model, f_subset)
            entry = (risk, time, event)
            if mode == 'top':
                results_top[percent].append(entry)
            else:
                results_bottom[percent].append(entry)

# Compute C-index
def compute_cindex(results_dict):
    cindex_dict = {}
    for percent, values in results_dict.items():
        if not values:
            continue
        risks, times, events = zip(*values)
        cidx = concordance_index(times, -np.array(risks), events)
        cindex_dict[percent] = cidx
    return cindex_dict

cindex_top = compute_cindex(results_top)
cindex_bottom = compute_cindex(results_bottom)

# Merge into DataFrame
df_top = pd.DataFrame.from_dict(cindex_top, orient='index', columns=['c_index_top']).reset_index().rename(columns={'index': 'percent_removed'})
df_bottom = pd.DataFrame.from_dict(cindex_bottom, orient='index', columns=['c_index_bottom']).reset_index().rename(columns={'index': 'percent_removed'})
df = pd.merge(df_top, df_bottom, on='percent_removed', how='outer').sort_values('percent_removed')
df.to_csv("cindex_attention_robustness_comparison.csv", index=False)
print("Saved: cindex_attention_robustness_comparison.csv")
df = pd.read_csv("cindex_attention_robustness_comparison.csv")
# Plot
plt.plot(df['percent_removed'], df['c_index_top'], label='Top-% Attention Removed', marker='o')
plt.plot(df['percent_removed'], df['c_index_bottom'], label='Bottom-% Attention Removed', marker='x')
plt.xlabel("Percent of Patches Removed")
plt.ylabel("C-Index")
plt.title("Robustness of Attention-Based Patch Removal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("attention_robustness_curve.png")
plt.show()

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Normalize both columns
scaler_top = MinMaxScaler()
scaler_bottom = MinMaxScaler()

c_index_top_norm = scaler_top.fit_transform(df[['c_index_top']]).flatten()
c_index_bottom_norm = scaler_bottom.fit_transform(df[['c_index_bottom']]).flatten()

plt.figure(figsize=(8,5))

# Plot c_index_top reversed values but x-axis 0-100
plt.plot(df['percent_removed'], c_index_top_norm[::-1], marker='o', label='top')

# Plot c_index_bottom normally
plt.plot(df['percent_removed'], c_index_bottom_norm, marker='x', label='bottom')

plt.xlabel("percentile of Patches Removed")
plt.ylabel("Normalized performance")
plt.legend()
plt.grid(True)
plt.savefig("both_curves.png")
plt.show()
plt.close()


row_100 = {
    'percent_removed': 100,
    'c_index_top': 0,
    'c_index_bottom': 0,
    'percent_retained': 100,
    'normalized_c_index_top': 0  # normalized to the value at 0% removed
}

df_sorted = pd.concat([df_sorted, pd.DataFrame([row_100])], ignore_index=True)

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe.
df = pd.read_csv('cindex_attention_robustness_comparison.csv')
df_sorted = df.sort_values(by='percent_removed')
df_sorted['percent_retained'] = 99 - df_sorted['percent_removed']
value_at_99_removed = df_sorted[df_sorted['percent_removed'] == 100]['c_index_top'].iloc[0]
df_sorted['normalized_c_index_top'] = df_sorted['c_index_top'] / value_at_99_removed
df_plot = df_sorted[(df_sorted['percent_removed'] >= 0) & (df_sorted['percent_removed'] <= 100)]
plt.figure(figsize=(6, 6))
plt.plot(df_plot['percent_retained'], df_plot['normalized_c_index_top'], marker='o', linestyle='-')
plt.xlim(0, 100)
plt.xlabel('Percent Dropped', fontsize=18)
plt.ylabel('Performance', fontsize=18)
plt.grid(True)
# plt.savefig('normalized_cindex_top_plot_retained.png', dpi=300)
plt.show()
plt.close()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cindex_attention_robustness_comparison.csv')
df_sorted = df.sort_values(by='percent_removed')
df_sorted['percent_retained'] = 100 - df_sorted['percent_removed']
df_sorted.loc[df_sorted['percent_removed'] == 100, 'c_index_top'] = 0
value_at_0_removed = df_sorted[df_sorted['percent_removed'] == 0]['c_index_top'].iloc[0]
df_sorted['normalized_c_index_top'] = df_sorted['c_index_top'] / value_at_0_removed
df_plot = df_sorted[(df_sorted['percent_removed'] >= 0) & (df_sorted['percent_removed'] <= 100)]
plt.figure(figsize=(6, 6))
plt.plot(df_plot['percent_retained'], df_plot['normalized_c_index_top'], marker='o', linestyle='-')
plt.xlim(0, 100)
plt.xlabel('Percent Retained', fontsize=18)
plt.ylabel('Normalized C-index (Top)', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()