import os
import glob
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

# --- Configuration ---
base_path = os.environ.get('ADG_RESULTS', 'results/tcga_gbmlgg_survival') + os.sep
time_col = "survival"
event_col = "censorship" # Assuming 1 for censored, 0 for event
risk_col = "risk"
# --- End Configuration ---

# Initialize a dictionary to store the combined C-index for the final matrix
combined_cindex_matrix_data = {}

# Get the list of outer folders (assuming these are the 8 folders)
outer_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
print(f"Found {len(outer_folders)} outer folders.")

# Loop through each outer folder
for outer_folder in outer_folders:
    outer_folder_path = os.path.join(base_path, outer_folder)
    print(f"\nProcessing outer folder: {outer_folder}")

    # Initialize dictionary for the current outer folder's row in the matrix
    combined_cindex_matrix_data[outer_folder] = {}

    # Get the list of inner folders (assuming these are the 9 folders inside each outer)
    #inner_folders = [f for f in os.listdir(outer_folder_path) if os.path.isdir(os.path.join(outer_folder_path, f))]
    inner_folders = ['chief_s1', 'conch_v15_s1', 'ctranspath_s1', 'gigapath_s1', 'hibou_s1', 'hoptimus1_s1', 'imagenet_s1', 'lunit_s1', 'prism_s1', 'titan_s1','uni_v2_s1']
    print(f"Found {len(inner_folders)} inner folders inside {outer_folder}.")

    # Loop through each inner folder
    for inner_folder in inner_folders:
        inner_folder_path = os.path.join(outer_folder_path, inner_folder)
        print(f"  Processing inner folder: {inner_folder}")

        # Find all split result pickle files in the inner folder
        # Note: Adjusted glob pattern based on user's initial snippet format
        split_files = glob.glob(os.path.join(inner_folder_path, "split_*_results.pkl"))

        if not split_files:
            print(f"    No split files found in {inner_folder_path}. Skipping.")
            combined_cindex_matrix_data[outer_folder][inner_folder] = np.nan # Use NaN for missing data
            continue

        # --- Process split files for the current inner folder ---
        all_split_data = []
        fold_c_indices = []

        # Load and combine data from all splits for the current inner folder
        for file in split_files:
            
            try:
                # Assuming the pkl contains a dictionary where keys are sample IDs and values are dictionaries of results
                split_df = pd.DataFrame(pd.read_pickle(file)).T
                all_split_data.append(split_df)

                # Calculate fold-wise C-index for this split
                if risk_col in split_df.columns and time_col in split_df.columns and event_col in split_df.columns:
                     # Handle potential list/array risk scores in split data as well
                    if split_df[risk_col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                         split_df_processed_risk = split_df.copy() # Avoid modifying original split_df if it's needed later
                         split_df_processed_risk[risk_col] = split_df_processed_risk[risk_col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
                    else:
                         split_df_processed_risk = split_df

                    # Ensure data types are float for calculations
                    split_risk_scores = split_df_processed_risk[risk_col].astype(float).values
                    split_survival_times = split_df_processed_risk[time_col].astype(float).values
                    # Assuming event_col is 1 for censored, 0 for event. Lifelines needs 1 for event.
                    split_event_observed = 1 - split_df_processed_risk[event_col].astype(float).values

                    # Filter out rows with NaN/inf in relevant columns before calculating C-index
                    valid_indices = np.isfinite(split_risk_scores) & np.isfinite(split_survival_times) & np.isfinite(split_event_observed)
                    if np.sum(valid_indices) > 1: # Need at least 2 samples to calculate C-index
                        c_index_fold = concordance_index(split_survival_times[valid_indices], -split_risk_scores[valid_indices], split_event_observed[valid_indices])
                        fold_c_indices.append(c_index_fold)
                    else:
                        #print(f"      Not enough valid data points in split file {os.path.basename(file)} to calculate C-index. Skipping.")
                        fold_c_indices.append(np.nan) # Add NaN if C-index cannot be calculated
                else:
                    #print(f"      Required columns ({risk_col}, {time_col}, {event_col}) not found in split file {os.path.basename(file)}. Skipping fold C-index.")
                    fold_c_indices.append(np.nan) # Add NaN if columns are missing

            except Exception as e:
                #print(f"      Error loading or processing split file {os.path.basename(file)}: {e}")
                # Append None or NaN to indicate failure for this split
                fold_c_indices.append(np.nan)


        # Concatenate all split DataFrames into one for the current inner folder
        if all_split_data:
            combined_df = pd.concat(all_split_data, ignore_index=True)

            # --- Process combined data ---
            # Handle potential list/array risk scores in the combined data
            if risk_col in combined_df.columns and combined_df[risk_col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                 combined_df[risk_col] = combined_df[risk_col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)

            # Save the combined DataFrame to CSV
            csv_path_combined = os.path.join(inner_folder_path, "combined_results.csv")
            combined_df.to_csv(csv_path_combined, index=False)
            #print(f"    Saved combined results to {csv_path_combined}")

            # Calculate concordance index for the combined data
            if risk_col in combined_df.columns and time_col in combined_df.columns and event_col in combined_df.columns:
                 # Ensure data types are float for calculations
                 risk_scores_combined = combined_df[risk_col].astype(float).values
                 survival_times_combined = combined_df[time_col].astype(float).values
                 # Assuming event_col is 1 for censored, 0 for event. Lifelines needs 1 for event.
                 event_observed_combined = 1 - combined_df[event_col].astype(float).values

                 # Filter out rows with NaN/inf before calculating C-index
                 valid_indices_combined = np.isfinite(risk_scores_combined) & np.isfinite(survival_times_combined) & np.isfinite(event_observed_combined)

                 if np.sum(valid_indices_combined) > 1: # Need at least 2 samples
                     c_index_combined = concordance_index(survival_times_combined[valid_indices_combined], -risk_scores_combined[valid_indices_combined], event_observed_combined[valid_indices_combined])
                     #print(f"    Combined Concordance Index: {c_index_combined:.4f}")
                     # Store for the final matrix
                     combined_cindex_matrix_data[outer_folder][inner_folder] = c_index_combined
                 else:
                     #print(f"    Not enough valid data points in combined data for {inner_folder} to calculate C-index. Storing NaN.")
                     combined_cindex_matrix_data[outer_folder][inner_folder] = np.nan
            else:
                # print(f"    Required columns ({risk_col}, {time_col}, {event_col}) not found in combined data for {inner_folder}. Storing NaN.")
                 combined_cindex_matrix_data[outer_folder][inner_folder] = np.nan

            # --- Save fold-wise C-indices ---
            if fold_c_indices:
                fold_results_df = pd.DataFrame({
                    'split_file': [os.path.basename(f) for f in split_files],
                    'c_index': fold_c_indices
                })
                # Calculate mean and std, ignoring NaNs
                mean_c_index = np.nanmean(fold_c_indices)
                std_c_index = np.nanstd(fold_c_indices)

                # Add mean and std rows to the fold results DataFrame for easy viewing
                # Using reset_index() to add a new column for metric type
                mean_std_df = pd.DataFrame({
                    'split_file': ['Mean', 'Std Dev'],
                    'c_index': [mean_c_index, std_c_index]
                })
                fold_results_df = pd.concat([fold_results_df, mean_std_df], ignore_index=True)

                csv_path_folds = os.path.join(inner_folder_path, "fold_results.csv")
                fold_results_df.to_csv(csv_path_folds, index=False)
               # print(f"    Saved fold-wise results (including mean/std) to {csv_path_folds}")
                print(f"    Fold C-index (Mean +/- Std): {mean_c_index:.4f} +/- {std_c_index:.4f}")

            else:
                 print(f"    No fold C-indices were calculated for {inner_folder}.")

        else:
            print(f"  No valid split data found in {inner_folder_path}. Skipping combined processing.")
            combined_cindex_matrix_data[outer_folder][inner_folder] = np.nan # Use NaN for missing data


# --- Create and save the final 8x9 matrix ---
print("\nCreating overall concordance index matrix...")
# Convert the dictionary of dictionaries into a DataFrame
overall_cindex_matrix_df = pd.DataFrame(combined_cindex_matrix_data)

# Save the matrix to a CSV in the base path
matrix_csv_path = os.path.join(base_path, "overall_combined_cindex_matrix.csv")
overall_cindex_matrix_df.to_csv(matrix_csv_path)
print(f"Saved overall combined concordance index matrix to {matrix_csv_path}")

# Print the final matrix
print("\nOverall Combined Concordance Index Matrix:")
print(overall_cindex_matrix_df)


import os
import pandas as pd

base_path = os.environ.get('ADG_RESULTS', 'results/tcga_gbmlgg_survival') + os.sep 
parent_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
example_parent_path = os.path.join(base_path, parent_dirs[0])
sub_dirs = sorted([d for d in os.listdir(example_parent_path) if os.path.isdir(os.path.join(example_parent_path, d))])
summary_df = pd.DataFrame(index=sub_dirs, columns=parent_dirs)

for parent in parent_dirs:
    parent_path = os.path.join(base_path, parent)
    combined_df = pd.DataFrame()
    for subdir in sub_dirs:
        csv_path = os.path.join(parent_path, subdir, 'fold_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.set_index('split_file')
            mean = df.loc['Mean', 'c_index']
            std = df.loc['Std Dev', 'c_index']
            summary_df.loc[subdir, parent] = f"{mean:.4f}±{std:.4f}"
    combined_all = []
    for subdir in sub_dirs:
        csv_path = os.path.join(parent_path, subdir, 'fold_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.set_index('split_file')
            combined_all.append(df['c_index'].rename(subdir))
    if combined_all:
        combined_df = pd.concat(combined_all, axis=1)
        combined_df.to_csv(os.path.join(parent_path, 'combined_results.csv'))
summary_df.to_csv(os.path.join(base_path, 'summary_mean_std.csv'))
