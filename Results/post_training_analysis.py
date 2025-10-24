import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# --- GLOBAL CONFIGURATION ---

# Define the root directory of the project
current_dir = os.getcwd() 

# Define the directory where all specific run folders (train_fold_1, etc.) reside.
# NOTE: Manually check/adjust 'Results YOLOv12s' to 'Results YOLOv12l' if analyzing the Large model.
BASE_RESULTS_PATH = os.path.join(current_dir, 'Results', 'Results YOLOv12s', 'runs', 'detect') 

# List of folders corresponding to the 5 folds created during training.
MODEL_FOLD_NAMES = [f'train_fold_{i}' for i in range(1, 6)] 

# Output directory for plots (relative to project root)
PLOTS_ROOT = os.path.join(current_dir, 'Results')
PLOTS_SUBFOLDER = os.path.join(PLOTS_ROOT, 'Plots')
os.makedirs(PLOTS_SUBFOLDER, exist_ok=True) 

# Define plotting aesthetics
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 10

# --- HELPER FUNCTIONS ---

def calculate_f1(precision: float, recall: float) -> float:
    """Calculates the F1-measure from precision and recall."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def get_best_metrics(file_path: str) -> Dict[str, float]:
    """Extracts the maximum values for key metrics from a single fold's results CSV."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data['metrics/precision(B)'] = pd.to_numeric(data['metrics/precision(B)'], errors='coerce')
    data['metrics/recall(B)'] = pd.to_numeric(data['metrics/recall(B)'], errors='coerce')
    data['metrics/mAP50(B)'] = pd.to_numeric(data['metrics/mAP50(B)'], errors='coerce')
    best_precision = data['metrics/precision(B)'].max()
    best_recall = data['metrics/recall(B)'].max()
    best_mAP50 = data['metrics/mAP50(B)'].max()
    best_f1 = calculate_f1(best_precision, best_recall)
    return {
        'precision': best_precision,
        'recall': best_recall,
        'F1': best_f1,
        'mAP50': best_mAP50
    }

def summarize_metrics(csv_files: List[str]):
    """Calculates and prints the absolute best and average best metrics across all folds."""
    all_best_metrics = []
    for file_path in csv_files:
        try:
            best_metrics = get_best_metrics(file_path)
            all_best_metrics.append(best_metrics)
        except Exception as e:
            print(f"Error processing metrics file {file_path}: {e}")
            continue

    if not all_best_metrics:
        print("No valid metrics found for averaging.")
        return

    mean_metrics = {
        key: sum(d[key] for d in all_best_metrics) / len(all_best_metrics)
        for key in all_best_metrics[0].keys()
    }

    print("\n--- Mean Best Performance Across All Folds ---")
    for i, metrics in enumerate(all_best_metrics, 1):
        print(f"Fold {i} - Precision: {metrics['precision']:.5f}, Recall: {metrics['recall']:.5f}, F1: {metrics['F1']:.5f}, mAP50: {metrics['mAP50']:.5f}")
    print("\n[AVERAGE] Best Metrics:")
    for key, value in mean_metrics.items():
        print(f"{key.capitalize()}: {value:.5f}")

# --- 2. CONFUSION MATRIX PLOTTING ---

HARDCODED_CM = np.array([
    [8630, 7645, 3809, 0],
    [9227, 7212, 4788, 0],
    [10666, 7281, 6010, 0],
    [9951, 7058, 4653, 0],
    [8141, 10149, 3796, 0],
])

def save_confusion_matrix(matrix: np.ndarray, title: str, filename: str, is_average: bool = False, classes: List[str] = None):
    """Saves the confusion matrix (sum or average) as a PNG file."""
    sns.set_style("whitegrid")
    
    # Reshape [TP, FP, FN, TN] into a 2x2 matrix: [[TP, FP], [FN, TN]]
    confusion_matrix_reshaped = np.array([[matrix[0], matrix[1]], [matrix[2], matrix[3]]])

    plt.figure(figsize=(8, 6))
    
    # Use standard labels for the biclass output (Gun, Human)
    labels = classes if classes else ['Gun', 'Human'] 
    
    sns.heatmap(confusion_matrix_reshaped, 
                annot=True, 
                fmt='.2f' if is_average else 'd', 
                cmap='Blues', 
                xticklabels=labels, 
                yticklabels=labels)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title(title)

    plt.savefig(os.path.join(PLOTS_SUBFOLDER, filename))
    plt.close()

def process_confusion_matrices(cm_array: np.ndarray):
    """Calculates and plots the sum and average of confusion matrices."""
    
    sum_matrix = np.sum(cm_array, axis=0)
    average_matrix = np.mean(cm_array, axis=0)
    
    # Save the sum confusion matrix
    save_confusion_matrix(sum_matrix, 'Sum of Confusion Matrices (5 Folds)', 'sum_confusion_matrix.png', classes=['Gun', 'Human'])
    
    # Save the average confusion matrix
    save_confusion_matrix(average_matrix, 'Average Confusion Matrix (5 Folds)', 'average_confusion_matrix.png', is_average=True, classes=['Gun', 'Human'])
    
    print("\n--- Confusion Matrix Analysis ---")
    print("Sum Matrix [TP, FP, FN, TN]:", sum_matrix)
    print("Average Matrix [TP, FP, FN, TN]:", average_matrix)
    print("Plots saved to Results/Plots/")

# --- 3. LOSS AND METRIC PLOTTING ---

def plot_all_results(subfolder: str, csv_files: List[str]):
    """Loads all fold results for a model run, calculates the mean, and generates all plots."""
    
    if not csv_files:
        print(f"Skipping plotting: No CSV files provided.")
        return

    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df = df.apply(pd.to_numeric, errors='coerce')
            dfs.append(df)
        else:
            print(f"File {file} does not exist. Skipping.")

    if not dfs:
        print(f"No valid data found. Skipping plots.")
        return

    # Concatenate the DataFrames and calculate the mean of numeric columns only
    mean_df = pd.concat(dfs).groupby(level=0).mean()

    # Create a specific output folder for this subfolder's plots
    output_folder = os.path.join(PLOTS_SUBFOLDER, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    loss_pairs = [('train/box_loss', 'val/box_loss'), 
                  ('train/cls_loss', 'val/cls_loss'), 
                  ('train/dfl_loss', 'val/dfl_loss')]

    metrics_fields = [
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'Mean Average Precision at IoU 50'),
        ('metrics/mAP50-95(B)', 'Mean Average Precision at IoU 50-95')
    ]

    # 1. Plot comparison of train and val means for each loss type
    for train_field, val_field in loss_pairs:
        plt.figure(figsize=(10, 6))
        
        plt.plot(mean_df[train_field], label='Mean Train', color='blue', linewidth=2)
        plt.plot(mean_df[val_field], label='Mean Validation', color='red', linestyle='--', linewidth=2)
        
        combined_title = f'{subfolder}: {train_field.split("/")[1].title()} Loss Comparison (Mean)'
        plt.title(combined_title, fontsize=TITLE_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.xlabel('Epochs', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel('Value', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
        plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
        
        filename = os.path.join(output_folder, f'combined_{train_field.split("/")[1].lower()}_mean_only.png')
        plt.savefig(filename)
        plt.close()

    # 2. Plot single graphs for each metric with folds and mean
    for field, pretty_name in metrics_fields:
        plt.figure(figsize=(10, 6))
        for idx, df in enumerate(dfs):
            plt.plot(df[field], alpha=0.5, label=f'Fold {idx+1}')
        plt.plot(mean_df[field], label='Mean', color='black', linewidth=2)
        plt.title(f'{subfolder}: {pretty_name}', fontsize=TITLE_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.xlabel('Epochs', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel('Value', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
        plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
        filename = os.path.join(output_folder, f'{field.split("/")[1].lower()}_metric.png')
        plt.savefig(filename)
        plt.close()
    
    print(f"✅ Plots for {subfolder} saved to {output_folder}")


# --- 4. OPTIMAL EPOCH DETERMINATION ---

def plot_best_epoch(subfolder: str, csv_files: List[str]):
    """
    Calculates the optimal epoch for each fold based on a weighted combined score 
    and plots the metrics with the optimal epoch marked.
    """
    
    if not csv_files:
        return
        
    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            cols_to_numeric = ['metrics/mAP50(B)', 'val/box_loss', 'val/cls_loss',
                                 'metrics/precision(B)', 'metrics/recall(B)']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)
        
    if not dfs:
        return
    
    # --- FIX: Define output_folder inside the function ---
    output_folder = os.path.join(PLOTS_SUBFOLDER, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    # --- END FIX ---
        
    # Normalized coefficients for calculating the combined score (Custom weights)
    coeff_box_loss = 0.5
    coeff_cls_loss = 0.5
    coeff_precision = 0.375
    coeff_recall = 0.375
    coeff_map50 = 0.25

    # Calculate and plot the optimal epoch for each fold
    for idx, df in enumerate(dfs):
        
        # Calculate a custom combined score: (mAP + Precision + Recall) - (Losses)
        combined_score = (df['metrics/mAP50(B)'] * coeff_map50 +
                             df['metrics/precision(B)'] * coeff_precision +
                             df['metrics/recall(B)'] * coeff_recall -
                             (df['val/box_loss'] * coeff_box_loss) -
                             (df['val/cls_loss'] * coeff_cls_loss))

        # Find the epoch (index) with the highest score
        best_epoch = combined_score.idxmax()
        best_score = combined_score.max()

        print(f"Fold {idx + 1} ({subfolder}): Best epoch = {best_epoch + 1}, Combined Score = {best_score:.4f}")

        # Plotting for each fold
        plt.figure(figsize=(12, 6))

        # Plot the individual metrics
        plt.plot(df['metrics/mAP50(B)'], label='mAP50', color='blue', linestyle='--')
        plt.plot(df['val/box_loss'], label='Validation Box Loss', color='green', linestyle='-')
        plt.plot(df['val/cls_loss'], label='Validation Class Loss', color='orange', linestyle='-')
        plt.plot(df['metrics/precision(B)'], label='Precision', color='purple', linestyle='-.')
        plt.plot(df['metrics/recall(B)'], label='Recall', color='red', linestyle=':')

        # Add a dashed vertical line for the best epoch
        plt.axvline(x=best_epoch, color='black', linestyle=':', linewidth=2, label='Optimal Epoch')

        # Add title, legend, and labels
        plt.title(f'{subfolder} - Fold {idx + 1}: Optimal Epoch Determination', fontsize=TITLE_FONT_SIZE)
        plt.xlabel('Epochs', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel('Value', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
        plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
        
        plt.legend(title=f'Optimal Epoch: {best_epoch + 1}', title_fontsize='13')

        # Save the plot
        filename = os.path.join(output_folder, f'fold_{idx + 1}_optimal_epoch.png')
        plt.savefig(filename)
        plt.close()
        
    print(f"✅ Optimal epoch plots for {subfolder} saved.")


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # 1. DEFINE RUN IDENTIFIER AND PATHS
    
    # Extract the run identifier (e.g., 'Results YOLOv12s') from the BASE_RESULTS_PATH structure
    RUN_IDENTIFIER = os.path.basename(os.path.dirname(os.path.dirname(BASE_RESULTS_PATH))) 
    
    # 2. GENERATE THE LIST OF CSV FILES TO PROCESS
    all_csv_paths = []
    for fold_name in MODEL_FOLD_NAMES:
        # Path: BASE_RESULTS_PATH / train_fold_1 / results.csv
        fold_path = os.path.join(BASE_RESULTS_PATH, fold_name)
        csv_file_path = os.path.join(fold_path, 'results.csv')
        
        if os.path.exists(csv_file_path):
            all_csv_paths.append(csv_file_path)
        else:
            print(f"WARNING: results.csv not found in {fold_path}. Skipping.")


    # 3. EXECUTE ANALYSIS
    
    print(f"\n==========================================")
    print(f"Processing Model Run: {RUN_IDENTIFIER}")
    
    csv_files_list = all_csv_paths
    
    summarize_metrics(csv_files_list)
    plot_all_results(RUN_IDENTIFIER, csv_files_list)
    plot_best_epoch(RUN_IDENTIFIER, csv_files_list)
    
    # 4. Process and plot confusion matrices (using hardcoded data)
    process_confusion_matrices(HARDCODED_CM)
    
    print("\n\n--- ALL POST-TRAINING ANALYSIS COMPLETE ---")