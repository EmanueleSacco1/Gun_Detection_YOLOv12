import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# --- GLOBAL CONFIGURATION ---
# Base directory for the script (assumes the script is inside the 'Results' folder)
BASE_DIR = 'Results'
# Correctly navigate up one level from 'Results' to the project root, then into 'runs/detect'
RUNS_DIR = os.path.join(os.path.dirname(os.getcwd()), 'runs', 'detect') 
DATA_SUBFOLDER = 'Data' # Folder where result CSVs are expected to be copied inside BASE_DIR
PLOTS_SUBFOLDER = os.path.join(BASE_DIR, 'Plots')

# Define the names of the model runs to analyze (Must adjust this list to match your subfolders)
# Example: If your training created folders like runs/detect/yolov12s_150_fold_1, etc.
# You need to manually copy your CSVs to Results/Data/yolov12s_150/
MODEL_RUN_FOLDERS = ['yolov12s_150_sample'] # EXAMPLE: ADJUST THIS LIST (Folder names containing results1.csv...results5.csv)

# --- HELPER FUNCTIONS ---

def create_output_dirs():
    """Ensures the necessary output directories exist."""
    # os.makedirs(DATA_SUBFOLDER, exist_ok=True) # Assuming data is manually copied here
    os.makedirs(PLOTS_SUBFOLDER, exist_ok=True)
    print(f"Output directories verified: {os.path.abspath(PLOTS_SUBFOLDER)}")

def calculate_f1(precision: float, recall: float) -> float:
    """Calculates the F1-measure from precision and recall."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# --- 1. METRICS SUMMARY AND AVERAGING (Adapted from mean_value_results.py) ---

def get_best_metrics(file_path: str) -> Dict[str, float]:
    """Extracts the maximum values for key metrics from a single fold's results CSV."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip() # Clean column names
    
    # Ensure columns are numeric
    data['metrics/precision(B)'] = pd.to_numeric(data['metrics/precision(B)'], errors='coerce')
    data['metrics/recall(B)'] = pd.to_numeric(data['metrics/recall(B)'], errors='coerce')
    data['metrics/mAP50(B)'] = pd.to_numeric(data['metrics/mAP50(B)'], errors='coerce')
    
    # Find the maximum value for precision, recall, and mAP50 across all epochs
    best_precision = data['metrics/precision(B)'].max()
    best_recall = data['metrics/recall(B)'].max()
    best_mAP50 = data['metrics/mAP50(B)'].max()
    
    # Calculate F1-measure
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
    
    # Extract the best metrics from each fold's CSV
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

    # Calculate the average of the best achieved values across all folds
    mean_metrics = {
        key: sum(d[key] for d in all_best_metrics) / len(all_best_metrics)
        for key in all_best_metrics[0].keys()
    }

    # Print results
    print("\n--- Mean Best Performance Across All Folds ---")
    for i, metrics in enumerate(all_best_metrics, 1):
        print(f"Fold {i} - Precision: {metrics['precision']:.5f}, Recall: {metrics['recall']:.5f}, F1: {metrics['F1']:.5f}, mAP50: {metrics['mAP50']:.5f}")
    
    print("\n[AVERAGE] Best Metrics:")
    for key, value in mean_metrics.items():
        print(f"{key.capitalize()}: {value:.5f}")

# --- 2. CONFUSION MATRIX PLOTTING (Adapted from plot_avg_confusion_matrix.py) ---

# NOTE: Hardcoded CM values are used for demonstration. Adjust these values based on your actual validation results.
# Format: [TP, FP, FN, TN]. TN is typically 0 in object detection for simplified CMs.
HARDCODED_CM = np.array([
    [8630, 7645, 3809, 0],  # Fold 1 
    [9227, 7212, 4788, 0],  # Fold 2
    [10666, 7281, 6010, 0],  # Fold 3
    [9951, 7058, 4653, 0],  # Fold 4
    [8141, 10149, 3796, 0], # Fold 5
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


# --- 3. LOSS AND METRIC PLOTTING (Adapted from plot_results.py) ---

def plot_all_results(subfolder: str):
    """Loads all fold results for a model run, calculates the mean, and generates all plots."""
    
    subfolder_path = os.path.join(DATA_SUBFOLDER, subfolder)
    
    if not os.path.isdir(subfolder_path):
        print(f"Skipping plotting: Data folder not found at {subfolder_path}")
        return
    
    # Prepare the list of CSV files (assuming 5 folds)
    csv_files = [os.path.join(subfolder_path, f'results{i}.csv') for i in range(1, 6)]

    # Create a specific output folder for this subfolder's plots
    output_folder = os.path.join(PLOTS_SUBFOLDER, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    dfs = [] # List to store the DataFrames

    # Load all the CSV files for this subfolder
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()  # Clean column names
            df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
            dfs.append(df)
        else:
            print(f"File {file} does not exist. Skipping.")

    if not dfs:
        print(f"No valid data found in folder: {subfolder_path}. Skipping plots.")
        return

    # Concatenate the DataFrames and calculate the mean of numeric columns only
    mean_df = pd.concat(dfs).groupby(level=0).mean()

    # Define plotting aesthetics
    axis_label_font_size = 14
    tick_label_font_size = 12
    title_font_size = 16
    legend_font_size = 10
    
    # Pairs of fields to plot (train and val of the same type)
    loss_pairs = [('train/box_loss', 'val/box_loss'), 
                  ('train/cls_loss', 'val/cls_loss'), 
                  ('train/dfl_loss', 'val/dfl_loss')]

    # Metrics fields to plot separately (precision, recall, etc.)
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
        plt.title(combined_title, fontsize=title_font_size)
        plt.legend(fontsize=legend_font_size)
        plt.xlabel('Epochs', fontsize=axis_label_font_size)
        plt.ylabel('Value', fontsize=axis_label_font_size)
        plt.xticks(fontsize=tick_label_font_size)
        plt.yticks(fontsize=tick_label_font_size)
        
        filename = os.path.join(output_folder, f'combined_{train_field.split("/")[1].lower()}_mean_only.png')
        plt.savefig(filename)
        plt.close()

    # 2. Plot single graphs for each metric with folds and mean
    for field, pretty_name in metrics_fields:
        plt.figure(figsize=(10, 6))
        for idx, df in enumerate(dfs):
            plt.plot(df[field], alpha=0.5, label=f'Fold {idx+1}')
        plt.plot(mean_df[field], label='Mean', color='black', linewidth=2)
        plt.title(f'{subfolder}: {pretty_name}', fontsize=title_font_size)
        plt.legend(fontsize=legend_font_size)
        plt.xlabel('Epochs', fontsize=axis_label_font_size)
        plt.ylabel('Value', fontsize=axis_label_font_size)
        plt.xticks(fontsize=tick_label_font_size)
        plt.yticks(fontsize=tick_label_font_size)
        filename = os.path.join(output_folder, f'{field.split("/")[1].lower()}_metric.png')
        plt.savefig(filename)
        plt.close()
    
    print(f"✅ Plots for {subfolder} saved to {output_folder}")


# --- 4. OPTIMAL EPOCH DETERMINATION (Adapted from plot_best_epoch.py) ---

def plot_best_epoch(subfolder: str):
    """
    Calculates the optimal epoch for each fold based on a weighted combined score 
    and plots the metrics with the optimal epoch marked.
    """
    subfolder_path = os.path.join(DATA_SUBFOLDER, subfolder)
    csv_files = [os.path.join(subfolder_path, f'results{i}.csv') for i in range(1, 6)]

    dfs = [] # List to store the DataFrames

    # Load all CSV files and clean columns
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            # Ensure all necessary columns are numeric
            cols_to_numeric = ['metrics/mAP50(B)', 'val/box_loss', 'val/cls_loss', 
                               'metrics/precision(B)', 'metrics/recall(B)']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            dfs.append(df)

    if not dfs:
        return

    output_folder = os.path.join(PLOTS_SUBFOLDER, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Normalized coefficients for calculating the combined score (Custom weights)
    coeff_box_loss = 0.5  # Weight for box losses (negative contribution)
    coeff_cls_loss = 0.5  # Weight for class losses (negative contribution)
    coeff_precision = 0.375  # Weight for precision
    coeff_recall = 0.375  # Weight for recall
    coeff_map50 = 0.25  # Weight for mAP50

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
        plt.title(f'{subfolder} - Fold {idx + 1}: Optimal Epoch Determination', fontsize=title_font_size)
        plt.xlabel('Epochs', fontsize=axis_label_font_size)
        plt.ylabel('Value', fontsize=axis_label_font_size)
        plt.xticks(fontsize=tick_label_font_size)
        plt.yticks(fontsize=tick_label_font_size)
        
        plt.legend(title=f'Optimal Epoch: {best_epoch + 1}', title_fontsize='13')

        # Save the plot
        filename = os.path.join(output_folder, f'fold_{idx + 1}_optimal_epoch.png')
        plt.savefig(filename)
        plt.close()
        
    print(f"✅ Optimal epoch plots for {subfolder} saved.")


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    create_output_dirs()
    
    # 1. Process all metrics and plotting for each defined model run
    for model_run in MODEL_RUN_FOLDERS:
        print(f"\n==========================================")
        print(f"Processing Model Run: {model_run}")
        
        # Assuming the CSV files are named 'results1.csv' through 'results5.csv'
        csv_files = [os.path.join(DATA_SUBFOLDER, model_run, f'results{i}.csv') for i in range(1, 6)]
        
        # Execute Metric Summarization
        summarize_metrics(csv_files)
        
        # Execute General Loss/Metric Plotting
        plot_all_results(model_run)
        
        # Execute Optimal Epoch Plotting
        plot_best_epoch(model_run)

    # 2. Process and plot confusion matrices (using hardcoded data for demonstration)
    process_confusion_matrices(HARDCODED_CM)
    
    print("\n\n--- ALL POST-TRAINING ANALYSIS COMPLETE ---")