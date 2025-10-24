import os

# --- Environment Setup and Imports ---
# Set Ultralytics environment variable to use Pandas for better reporting
os.environ["ULTRALYTICS_USE_PANDAS"] = "True"
# Disable Weights & Biases (W&B) logging to suppress warnings
os.environ["WANDB_MODE"] = "disabled"

import numpy as np  # Used for calculating average performance metrics
import torch        # Core PyTorch library for GPU management
from codecarbon import EmissionsTracker  # Tool for tracking CO2 emissions
from ultralytics import YOLO  # YOLO model class for object detection

# Define the absolute path to the current working directory
current_dir = os.getcwd()

# --- GPU and CUDA Verification ---
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)

# Critical check: ensure a CUDA-enabled GPU is available before proceeding
if not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled GPU not found. The script requires a GPU with CUDA support.")

# Set the device for all operations to GPU
device = 'cuda'

# Define the log file path for CodeCarbon emissions
emissions_log_file = os.path.join(current_dir, "co2_emissions_log.txt")

# --- Emissions Logging Function ---
def log_emissions(emissions, total=False):
    """Appends total CO2 emissions data to the log file after all folds are complete."""
    with open(emissions_log_file, 'a') as f:
        if total:
            f.write(f"Total CO2 emissions for all folds: {emissions:.4f} kg\n")
        # Individual run data is typically logged automatically by CodeCarbon

# --- Checkpoint Management Function ---
def find_last_checkpoint(save_path):
    """
    Scans the training run directory for the latest epoch checkpoint file (epoch_X.pt).
    Returns the path to the latest .pt file or None if not found.
    """
    weights_dir = os.path.join(save_path, 'weights')
    if os.path.isdir(weights_dir):
        # Filter for files named 'epoch_X.pt'
        checkpoints = [f for f in os.listdir(weights_dir) if f.startswith('epoch_') and f.endswith('.pt')]
        
        # Extract epoch number and find the one corresponding to the latest training progress
        epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]
        if epoch_numbers:
            last_epoch = max(epoch_numbers)
            last_checkpoint_path = os.path.join(weights_dir, f'epoch_{last_epoch}.pt')
            return last_checkpoint_path
    return None

# --- Main Cross-Validation Execution Function ---
def run_cross_validation(num_folds=2, epochs=150, checkpoint_interval=10, resume_from_last_checkpoint=False):
    fold_performances = []  # List to store validation mAP for each fold
    total_emissions = 0.0   # Accumulator for overall CO2 emissions

    # Initialize or clear the emissions log file
    with open(emissions_log_file, 'w') as f:
        f.write("CO2 Emissions Log\n==================\n")

    # Define the name of the required base model file
    model_name = 'yolo12s.pt'
    
    # --- Critical Local File Check (Prevents Unnecessary Downloads) ---
    # Ensures yolo12s.pt is present locally to bypass Ultralytics fallback logic.
    if not os.path.exists(model_name):
        raise FileNotFoundError(
            f"FATAL ERROR: The required base model file '{model_name}' was not found in the current directory."
        )

    # Loop through each fold defined by num_folds
    for fold in range(1, num_folds + 1): 
        print(f"\n--- Fold {fold} ---")

        # Define dataset paths for the current fold
        folds_root = os.path.join(current_dir, "k_folds")
        train_images = os.path.join(folds_root, f'fold_{fold}', 'training', 'images')
        test_images = os.path.join(folds_root, f'fold_{fold}', 'test', 'images')

        # --- Dynamic YAML Configuration Generation ---
        # Create temporary YAML content pointing to the current fold's data
        data_yaml_content = f"""
        train: {train_images}
        val: {test_images}
        nc: 2
        names: ['Gun', 'Human']
        """
        # Save the YAML content to a temporary file
        data_yaml_path = os.path.join(current_dir, f'fold_{fold}_data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml_content)

        # Define project and name for Ultralytics run save directories
        train_project_dir = os.path.join(current_dir, "runs", "detect")
        train_save_name = f"train_fold_{fold}"
        run_dir = os.path.join(train_project_dir, train_save_name)
        
        # --- Model Loading and Resume Logic (YOLOv12 Obligatory) ---
        last_checkpoint = find_last_checkpoint(run_dir)
        
        # Variable to hold the actual model path (base model or checkpoint)
        model_to_load = model_name 
        
        if resume_from_last_checkpoint:
            
            # Path to the generic last checkpoint ('last.pt')
            last_pt_path = os.path.join(run_dir, 'weights', 'last.pt')
            
            if last_checkpoint:
                # Case 1: Found a specific epoch_X.pt checkpoint (most reliable)
                print(f"Resuming training for fold {fold} from specific checkpoint: {last_checkpoint}")
                model_to_load = last_checkpoint
                resume_flag = True
            elif os.path.exists(last_pt_path):
                 # Case 2: Found 'last.pt' (covers the scenario where epoch_X.pt failed to save)
                print(f"Resuming training for fold {fold} from last.pt: {last_pt_path}")
                model_to_load = last_pt_path
                resume_flag = True
            else:
                # Case 3: No checkpoints found. Starting from base model.
                print(f"No checkpoint found in {run_dir}. Starting from base model {model_name}.")
                model_to_load = model_name
                resume_flag = False
        else:
            # New training: load the base model
            print(f"Attempting to load the base model {model_name} for a new run...")
            model_to_load = model_name
            resume_flag = False
        
        # Load the model (either base model or checkpoint)
        try:
            model = YOLO(model_to_load)
        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Failed to load model {model_to_load}. The file may be corrupted. Error: {e}")
            
        # --- Emissions Tracking Start ---
        tracker = EmissionsTracker(log_level="WARNING", output_file=emissions_log_file)
        try:
            tracker.start()
        except Exception as e:
            print(f"Error starting emissions tracker: {e}")

        # --- Training Phase Execution ---
        
        if resume_flag:
            # If resuming, we rely on the model loaded from the checkpoint.
            # We omit 'project' and 'name' to force Ultralytics to continue in the model's existing directory.
            print(f"Resuming training directly in the loaded model's directory.")
            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=device,
                workers=8,
                plots=True,
                save_period=checkpoint_interval,
                resume=True # Forces the resume state
            )
        else:
            # New training: use project/name to create the new fold directory
            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=device,
                workers=8,
                plots=True,
                project=train_project_dir,
                name=train_save_name,
                save_period=checkpoint_interval,
                resume=False
            )


        # --- Validation Phase Execution ---
        val_save_name = f"val_fold_{fold}"
        val_results = model.val(
            data=data_yaml_path,
            device=device,
            batch=16,
            workers=8,
            project=train_project_dir,
            name=val_save_name,
        )
        print(f"Validation Results for Fold {fold}: {val_results.results_dict}")

        # Store results for final average calculation
        fold_performances.append(val_results.results_dict)

        # --- Emissions Tracking Stop and Logging ---
        try:
            emissions = tracker.stop()
            total_emissions += emissions
            print(f"CO2 emitted for fold {fold}: {emissions:.4f} kg")

        except Exception as e:
            print(f"Error stopping tracker or saving data: {e}")
        
        # Cleanup: remove the temporary YAML file
        os.remove(data_yaml_path)

    # --- Final Results Aggregation ---
    # Calculate the mean Average Precision at IoU=0.5 (mAP50) across all folds
    avg_map50 = np.mean([fold['metrics/mAP50(B)'] for fold in fold_performances])
    print("\n==================================")
    print(f"Average mAP50 across all folds: {avg_map50:.4f}")
    
    # Log the total accumulated CO2 emissions
    log_emissions(emissions=total_emissions, total=True)
    print(f"Total CO2 emissions for {num_folds} folds: {total_emissions:.4f} kg")
    print("==================================")

# --- Script Entry Point ---
if __name__ == '__main__':
    # ACTION: Set 'resume_from_last_checkpoint=True' to continue training from the last saved epoch.
    # If the script successfully resumed Fold 2, it should now continue or complete the remaining folds.
    run_cross_validation(num_folds=2, epochs=150, checkpoint_interval=10, resume_from_last_checkpoint=False)