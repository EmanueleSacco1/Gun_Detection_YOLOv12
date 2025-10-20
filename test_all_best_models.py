import os
from ultralytics import YOLO

# --- Configuration Paths ---

# Define the root directory of the project 
current_dir = os.getcwd()

# Base path for the results folders containing 'runs/detect'
RESULTS_ROOT = os.path.join(current_dir, "Results YOLOv12") 

# Folder name containing the test images
TEST_IMAGE_FOLDER = 'Testing'
# Full path to the source images
source_image_path = os.path.join(current_dir, TEST_IMAGE_FOLDER)

# Output directory path (results will be saved within the 'Testing' folder)
OUTPUT_PROJECT_DIR = source_image_path

# --- Models to Test Mapping ---

# Define all specific model paths using the current directory as root for safety and clarity.
# Keys are used as the output subfolder names.
MODELS_TO_TEST = {
    # YOLOv12-Small Folds 
    "yolov12s_fold1": os.path.join(current_dir, "Results YOLOv12s", "runs", "detect", "train_fold_1", "weights", "best.pt"),
    "yolov12s_fold2": os.path.join(current_dir, "Results YOLOv12s", "runs", "detect", "train_fold_2", "weights", "best.pt"),
    # YOLOv12-Large Folds 
    "yolov12l_fold1": os.path.join(current_dir, "Results YOLOv12l", "runs", "detect", "train_fold_1", "weights", "best.pt"),
    "yolov12l_fold2": os.path.join(current_dir, "Results YOLOv12l", "runs", "detect", "train_fold_2", "weights", "best.pt"),
}

# --- Main Testing Function ---

def run_all_tests():
    """Executes inference for all specified best.pt models on the dedicated test set."""
    
    # Pre-check for the existence of the test image folder
    if not os.path.isdir(source_image_path):
        raise FileNotFoundError(f"ERROR: Test folder not found: {source_image_path}. Please ensure that '{TEST_IMAGE_FOLDER}' exists in the root.")

    print(f"Starting inference test on {len(MODELS_TO_TEST)} models.")
    print("-" * 50)

    # Iterate through the list of models
    for run_name, model_path in MODELS_TO_TEST.items():
        
        if not os.path.exists(model_path):
            # If the model does not exist, raise a FileNotFoundError to stop the script.
            raise FileNotFoundError(
                f"FATAL ERROR: Required model not found for {run_name}. "
                f"Missing path: {model_path}"
            )
        
        print(f"Testing Model: {run_name}")
        print(f"Loading from: {model_path}")
        
        try:
            # 1. Load the model using the Ultralytics YOLO class
            model = YOLO(model_path)
            
            # 2. Run inference (predict) on the test images
            model.predict(
                source=source_image_path,
                conf=0.25,        # Confidence threshold (adjust based on performance needs)
                iou=0.7,          # IoU threshold for Non-Maximum Suppression (NMS)
                imgsz=640,        # Image size (must match training resolution)
                device='cuda',    # Use GPU for accelerated inference
                save=True,        # Save images with drawn bounding boxes
                save_txt=True,    # Save results in standard YOLO text format (.txt)
                project=OUTPUT_PROJECT_DIR, # Sets 'Testing' as the base output folder
                name=run_name,              # Sets a unique subfolder name (e.g., 'yolov12s_fold1')
                exist_ok=True,    # Allows overwriting/resuming runs without raising an error
                verbose=False     # Suppress detailed per-image logging
            )
            
            output_dir = os.path.join(OUTPUT_PROJECT_DIR, "runs", "detect", run_name)
            print(f"✅ Success: Results saved in {output_dir}")

        except Exception as e:
            # This block captures GPU errors, YOLO loading issues, or other runtime problems.
            print(f"❌ CRITICAL ERROR during inference for {run_name}: {e}")
        

    print("-" * 50)
    print("All model inference tests completed (or terminated due to error).")

if __name__ == "__main__":
    run_all_tests()