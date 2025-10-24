import os
from ultralytics import YOLO

# --- Configuration Paths and Parameters ---

# Define the root directory of the project
current_dir = os.getcwd()

#  SET YOUR VIDEO FILENAME HERE 
VIDEO_FILENAME = 'test_video.mp4' 

# Full path to the source video file, expected to be in the 'Testing' folder
source_video_path = os.path.join(current_dir, 'Test', 'Testing', VIDEO_FILENAME)

# Path to the specific BEST model checkpoint to be used for inference.
# Example uses YOLOv12l Fold 1. ADJUST THE PATH IF NECESSARY.
MODEL_PATH = os.path.join(current_dir, 'Results' "Results YOLOv12s", "runs", "detect", "train_fold_1", "weights", "best.pt")

# Output directory for saving inference results (if 'save=True' is enabled)
OUTPUT_PROJECT_DIR = os.path.join(current_dir, 'Test', 'Testing', 'Realtime_Output')
OUTPUT_RUN_NAME = 'live_run_yolo12l_f1'

# --- Initial Verification Checks ---

# Check if the specified model file exists before proceeding
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FATAL ERROR: Model not found: {MODEL_PATH}")

# Check if the source video file exists
if not os.path.exists(source_video_path):
    raise FileNotFoundError(f"FATAL ERROR: Video file not found at path: {source_video_path}")

# --- Real-Time Detection Function ---

def run_realtime_detection():
    """Loads the model and performs real-time object detection on the video stream."""
    print(f"Loading model: {os.path.basename(MODEL_PATH)}")
    
    try:
        # 1. Load the YOLO model from the best checkpoint
        model = YOLO(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    print("-" * 50)
    print(f"Starting real-time detection on video: {VIDEO_FILENAME}")
    print("Press 'q' on the video window to stop the stream.")
    print("-" * 50)

    # 2. Execute inference in streaming mode
    # 'stream=True' enables processing frame-by-frame using a generator.
    # 'show=True' uses OpenCV internally to display the output with drawn bounding boxes.
    results_generator = model.predict(
        source=source_video_path,
        conf=0.40,        # Confidence threshold (adjusted higher to reduce live false positives)
        imgsz=640,        # Image size (must match training input)
        device='cuda',    # Use GPU for accelerated processing
        stream=True,      # Key parameter for video/live feed processing
        show=True,        # Key parameter to display the results in a pop-up window
        save=False,       # Set to False to prevent saving frames during live view
        project=OUTPUT_PROJECT_DIR, 
        name=OUTPUT_RUN_NAME,     
    )
    
    # 3. Iterate over the results generator to keep the stream active
    # The loop ensures all frames are processed until the end of the video or 'q' is pressed.
    for r in results_generator:
        pass 
    
    print("\nVideo detection stream terminated.")

if __name__ == "__main__":
    run_realtime_detection()