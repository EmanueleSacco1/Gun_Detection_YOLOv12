# Gun Action Recognition Dataset

This repository contains scripts and utilities for preparing, processing, and training object detection models (YOLOv12s) on a custom gun action recognition dataset. The workflow includes frame extraction, annotation conversion, file renaming, K-fold dataset splitting, and model training with CO2 emissions tracking.

---

## Dataset Structure

The dataset is organized as follows:

```
Gun_Action_Recognition_Dataset/
├── Handgun/
│   ├── PAH1_C1_P1_V1_HB_3/
│   │   ├── video.mp4
│   │   └── label.json
│   └── ...
├── No_Gun/
│   └── ...
├── frames/                # Generated: extracted frames and YOLO labels
├── k_folds/               # Generated: K-fold split for cross-validation
├── runs/                  # Generated: YOLO training outputs
├── *.py                   # Scripts (see below)
└── README.md
```

- Each subfolder under `Handgun` or `No_Gun` contains a `video.mp4` and, if available, a `label.json` in COCO format.

---

## Scripts Overview

### 1. `extract_frames_from_videos.py`

- **Purpose:** Recursively extracts frames from all `video.mp4` files, draws bounding boxes if `label.json` is present, and generates YOLO-format `.txt` labels.
- **Output:** Saves frames and labels in a mirrored folder structure under `frames/`.

### 2. `rename_img_with_subfolder.py`

- **Purpose:** Renames all `.jpg` and `.txt` files in `frames/` to include their subfolder path as a suffix, ensuring unique filenames for K-fold splitting.

### 3. `k_fold_division.py`

- **Purpose:** Splits all images (and corresponding labels) in `frames/` into K folds for cross-validation. Each fold contains separate `training` and `test` sets with `images/` and `labels/` subfolders.

### 4. `gun_detection_yolov12s.py`

- **Purpose:** Trains and evaluates a YOLOv12s model using the K-fold splits. Tracks CO2 emissions using [CodeCarbon](https://mlco2.github.io/codecarbon/). Supports checkpointing and resuming.

---

## Setup & Requirements

- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [CodeCarbon](https://mlco2.github.io/codecarbon/)
- [PyTorch](https://pytorch.org/)
- CUDA-enabled GPU (required for training)

Install dependencies:
```bash
pip install ultralytics opencv-python codecarbon torch
```

---

## Usage

1. **Extract frames and labels:**
   ```bash
   python extract_frames_from_videos.py
   ```

2. **Rename files for uniqueness:**
   ```bash
   python rename_img_with_subfolder.py
   ```

3. **Split dataset into K folds:**
   ```bash
   python k_fold_division.py
   ```
   - You can adjust the number of folds by editing the `num_groups` parameter in the script.

4. **Train and evaluate YOLOv12s with K-fold cross-validation:**
   ```bash
   python gun_detection_yolov12s.py
   ```
   - Adjust `num_folds`, `epochs`, and other parameters as needed in the script.

---

## Notes

- The scripts assume the presence of `video.mp4` and, if available, `label.json` in each subfolder.
- YOLO labels are generated only if bounding box annotations are present.
- CO2 emissions for each fold are logged in `co2_emissions_log.txt`.
- The YOLOv12s model weights (`yolo12s.pt`) must be available in the working directory or accessible by Ultralytics.

---

## Citation

If you use this dataset or code, please cite the original authors and this repository.

---

## License

This project is licensed under the MIT License.
