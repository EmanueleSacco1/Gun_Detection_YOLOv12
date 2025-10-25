# 🎯 **Real-Time Gun Detection using YOLOv12 with Cross-Validation**

> 🧠 **Deep Learning Project — B.Sc. Thesis (Computer & Automation Engineering)**  
> **Università Politecnica delle Marche (UNIVPM)**  

A robust, real-time firearm detection framework leveraging **YOLOv12** and **5-Fold Cross-Validation** for enhanced **generalization**, **sustainability**, and **performance stability**.

---

## 🌟 **Highlights & Methodology**

### 🧩 Model Configurations
| Variant | Description |
| :-- | :-- |
| 🟢 **YOLOv12-Small (Standard)** | Baseline configuration for performance comparison. |
| 🔵 **YOLOv12-Large** | Larger model used to evaluate upper-bound accuracy and computational cost. |
| 🟣 **YOLOv12-Small (Augmented)** | Small model trained with **aggressive augmentation** to enhance generalization and reduce overfitting. |

### ⚖️ Cross-Validation
- **5-Fold CV** ensures model performance consistency and prevents data-split bias.  
- **Automated metric aggregation** and **epoch selection** scripts are included for analysis.

### 🌱 Sustainability
- Integrated **CO₂ emission tracking** using [CodeCarbon](https://codecarbon.io) for eco-aware training.

---

## 📁 **Project Structure**


```
GUN_DETECTION_YOLOv12/
├── Model/                           <-- Source code and configuration for training
│   ├── custom_hyp_s.yaml            <-- Augmentation settings (Mosaic, scale, etc.)
│   ├── gun_detection_yolov12s.py    <-- Training script (YOLOv12s)
│   └── ...
├── Preprocessing/                   <-- Scripts and output data for dataset preparation
│   ├── raw_dataset/                 <-- INPUT: Raw data/labels (multiclass, etc.)
│   ├── frames/                      <-- INTERMEDIATE: Cleaned, unique images/labels
├── Results/                         <-- Final analysis, metrics, and training artifacts
│   ├── post_training_analysis.py    <-- Automated metric aggregation and plotting script
│   └── Results YOLOv12s/l/          <-- Original run folders (containing best.pt, logs)
├── Test/                            <-- Inference scripts and test images
│   ├── Testing/                     <-- Test images/videos for inference
│   └── test_all_best_models.py      <-- Script for testing all 'best.pt' models
├──  k_folds/                        <-- FINAL INPUT: 5-Fold structure for CV

```


---

## ⚙️ **Pipeline Overview**

> The project runs in **3 main phases**, ensuring modularity and full reproducibility.

---

### 🧮 **Phase 1 – Data Preparation** (`Preprocessing/`)

| File | Function | Description |
| :-- | :-- | :-- |
| `process_and_rename_dataset.py` | 🧹 Filter & Rename | Filters irrelevant classes (e.g., “dog”), remaps labels (`Gun → 0`, `Human → 1`), and renames files to avoid collisions. |
| `k_fold_division.py` | 🔄 Cross-Validation Split | Randomly shuffles and divides cleaned data into **5 folds** for CV. |

---

### 🧠 **Phase 2 – Training & Evaluation** (`Model/`)

| File | Function | Description |
| :-- | :-- | :-- |
| `gun_detection_yolov12s_augmentation.py` | 🚀 Model Training | Executes 5-Fold training with **YOLOv12s**, applying augmentation from `custom_hyp_s.yaml`. |
| `custom_hyp_s.yaml` | 🎨 Augmentation Config | Defines transformations (Mosaic, scaling, color) to improve model generalization. |

---

### 📊 **Phase 3 – Post-Analysis & Testing** (`Results/` & `Test/`)

| File | Function | Description |
| :-- | :-- | :-- |
| `post_training_analysis.py` | 📈 Metrics & Plots | Aggregates performance metrics (mAP, F1), selects best epochs, and visualizes loss curves. |
| `test_all_best_models.py` | 🧩 Batch Inference | Tests all `best.pt` checkpoints from each fold for each model configuration. |
| `realtime_video_detection.py` | 🎥 Real-Time Detection | Performs live firearm detection on video streams. |

---

## ⚡ **Setup & Installation**

### 🧰 Requirements
Ensure you have:
- Python ≥ 3.10  
- CUDA-enabled GPU (recommended)  
- Conda or venv environment

### 📦 Installation
```bash
pip install -r requirements.txt
```

## 🚀 **Execution Steps**

All commands **must** be run from the project root:  
`GUN_DETECTION_YOLOv12/`

| Step | Command | Description |
| :-- | :-- | :-- |
| 1️⃣ | `python Preprocessing/process_and_rename_dataset.py` | Clean and prepare dataset. |
| 2️⃣ | `python Preprocessing/k_fold_division.py --num_groups 5` | Generate 5-Fold structure. |
| 3️⃣ | `python Model/gun_detection_yolov12s_augmentation.py` | Train YOLOv12-Small with augmentation. |
| 4️⃣ | `python Test/test_all_best_models.py` | Run inference on all best models. |
| 5️⃣ | `python Results/post_training_analysis.py` | Aggregate metrics and generate charts. |

---

## 🏁 **Conclusion**

This project delivers a **robust**, **generalized**, and **eco-conscious** firearm detection framework based on **YOLOv12**.  
By combining **Cross-Validation**, **Aggressive Augmentation**, and **Emission Tracking**, the system achieves both **technical performance** and **sustainability awareness**.

---

### 🧩 **Key Takeaways**

✅ **Generalization through Augmentation**  
→ The augmented YOLOv12-Small outperformed the standard model in cross-fold consistency.  

✅ **Statistical Validation via 5-Fold CV**  
→ Ensures model reliability, minimizing variance across data partitions.  

✅ **Performance vs. Efficiency Trade-off**  
→ YOLOv12-Large provides maximum performance; Small variant offers efficiency with minimal loss.  

✅ **Sustainability Tracking**  
→ Integrated **CodeCarbon** provides quantifiable training impact on CO₂ emissions.  

✅ **Reproducible Pipeline**  
→ Clear, modular folder structure for complete experimental traceability.  

---

> 📘 *This repository forms part of the B.Sc. Thesis presented at Università Politecnica delle Marche (UNIVPM).*  
