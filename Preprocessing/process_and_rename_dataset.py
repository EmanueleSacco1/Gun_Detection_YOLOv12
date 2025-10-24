import os
import shutil

# --- GLOBAL YOLO MAPPING CONFIGURATION ---

# Map the class IDs from the external dataset to your new standard IDs (0 and 1).
# [Original_ID]: [New_Standard_ID]
ID_MAPPING = {
    16: 0,  # 16 (Gun)  -> 0 (Gun)
    15: 1,  # 15 (Human) -> 1 (Human)
}

# --- UNIFIED PROCESSING FUNCTION ---

def process_and_rename_dataset(raw_dir, output_frames_dir, raw_dataset_name="raw_dataset"):
    """
    Scans the raw_dir, filters and remaps labels, and saves them to the final 'frames/'
    directory with unique names based on their original folder path.
    """
    print(f"Starting unified processing (Filter, Map, Rename) in: {raw_dir}")
    
    # 1. Create the final output structure (frames/)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    labels_processed_count = 0
    images_copied_count = 0

    # 2. Recursive Scan
    for subdir, _, files in os.walk(raw_dir):
        
        # Calculate the unique suffix based on the relative path from the raw_dir root
        # Example: if raw_dir is 'raw_dataset', and subdir is 'raw_dataset/set_A', 
        # relative_path will be 'set_A'.
        relative_path = os.path.relpath(subdir, raw_dir)
        
        # Create a unique suffix: "_set_A_subfolder_name"
        if relative_path == ".":
            suffix_to_add = ""
        else:
            # Use the entire relative path, including the raw_dataset root name for maximum uniqueness
            full_relative_path = os.path.join(raw_dataset_name, relative_path)
            # Joins parts with underscores
            suffix_to_add = "_" + "_".join(full_relative_path.split(os.sep)) 
        
        
        for filename in files:
            name, ext = os.path.splitext(filename)
            
            # 3. Process Label Files (.txt)
            if ext.lower() == '.txt':
                label_path = os.path.join(subdir, filename)
                new_lines = []
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Apply Filter and Map Logic
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5: continue 
                            
                        original_class_id_str = parts[0]
                        
                        try:
                            original_id = int(original_class_id_str)
                            
                            if original_id in ID_MAPPING:
                                # Found a relevant class (15 or 16)
                                new_id = ID_MAPPING[original_id] 
                                
                                # Reconstruct the line with the new ID (0 or 1)
                                new_line = f"{new_id} {' '.join(parts[1:])}"
                                new_lines.append(new_line)
                            
                        except ValueError:
                            # Skip if class ID is non-numeric
                            continue

                    
                    # 4. RENAME and Save Mapped File
                    if new_lines:
                        # Create the unique, renamed filename
                        renamed_label_filename = f"{name}{suffix_to_add}{ext}"
                        output_label_path = os.path.join(output_frames_dir, renamed_label_filename)
                        
                        # Save the new .txt file
                        with open(output_label_path, 'w') as f:
                            f.write('\n'.join(new_lines) + '\n')
                        labels_processed_count += 1

                        # 5. RENAME and Copy Image
                        image_name = name + '.jpg'
                        image_path = os.path.join(subdir, image_name)
                        
                        if os.path.exists(image_path):
                            # Create the unique, renamed image filename
                            renamed_image_filename = f"{name}{suffix_to_add}.jpg"
                            shutil.copy(image_path, os.path.join(output_frames_dir, renamed_image_filename))
                            images_copied_count += 1
                        
                except Exception as e:
                    print(f"[FATAL ERROR] processing {filename}: {e}. Skipping file.")

    print("-" * 50)
    print(f"Process completed.")
    print(f"Total Labels Processed/Saved: {labels_processed_count}")
    print(f"Total Images Copied: {images_copied_count}")
    print(f"All files saved to: {output_frames_dir}")


if __name__ == "__main__":
    
    current_dir = os.getcwd()
    
    # --- Configuration ---
    RAW_DATASET_NAME = 'raw_dataset' # The name of your single input folder
    RAW_DATASET_PATH = os.path.join(current_dir, RAW_DATASET_NAME)
    OUTPUT_FRAMES_PATH = os.path.join(current_dir, 'Preprocessing', 'frames') # The final destination folder
    
    if not os.path.isdir(RAW_DATASET_PATH):
        print(f"CRITICAL ERROR: The input folder '{RAW_DATASET_NAME}' was not found at {RAW_DATASET_PATH}.")
    else:
        process_and_rename_dataset(RAW_DATASET_PATH, OUTPUT_FRAMES_PATH, RAW_DATASET_NAME)