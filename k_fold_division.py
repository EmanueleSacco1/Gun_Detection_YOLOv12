import os
import random
import shutil
import argparse

def divide_images_and_texts_into_groups(root_dir=".", frames_folder="frames", output_folder="k_folds", num_groups=2):
    """
    Recursively collects all .jpg and corresponding .txt files from the 'frames' folder
    and divides them into num_groups (folds) for K-Fold Cross-Validation.
    """
    
    full_frames_path = os.path.join(root_dir, frames_folder)
    
    if not os.path.isdir(full_frames_path):
        print(f"ERROR: Source folder '{full_frames_path}' does not exist. Please run extraction and renaming scripts first.")
        return

    # 1. Recursive Collection of Files and Path Mapping
    all_images = []
    # Dictionary to map unique filename (key) to its full source path (value)
    file_map = {} 

    print(f"Starting file collection in: {full_frames_path}")
    
    # Traverse the frames folder and its subdirectories
    for subdir, _, files in os.walk(full_frames_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg')):
                # Assumes the filename is unique due to prior renaming process
                image_name_only = filename 
                full_path = os.path.join(subdir, filename)
                
                # Add the filename to the list to be shuffled and divided
                all_images.append(image_name_only)
                # Map the filename to its absolute source path for later copying
                file_map[image_name_only] = full_path

    if not all_images:
        print("No images found in the 'frames' folder.")
        return

    # 2. Random Shuffling and Group Subdivision
    random.shuffle(all_images)
    
    # Determine group sizes for even distribution
    num_images = len(all_images)
    num_images_per_group = num_images // num_groups
    remainder = num_images % num_groups

    groups = []
    index = 0
    # Create N groups (folds) containing lists of filenames
    for i in range(num_groups):
        group_size = num_images_per_group + (1 if i < remainder else 0)
        group = all_images[index:index + group_size]
        groups.append(group)
        index += group_size

    # 3. Folder Creation and File Copying (Training vs Test Split)
    print(f"\nStarting division and copying into {num_groups} total folds ({num_images} images).")

    for fold_index in range(num_groups):
        fold_name = f'fold_{fold_index + 1}'
        fold_folder = os.path.join(root_dir, output_folder, fold_name)
        test_folder = os.path.join(fold_folder, 'test')
        training_folder = os.path.join(fold_folder, 'training')

        # Create the Ultralytics-compatible output structure
        os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'labels'), exist_ok=True)

        # The current group (fold_index) is designated as the TEST group for this iteration
        test_group = groups[fold_index]
        print(f"-> Creating {fold_name}: {len(test_group)} images allocated for Test.")

        # Copy Loop: TEST Group
        for image_name in test_group:
            source_image_path = file_map[image_name]
            image_name_base, _ = os.path.splitext(image_name)
            # Find the corresponding label file in the source directory
            source_txt_path = os.path.join(os.path.dirname(source_image_path), f"{image_name_base}.txt")
            
            # Copy Image to the Test/images folder
            shutil.copy(source_image_path, os.path.join(test_folder, 'images', image_name))
            
            # Copy Label (.txt) if it exists (necessary for labeled files, ignored for 'No_Gun' images)
            if os.path.exists(source_txt_path):
                shutil.copy(source_txt_path, os.path.join(test_folder, 'labels', f"{image_name_base}.txt"))

        # Copy Loop: TRAINING Group (All other groups are training data)
        for i in range(num_groups):
            if i != fold_index:
                training_group = groups[i]
                for image_name in training_group:
                    source_image_path = file_map[image_name]
                    image_name_base, _ = os.path.splitext(image_name)
                    source_txt_path = os.path.join(os.path.dirname(source_image_path), f"{image_name_base}.txt")
                    
                    # Copy Image to the Training/images folder
                    shutil.copy(source_image_path, os.path.join(training_folder, 'images', image_name))
                    
                    # Copy Label (.txt) if it exists
                    if os.path.exists(source_txt_path):
                        shutil.copy(source_txt_path, os.path.join(training_folder, 'labels', f"{image_name_base}.txt"))

    print(f'\nDivision complete. Files are organized into {num_groups} folders (folds) in the "{output_folder}" directory.')

if __name__ == "__main__":
    # Allows command-line specification for number of groups (folds)
    parser = argparse.ArgumentParser(description="Divide images and labels into K-folds for cross-validation.")
    parser.add_argument('--num_groups', type=int, default=2, help='Number of folds (K) to divide the dataset into.')
    args = parser.parse_args()
    
    # Execute from the project root directory
    divide_images_and_texts_into_groups(root_dir=".", frames_folder="frames", output_folder="k_folds", num_groups=args.num_groups)