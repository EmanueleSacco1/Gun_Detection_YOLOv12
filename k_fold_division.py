import os
import random
import shutil

def divide_images_and_texts_into_groups(root_dir=".", frames_folder="frames", output_folder="k_folds", num_groups=2):
    """
    Raccoglie ricorsivamente tutti i file .jpg e .txt dalla cartella 'frames'
    e li divide in num_groups (fold) per la Cross-Validation.
    """
    
    full_frames_path = os.path.join(root_dir, frames_folder)
    
    if not os.path.isdir(full_frames_path):
        print(f"ERRORE: La cartella sorgente '{full_frames_path}' non esiste. Esegui prima gli script di estrazione e rinomina.")
        return

    # 1. Raccolta ricorsiva dei file e mappatura del percorso originale
    all_images = []
    # Usiamo un dizionario per mappare il nome file (chiave) al suo percorso completo (valore)
    file_map = {} 

    print(f"Avvio la raccolta dei file nella cartella: {full_frames_path}")
    
    for subdir, _, files in os.walk(full_frames_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg')):
                # Assumiamo che il nome del file sia univoco grazie alla rinomina
                image_name_only = filename 
                full_path = os.path.join(subdir, filename)
                
                # Aggiunge il nome del file alla lista da suddividere
                all_images.append(image_name_only)
                # Mappa il nome del file al suo percorso completo per la copia successiva
                file_map[image_name_only] = full_path

    if not all_images:
        print("Nessuna immagine trovata nella cartella 'frames'.")
        return

    # 2. Suddivisione casuale in K-Folds
    random.shuffle(all_images)
    
    # Determinare la dimensione dei gruppi
    num_images = len(all_images)
    num_images_per_group = num_images // num_groups
    remainder = num_images % num_groups

    groups = []
    index = 0
    for i in range(num_groups):
        group_size = num_images_per_group + (1 if i < remainder else 0)
        group = all_images[index:index + group_size]
        groups.append(group)
        index += group_size

    # 3. Creazione delle cartelle e copia dei file (Training vs Test)
    print(f"\nInizio la divisione e copia in {num_groups} fold totali ({num_images} immagini).")

    for fold_index in range(num_groups):
        fold_name = f'fold_{fold_index + 1}'
        fold_folder = os.path.join(root_dir, output_folder, fold_name)
        test_folder = os.path.join(fold_folder, 'test')
        training_folder = os.path.join(fold_folder, 'training')

        # Creazione della struttura di output
        os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'labels'), exist_ok=True)

        # Il gruppo corrente (fold_index) è il gruppo di TEST
        test_group = groups[fold_index]
        print(f"-> Creazione {fold_name}: {len(test_group)} immagini per Test.")

        # Ciclo di copia: Gruppo di TEST
        for image_name in test_group:
            # Recupera il percorso sorgente (in frames/sottocartella/...)
            source_image_path = file_map[image_name]
            image_name_base, _ = os.path.splitext(image_name)
            source_txt_path = os.path.join(os.path.dirname(source_image_path), f"{image_name_base}.txt")
            
            # Copia Immagine
            shutil.copy(source_image_path, os.path.join(test_folder, 'images', image_name))
            
            # Copia Label (.txt) se esiste
            if os.path.exists(source_txt_path):
                shutil.copy(source_txt_path, os.path.join(test_folder, 'labels', f"{image_name_base}.txt"))

        # Ciclo di copia: Gruppo di TRAINING (tutti gli altri gruppi)
        for i in range(num_groups):
            if i != fold_index:
                training_group = groups[i]
                for image_name in training_group:
                    source_image_path = file_map[image_name]
                    image_name_base, _ = os.path.splitext(image_name)
                    source_txt_path = os.path.join(os.path.dirname(source_image_path), f"{image_name_base}.txt")
                    
                    # Copia Immagine
                    shutil.copy(source_image_path, os.path.join(training_folder, 'images', image_name))
                    
                    # Copia Label (.txt) se esiste
                    if os.path.exists(source_txt_path):
                        shutil.copy(source_txt_path, os.path.join(training_folder, 'labels', f"{image_name_base}.txt"))

    print(f'\nDivisione completata. I file sono stati organizzati in {num_groups} cartelle (fold) nella directory "{output_folder}".')

if __name__ == "__main__":
    # Esegui dalla cartella Gun_Action_Recognition_Dataset
    divide_images_and_texts_into_groups(root_dir=".", frames_folder="frames", output_folder="k_folds", num_groups=2)