import os

def rename_files_in_frames_folder(root_dir, target_dir="frames"):
    """
    Rinomina ricorsivamente i file .jpg e .txt all'interno della cartella 'frames',
    aggiungendo la struttura delle sottocartelle come suffisso al nome del file.
    Esempio: frame_00000.jpg in frames/Handgun/PAH1_... diventa frame_00000_Handgun_PAH1_....jpg
    """
    
    # Percorso completo della cartella 'frames'
    full_target_path = os.path.join(root_dir, target_dir)
    
    if not os.path.isdir(full_target_path):
        print(f"ERRORE: La cartella di destinazione '{target_dir}' non esiste in {root_dir}. Esegui prima lo script di estrazione.")
        return

    print(f"Avvio la rinomina ricorsiva nella cartella: {full_target_path}")
    print("-" * 50)

    # Scansiona ricorsivamente tutte le sottocartelle all'interno di 'frames'
    for subdir, _, files in os.walk(full_target_path):
        
        # Calcola il percorso relativo A PARTIRE dalla cartella 'frames'
        relative_path = os.path.relpath(subdir, full_target_path)
        
        # Se siamo nella cartella 'frames' stessa, non aggiungiamo suffisso di cartella
        if relative_path == ".":
            suffix_to_add = ""
        else:
            # Unisce le parti del percorso relativo con '_'
            # Esempio: "Handgun/PAH1_C1_P1_V1_HB_3" diventa "_Handgun_PAH1_C1_P1_V1_HB_3"
            suffix_to_add = "_" + "_".join(relative_path.split(os.sep)) 
        
        renamed_count = 0
        for filename in files:
            name, ext = os.path.splitext(filename)
            
            # Applica la rinomina solo a .jpg e .txt
            if ext.lower() in ['.jpg', '.txt']:
                old_path = os.path.join(subdir, filename)
                
                # Crea il nuovo nome file
                new_filename = f"{name}{suffix_to_add}{ext}"
                new_path = os.path.join(subdir, new_filename)
                
                # Rinomina
                os.rename(old_path, new_path)
                renamed_count += 1
        
        if renamed_count > 0:
            print(f"Rinominate {renamed_count} file in {os.path.relpath(subdir, full_target_path)}")

    print("-" * 50)
    print("Rinomina completata.")

if __name__ == "__main__":
    # La cartella di base (dovrebbe essere Gun_Action_Recognition_Dataset)
    base_folder = os.getcwd() 
    rename_files_in_frames_folder(base_folder)