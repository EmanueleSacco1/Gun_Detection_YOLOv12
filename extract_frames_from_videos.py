import os
import cv2
import json

# --- Configurazione YOLO ---
# Assumiamo che la classe per le armi sia 0, come è prassi con un solo oggetto.
GUN_CLASS_ID = 0 

def extract_frames_from_videos(root_dir, output_dir="frames"):
    """
    Scorre ricorsivamente tutte le sottocartelle.
    - Se trova video.mp4 e label.json: Estrae, disegna i BB e crea label YOLO (.txt).
    - Se trova solo video.mp4: Estrae solo i frame.
    """
    
    video_count = 0 
    print(f"Avvio l'elaborazione ricorsiva nella cartella: {os.path.abspath(root_dir)}")
    print("-" * 80)

    for subdir, _, files in os.walk(root_dir):
        
        # Ignora la cartella di output
        if os.path.basename(subdir) == output_dir and os.path.abspath(os.path.join(subdir, '..')) == os.path.abspath(root_dir):
            continue
            
        has_video = "video.mp4" in files
        has_label = "label.json" in files

        if has_video:
            video_path = os.path.join(subdir, "video.mp4")
            
            # Crea la cartella di output mantenendo la struttura
            rel_path = os.path.relpath(subdir, root_dir)
            output_subdir = os.path.join(root_dir, output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            print(f"\n-> Scansione di: {os.path.relpath(subdir, root_dir)}")

            # --- Preparazione Annotazioni (Solo se label.json è presente) ---
            frame_annotations = {}
            category_name = "Object"
            
            if has_label:
                label_path = os.path.join(subdir, "label.json")
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                    
                    for annotation in labels.get("annotations", []):
                        image_id = annotation["image_id"]
                        bbox = annotation["bbox"] # Formato COCO: [x_min, y_min, width, height]
                        
                        if image_id not in frame_annotations:
                            frame_annotations[image_id] = []
                        
                        frame_annotations[image_id].append(bbox)
                    
                    category_name = labels.get("categories", [{}])[0].get("name", "Object")
                    print(f"[LABEL] Trovate annotazioni per la categoria: {category_name}")

                except Exception as e:
                    print(f"[ERRORE] durante l'analisi di {label_path}: {e}. Procedo con la sola estrazione dei frame.")
                    has_label = False # Disabilita l'elaborazione delle label per questo video
            
            # --- 2. Estrazione e Disegno ---
            cap = cv2.VideoCapture(video_path)
            frame_index = 0
            
            if not cap.isOpened():
                print(f"[ERRORE] Impossibile aprire il video {video_path}. Salto.")
                continue
                
            # Ottieni le dimensioni del frame per la normalizzazione YOLO
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if W == 0 or H == 0:
                print(f"[ERRORE] Dimensioni del video non valide in {video_path}. Salto.")
                continue
                
            print(f"[VIDEO] Dimensione frame: {W}x{H}")


            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Fine video

                image_id = frame_index + 1 
                
                # Lista per contenere le label YOLO per questo frame
                yolo_labels = []

                if has_label and image_id in frame_annotations:
                    boxes = frame_annotations[image_id]
                    
                    for bbox in boxes:
                        x, y, w, h = bbox
                        
                        # Calcolo delle coordinate per il disegno (Opencv - Pixel)
                        x_min, y_min = int(x), int(y)
                        x_max, y_max = int(x + w), int(y + h)
                        
                        # Disegna rettangolo (verde)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, category_name, (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Calcolo delle coordinate per il formato YOLO (Normalizzate)
                        x_center = (x + w / 2) / W
                        y_center = (y + h / 2) / H
                        normalized_w = w / W
                        normalized_h = h / H
                        
                        # Aggiungi la label YOLO alla lista
                        yolo_labels.append(f"{GUN_CLASS_ID} {x_center:.6f} {y_center:.6f} {normalized_w:.6f} {normalized_h:.6f}")
                
                
                # --- Salvataggio ---
                
                # 1. Salva l'immagine
                output_base = os.path.join(output_subdir, f"frame_{frame_index:05d}")
                cv2.imwrite(output_base + ".jpg", frame)
                
                # 2. Salva il file TXT (solo se ci sono label e label.json era presente)
                if yolo_labels:
                    label_output_path = output_base + ".txt"
                    with open(label_output_path, 'w') as f:
                        f.write("\n".join(yolo_labels) + "\n")
                
                # NOTA: Per le cartelle "No_Gun" (has_label=False), yolo_labels è vuota e il file .txt non viene creato.

                frame_index += 1

            cap.release()
            video_count += 1
            print(f"--- Estratti {frame_index} frame in {output_subdir}")
        
        elif has_video or has_label:
            # Stampa un messaggio di debug solo per le cartelle che non sono la radice
            if subdir != root_dir:
                print(f"[DEBUG] Saltata cartella {os.path.relpath(subdir, root_dir)}: Manca {'label.json' if has_video else 'video.mp4'}")


    print("-" * 80)
    print(f"Processo completato! Elaborati {video_count} video totali.")

if __name__ == "__main__":
    # Esegui dalla cartella Gun_Action_Recognition_Dataset
    root_dataset = r"." 
    extract_frames_from_videos(root_dataset)