import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import csv
from ultralytics import YOLO

# Configuración de carpetas
VIDEO_PATH = "video_de_prueba.mp4"  # Cambia por tu video
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de ROIs
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 250, 350, 300)
roi_hinge       = (1400, 380, 500, 200)
original_width = 1920
original_height = 1080

# Cargar modelo YOLOv5 (puedes usar yolov5n, yolov5s, etc.)
model = YOLO("yolov5n.pt")  # Descarga automática

# CSV
csv_path = os.path.join(OUTPUT_DIR, "person_stats.csv")
new_csv = not os.path.exists(csv_path)
csv_file = open(csv_path, "a", newline="")
csv_writer = csv.writer(csv_file)
if new_csv:
    csv_writer.writerow([
        "Fecha", "Hora", "Minuto", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Personas", "VideoFile", "Script", "objeto_hinge"
    ])

# Función para escalar ROIs
def escalar_roi(roi, shape, orig_shape):
    return (
        int(roi[0] * shape[1] / orig_shape[0]),
        int(roi[1] * shape[0] / orig_shape[1]),
        int(roi[2] * shape[1] / orig_shape[0]),
        int(roi[3] * shape[0] / orig_shape[1])
    )

# Procesamiento de video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segment_duration = 60  # segundos por segmento
frames_per_segment = int(fps * segment_duration)

segment_idx = 0
frame_idx = 0

roi_left_frames = 0
roi_center_frames = 0
roi_right_frames = 0
out_roi_frames = 0
person_counts = []
objeto_hinge_count = 0
objeto_hinge_presente_anterior = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frames_per_segment == 0:
        # Nuevo segmento
        now = datetime.now()
        day_folder = now.strftime("%Y%m%d")
        hour_folder = now.strftime("%H")
        output_dir = os.path.join(OUTPUT_DIR, day_folder, hour_folder)
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        # Reinicia estadísticas
        roi_left_frames = 0
        roi_center_frames = 0
        roi_right_frames = 0
        out_roi_frames = 0
        person_counts = []
        objeto_hinge_count = 0
        objeto_hinge_presente_anterior = False
        segment_idx += 1

    # Detección con YOLOv5
    results = model(frame)
    detections = results[0].boxes
    ids_presentes = set()
    roi_left_present = False
    roi_center_present = False
    roi_right_present = False

    # Escalar ROIs
    roi_left = escalar_roi(roi_left_orig, frame.shape, (original_width, original_height))
    roi_center = escalar_roi(roi_center_orig, frame.shape, (original_width, original_height))
    roi_right = escalar_roi(roi_right_orig, frame.shape, (original_width, original_height))
    roi_hinge_scaled = escalar_roi(roi_hinge, frame.shape, (original_width, original_height))
    roi_hinge_area = roi_hinge_scaled[2] * roi_hinge_scaled[3]

    # --- Evento objeto_hinge (solo objetos que NO sean persona, y solo cuenta una vez por evento) ---
    objeto_hinge_presente = False

    for box in detections:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Personas = 0 en COCO
        if cls == 0:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
                roi_left_present = True
            elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
                roi_center_present = True
            elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
                roi_right_present = True
        else:
            # Solo objetos que NO sean persona
            inter_x1 = max(x1, roi_hinge_scaled[0])
            inter_y1 = max(y1, roi_hinge_scaled[1])
            inter_x2 = min(x2, roi_hinge_scaled[0] + roi_hinge_scaled[2])
            inter_y2 = min(y2, roi_hinge_scaled[1] + roi_hinge_scaled[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if roi_hinge_area > 0 and (inter_area / roi_hinge_area) > 0.4:
                objeto_hinge_presente = True

    if objeto_hinge_presente and not objeto_hinge_presente_anterior:
        objeto_hinge_count += 1
    objeto_hinge_presente_anterior = objeto_hinge_presente

    # Estadísticas
    person_counts.append(len([box for box in detections if int(box.cls[0]) == 0]))
    if roi_left_present:
        roi_left_frames += 1
    if roi_center_present:
        roi_center_frames += 1
    if roi_right_present:
        roi_right_frames += 1
    if (len([box for box in detections if int(box.cls[0]) == 0]) > 0) and not (roi_left_present or roi_center_present or roi_right_present):
        out_roi_frames += 1

    # Guardar imágenes al final del segmento
    if (frame_idx + 1) % frames_per_segment == 0 or not ret:
        # Guardar imagen original
        img_original_path = os.path.join(img_dir, filename.replace('.mp4', f'_frame{frame_idx}_original.jpg'))
        cv2.imwrite(img_original_path, frame)
        # Guardar imagen con ROIs
        frame_roi = frame.copy()
        cv2.rectangle(frame_roi, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
        cv2.rectangle(frame_roi, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
        cv2.rectangle(frame_roi, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
        cv2.rectangle(frame_roi, (roi_hinge_scaled[0], roi_hinge_scaled[1]), (roi_hinge_scaled[0]+roi_hinge_scaled[2], roi_hinge_scaled[1]+roi_hinge_scaled[3]), (0,128,255), 2)
        img_roi_path = os.path.join(img_dir, filename.replace('.mp4', f'_frame{frame_idx}_roi.jpg'))
        cv2.imwrite(img_roi_path, frame_roi)

        # Guardar CSV
        pct_left = 100 * roi_left_frames / frames_per_segment if frames_per_segment else 0
        pct_center = 100 * roi_center_frames / frames_per_segment if frames_per_segment else 0
        pct_right = 100 * roi_right_frames / frames_per_segment if frames_per_segment else 0
        pct_out_roi = 100 * out_roi_frames / frames_per_segment if frames_per_segment else 0
        avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

        fecha = now.strftime('%Y-%m-%d')
        hora = now.strftime('%H')
        minuto = now.strftime('%M')

        csv_writer.writerow([
            fecha, hora, minuto,
            f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}", f"{pct_out_roi:.1f}", avg_personas,
            filename, "analisis_video_pc.py", objeto_hinge_count
        ])
        csv_file.flush()

    frame_idx += 1

cap.release()
csv_file.close()
print("¡Procesamiento terminado!")