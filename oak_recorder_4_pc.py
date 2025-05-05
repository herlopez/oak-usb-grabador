import cv2
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import csv
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sort.sort import Sort
LOGGER.setLevel(logging.WARNING)

# Configuración lectura video
VIDEO_PATH = r"C:\Planta101\rpi7\20250505\13\output_20250505_135200.mp4"
print("¿Existe el video?", os.path.exists(VIDEO_PATH))
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de ROIs (imagen original 1920x1080)
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 250, 350, 300)
roi_hinge_orig  = (1400, 380, 500, 200)
original_width = 1920
original_height = 1080

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

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
        int(roi[0] * shape[1] / orig_shape[1]),
        int(roi[1] * shape[0] / orig_shape[0]),
        int(roi[2] * shape[1] / orig_shape[1]),
        int(roi[3] * shape[0] / orig_shape[0])
    )

# Procesamiento de video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("No se pudo abrir el video:", VIDEO_PATH)
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segment_duration = 60  # segundos por segmento
frames_per_segment = int(fps * segment_duration)

segment_idx = 0
frame_idx = 0

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
        frames_in_segment = 0

    # Detección con YOLOv5
    results = model(frame)
    detections = results[0].boxes
    roi_left_present = False
    roi_center_present = False
    roi_right_present = False
    person_count_this_frame = 0

    # Escalar ROIs
    roi_left = escalar_roi(roi_left_orig, frame.shape, (original_width, original_height))
    roi_center = escalar_roi(roi_center_orig, frame.shape, (original_width, original_height))
    roi_right = escalar_roi(roi_right_orig, frame.shape, (original_width, original_height))
    roi_hinge_scaled = escalar_roi(roi_hinge_orig, frame.shape, (original_width, original_height))
    roi_hinge_area = roi_hinge_scaled[2] * roi_hinge_scaled[3]
    objeto_hinge_presente = False

    # --- Dibuja bounding boxes de personas con ID sobre el frame ---
    frame_roi = frame.copy()

    # 1. Recolectar bounding boxes de personas para el tracker
    person_boxes = []
    for box in detections:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
            person_boxes.append([x1, y1, x2, y2, conf])

    # 2. Actualizar el tracker y obtener IDs
    if person_boxes:
        tracks = tracker.update(np.array(person_boxes))
    else:
        tracks = []

    # 3. Dibujar bounding boxes y IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track[:5])
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        # Determinar color y ROI
        if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
            color = (255, 0, 0)
            roi_left_present = True
            roi_label = f"ID:{track_id} Left"
        elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
            color = (0, 255, 0)
            roi_center_present = True
            roi_label = f"ID:{track_id} Center"
        elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
            color = (0, 0, 255)
            roi_right_present = True
            roi_label = f"ID:{track_id} Right"
        else:
            color = (0, 255, 255)
            roi_label = f"ID:{track_id} Fuera ROI"
        cv2.rectangle(frame_roi, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_roi, roi_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        person_count_this_frame += 1

    # Detección de objetos no persona para hinge con entrada por izquierda o arriba
    for box in detections:
        cls = int(box.cls[0])
        if cls != 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            inter_x1 = max(x1, roi_hinge_scaled[0])
            inter_y1 = max(y1, roi_hinge_scaled[1])
            inter_x2 = min(x2, roi_hinge_scaled[0] + roi_hinge_scaled[2])
            inter_y2 = min(y2, roi_hinge_scaled[1] + roi_hinge_scaled[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if roi_hinge_area > 0 and (inter_area / roi_hinge_area) > 0.1:
                # Chequea si el objeto entra desde la izquierda o arriba
                entra_por_izquierda = (x1 < roi_hinge_scaled[0] + 10) and (inter_w > 0)
                entra_por_arriba = (y1 < roi_hinge_scaled[1] + 10) and (inter_h > 0)
                if entra_por_izquierda or entra_por_arriba:
                    objeto_hinge_presente = True

    if objeto_hinge_presente and not objeto_hinge_presente_anterior:
        objeto_hinge_count += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Event HINGE detected (Hinge in ROI, entrada por izq/arriba)")

    objeto_hinge_presente_anterior = objeto_hinge_presente

    # Estadísticas de personas y ROIs
    person_counts.append(person_count_this_frame)
    if roi_left_present:
        roi_left_frames += 1
    if roi_center_present:
        roi_center_frames += 1
    if roi_right_present:
        roi_right_frames += 1
    if (person_count_this_frame > 0) and not (roi_left_present or roi_center_present or roi_right_present):
        out_roi_frames += 1

    # Visualizar el frame con ROIs en cada iteración
    cv2.rectangle(frame_roi, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
    cv2.rectangle(frame_roi, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
    cv2.rectangle(frame_roi, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
    cv2.rectangle(frame_roi, (roi_hinge_scaled[0], roi_hinge_scaled[1]), (roi_hinge_scaled[0]+roi_hinge_scaled[2], roi_hinge_scaled[1]+roi_hinge_scaled[3]), (0,128,255), 2)
    cv2.imshow("Video con ROIs", frame_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Procesamiento interrumpido por el usuario.")
        break

    frames_in_segment += 1
    if (frame_idx + 1) % frames_per_segment == 0 or not ret:

        # Guardar CSV
        pct_left = 100 * roi_left_frames / frames_in_segment if frames_in_segment else 0
        pct_center = 100 * roi_center_frames / frames_in_segment if frames_in_segment else 0
        pct_right = 100 * roi_right_frames / frames_in_segment if frames_in_segment else 0
        pct_out_roi = 100 * out_roi_frames / frames_in_segment if frames_in_segment else 0
        avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

        fecha = now.strftime('%Y-%m-%d')
        hora = now.strftime('%H')
        minuto = now.strftime('%M')

        csv_writer.writerow([
            fecha, hora, minuto,
            f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}", f"{pct_out_roi:.1f}", avg_personas,
            filename, "analisis_video_pc.py", objeto_hinge_count
        ])
        print(
            f"Fecha: {fecha}, Hora: {hora}, Minuto: {minuto}, "
            f"%ROI_Left: {pct_left:.1f}, %ROI_Center: {pct_center:.1f}, %ROI_Right: {pct_right:.1f}, "
            f"%Fuera_ROI: {pct_out_roi:.1f}, Personas: {avg_personas}, "
            f"VideoFile: {filename}, Script: analisis_video_pc.py, objeto_hinge: {objeto_hinge_count}"
        )
        csv_file.flush()

    frame_idx += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("¡Procesamiento terminado!")