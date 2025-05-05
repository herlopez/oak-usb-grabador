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
VIDEO_PATH = r"C:\Planta101\rpi7\20250505\13\output_20250505_134000.mp4"
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
        int(roi[0] * shape[1] / orig_shape[0]),  # x
        int(roi[1] * shape[0] / orig_shape[1]),  # y
        int(roi[2] * shape[1] / orig_shape[0]),  # w
        int(roi[3] * shape[0] / orig_shape[1])   # h
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

    # --- Detección del hinge usando detections de YOLO ---
    HINGE_CONFIDENCE = 0.2  # Puedes ajustar este valor

    hinge_detected_yolo = False
    hinge_bbox = None  # <-- Guarda el bounding box del hinge
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
        if cls != 0 and conf > HINGE_CONFIDENCE:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            if (roi_hinge_scaled[0] <= cx < roi_hinge_scaled[0] + roi_hinge_scaled[2] and
                roi_hinge_scaled[1] <= cy < roi_hinge_scaled[1] + roi_hinge_scaled[3]):
                hinge_detected_yolo = True
                hinge_bbox = (x1, y1, x2, y2)  # <-- Guarda el bbox

    # --- Detección del hinge por figura y contraste ---
    HINGE_BRIGHTNESS_THRESHOLD = 180
    hinge_roi = frame[roi_hinge_scaled[1]:roi_hinge_scaled[1]+roi_hinge_scaled[3],
                    roi_hinge_scaled[0]:roi_hinge_scaled[0]+roi_hinge_scaled[2]]
    hinge_gray = cv2.cvtColor(hinge_roi, cv2.COLOR_BGR2GRAY)
    _, hinge_bin = cv2.threshold(hinge_gray, HINGE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(hinge_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hinge_detected_shape = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0.03 * hinge_bin.shape[0] * hinge_bin.shape[1]:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                # Contraste local
                mask = np.zeros_like(hinge_gray)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                mean_in = cv2.mean(hinge_gray, mask=mask)[0]
                mean_out = cv2.mean(hinge_gray, mask=cv2.bitwise_not(mask))[0]
                if abs(mean_in - mean_out) > 30:  # Contraste suficiente
                    hinge_detected_shape = True
                    # Calcula el bbox absoluto del contorno detectado
                    x, y, w, h = cv2.boundingRect(approx)
                    x1 = roi_hinge_scaled[0] + x
                    y1 = roi_hinge_scaled[1] + y
                    x2 = x1 + w
                    y2 = y1 + h
                    hinge_bbox = (x1, y1, x2, y2)
                    break

    # --- Combinación de ambos métodos ---
    hinge_detected = hinge_detected_yolo or hinge_detected_shape

    if hinge_detected and hinge_bbox:
        cv2.rectangle(frame_roi, (hinge_bbox[0], hinge_bbox[1]), (hinge_bbox[2], hinge_bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame_roi, "HINGE", (hinge_bbox[0], hinge_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if hinge_detected and not objeto_hinge_presente_anterior:
        objeto_hinge_count += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Event HINGE detected! (ID: {objeto_hinge_count})")
    objeto_hinge_presente_anterior = hinge_detected

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

    # Visualizar el frame con ROIs y centro del hinge
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