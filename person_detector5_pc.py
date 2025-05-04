import cv2
import torch
import numpy as np
from sort.sort import Sort
import time
import warnings
import csv
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings("ignore", category=FutureWarning)

# Selección de archivo de video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Selecciona el video",
    filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
)
if not video_path:
    print("No se seleccionó ningún video.")
    exit()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

roi_left = (100, 550, 300, 250)
roi_center = (880, 400, 100, 150)
roi_right = (1200, 300, 300, 200)

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1920, 1080)

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 60)  # 10 segundos para prueba, cambia a 60 para 1 minuto
frame_count = 0

personas_intervalo = set()
ultimo_reporte_texto = ""

# Para estadísticas por minuto
frames_in_minute = 0
roi_left_frames = 0
roi_center_frames = 0
roi_right_frames = 0
person_counts = []
out_roi_frames = 0  # <-- NUEVO: frames con personas fuera de los ROIs

# CSV setup
csv_file = open("person_stats.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Avg"])

zoom_scale = 1.0  # Escala inicial

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    for det in results.xyxy[0]:
        class_id = int(det[5])
        if class_id == 0:  # Persona
            x1, y1, x2, y2 = map(int, det[:4])
            conf = float(det[4])
            detections.append([x1, y1, x2, y2, conf])

    if detections:
        dets_np = np.array(detections)
        tracks = tracker.update(dets_np)
    else:
        tracks = []

    ids_presentes = set()
    roi_left_present = False
    roi_center_present = False
    roi_right_present = False

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        ids_presentes.add(track_id)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Determina en qué ROI está el centroide
        if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
            roi_label = "Left"
            color = (255, 0, 0)
            roi_left_present = True
        elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
            roi_label = "Center"
            color = (0, 255, 0)
            roi_center_present = True
        elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
            roi_label = "Right"
            color = (0, 0, 255)
            roi_right_present = True
        else:
            roi_label = ""
            color = (128, 128, 128)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Dibuja los tres ROIs
    cv2.rectangle(frame, (roi_left[0], roi_left[1]), (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_center[0], roi_center[1]), (roi_center[0] + roi_center[2], roi_center[1] + roi_center[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_right[0], roi_right[1]), (roi_right[0] + roi_right[2], roi_right[1] + roi_right[3]), (0, 0, 255), 2)

    personas_intervalo.update(ids_presentes)
    frame_count += 1

    # Estadísticas por minuto
    frames_in_minute += 1
    person_counts.append(len(ids_presentes))
    if roi_left_present:
        roi_left_frames += 1
    if roi_center_present:
        roi_center_frames += 1
    if roi_right_present:
        roi_right_frames += 1
    # Si hay personas y ninguna está en un ROI, cuenta como fuera de ROI
    if (len(ids_presentes) > 0) and not (roi_left_present or roi_center_present or roi_right_present):
        out_roi_frames += 1

    if frame_count >= intervalo:
        # Porcentaje de tiempo con personas en cada ROI
        pct_left = 100 * roi_left_frames / frames_in_minute
        pct_center = 100 * roi_center_frames / frames_in_minute
        pct_right = 100 * roi_right_frames / frames_in_minute
        pct_out_roi = 100 * out_roi_frames / frames_in_minute  # <-- NUEVO

        avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        reporte = (f"{timestamp} | %ROI_Left: {pct_left:.1f} | %ROI_Center: {pct_center:.1f} | "
                   f"%ROI_Right: {pct_right:.1f} | %Fuera_ROI: {pct_out_roi:.1f} | Ocupacion: {avg_personas}")
        print(reporte)
        csv_writer.writerow([timestamp, f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}",
                             f"{pct_out_roi:.1f}", avg_personas])
        csv_file.flush()

        # Reset stats
        personas_intervalo.clear()
        frame_count = 0
        frames_in_minute = 0
        roi_left_frames = 0
        roi_center_frames = 0
        roi_right_frames = 0
        person_counts = []
        out_roi_frames = 0  # <-- RESET

    # --- ZOOM OUT/IN ---
    display_frame = frame
    if zoom_scale != 1.0:
        h, w = frame.shape[:2]
        display_frame = cv2.resize(frame, (int(w * zoom_scale), int(h * zoom_scale)), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Detection', display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()