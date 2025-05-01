import cv2
import torch
import numpy as np
from sort.sort import Sort
import time

# Cargar modelo YOLOv5 de PyTorch (requiere internet la primera vez)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Abrir archivo de video
cap = cv2.VideoCapture(r'C:\Planta101\rpi7\output_20250430_135017.mp4')

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# ROIs
roi_left = (50, 200, 150, 250)
roi_center = (500, 200, 600, 300)
roi_right = (650, 200, 800, 400)

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1280, 720)

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 5)
frame_count = 0
personas_intervalo = set()
ultimo_reporte_texto = ""

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
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        ids_presentes.add(track_id)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Determina en qué ROI está el centroide
        if roi_left[0] <= cx < roi_left[0] + roi_left[2]:
            roi_label = "Hinge Big"
            color = (255, 0, 0)
        elif roi_center[0] <= cx < roi_center[0] + roi_center[2]:
            roi_label = "Hinge Small"
            color = (0, 255, 0)
        elif roi_right[0] <= cx < roi_right[0] + roi_right[2]:
            roi_label = "Outside"
            color = (0, 0, 255)
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

    if frame_count >= intervalo:
        ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Personas únicas en 5s: {len(personas_intervalo)}"
        print(f"[{ultimo_reporte_texto}] (IDs: {sorted(personas_intervalo)})")
        personas_intervalo.clear()
        frame_count = 0

    if ultimo_reporte_texto:
        cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()