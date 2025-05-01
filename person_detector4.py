import cv2
import torch
import numpy as np
from sort.sort import Sort
import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(r'C:\Planta101\rpi7\output_20250430_135017.mp4')

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Leer el primer frame para definir los ROIs
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el primer frame.")
    exit()

height, width = frame.shape[:2]
roi_width = width // 3

# ROI izquierda
roi_left = (150, 500, 250, 250)
# ROI centro
roi_center = (800, 400, 200, 200)
# ROI derecha
roi_right = (1100, 300, 300, 300)

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Procesar el video desde el primer frame leído
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1280, 720)

# Parámetros para el reporte cada 5 segundos
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
        if model.names[class_id] == 'person':
            x1, y1, x2, y2, conf = map(float, det[:5])
            detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:
        dets = np.array(detections)
        tracks = tracker.update(dets)
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

    # Acumula los IDs vistos en este intervalo
    personas_intervalo.update(ids_presentes)
    frame_count += 1

    # Reporte cada 5 segundos
    if frame_count >= intervalo:
        ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Personas únicas en 5s: {len(personas_intervalo)}"
        print(f"[{ultimo_reporte_texto}] (IDs: {sorted(personas_intervalo)})")
        personas_intervalo.clear()
        frame_count = 0

    # Muestra el reporte en la ventana
    if ultimo_reporte_texto:
        cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Detection', frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 83 or key == ord('d'):  # Flecha derecha o 'd'
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current + 30)
    elif key == ord('+'):
        cv2.resizeWindow('Detection', 1920, 1080)
    elif key == ord('-'):
        cv2.resizeWindow('Detection', 640, 360)

cap.release()
cv2.destroyAllWindows()
