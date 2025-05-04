#Para detectar objetos en la salida del horno sobre la banda transportadora.  Testeado con el video de la camara planta101/rpi5

import cv2
import torch
import numpy as np
from sort.sort import Sort
import time

# Cargar modelo YOLOv5 de PyTorch (requiere internet la primera vez)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Abrir archivo de video
cap = cv2.VideoCapture(r'C:\Planta101\rpi5\20250430\video_20250430_095248.mp4')

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Solo un ROI
roi_left = (600, 0, 900, 400)  # ROI definido por (x, y, ancho, alto)

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Detection', 1024, 768)  # Tamaño inicial de ventana

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 5)
frame_count = 0
objetos_intervalo = set()
ultimo_reporte_texto = ""

zoom_out = False
zoom_factor = 0.5  # 50% del tamaño original

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- FILTRO DE COLOR BLANCO ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('WhiteMask', mask_white)  # DEBUG: ver la máscara de blancos

    results = model(frame)
    detections = []

    for det in results.xyxy[0]:
        class_id = int(det[5])
        conf = float(det[4])
        if class_id != 0 and conf > 0.15:  # Detectar cualquier objeto que NO sea persona y confianza > 0.15
            x1, y1, x2, y2 = map(int, det[:4])
            # Refuerzo: solo aceptar si el área es mayormente blanca
            roi_mask = mask_white[y1:y2, x1:x2]
            white_ratio = np.sum(roi_mask == 255) / (roi_mask.size + 1e-6)
            if white_ratio > 0.4:  # Ajusta este umbral según tus objetos
                detections.append([x1, y1, x2, y2, conf])
                # DEBUG: dibuja el bbox en magenta si pasa el filtro de blanco
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(frame, f"white:{white_ratio:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            else:
                # DEBUG: dibuja el bbox en rojo si NO pasa el filtro de blanco
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, f"white:{white_ratio:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

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

        # Verificar si el centroide (cx, cy) está dentro del ROI
        if (roi_left[0] <= cx <= roi_left[0] + roi_left[2] and
            roi_left[1] <= cy <= roi_left[1] + roi_left[3]):
            roi_label = "IN"
            color = (0, 255, 0)  # Verde para objetos dentro
        else:
            roi_label = "OUT"
            color = (128, 128, 128)  # Gris para objetos fuera

        # Dibujar el objeto
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Dibuja solo el ROI izquierdo
    cv2.rectangle(frame, (roi_left[0], roi_left[1]), (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]), (255, 0, 0), 2)

    objetos_intervalo.update(ids_presentes)
    frame_count += 1

    if frame_count >= intervalo:
        ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Objetos únicos en 5s: {len(objetos_intervalo)}"
        print(f"[{ultimo_reporte_texto}] (IDs: {sorted(objetos_intervalo)})")
        objetos_intervalo.clear()
        frame_count = 0

    if ultimo_reporte_texto:
        cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar con o sin zoom
    frame_to_show = frame
    if zoom_out:
        frame_to_show = cv2.resize(frame, (0, 0), fx=zoom_factor, fy=zoom_factor)
    cv2.imshow('Detection', frame_to_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        zoom_out = not zoom_out

cap.release()
cv2.destroyAllWindows()