import cv2
import numpy as np
from sort.sort import Sort
import time

# Abrir archivo de video
cap = cv2.VideoCapture(r'C:\Planta101\rpi5\20250430\video_20250430_095248.mp4')

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# ROI donde quieres contar objetos
roi_left = (600, 0, 900, 400)  # x, y, ancho, alto

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 5)
frame_count = 0
objetos_intervalo = set()
ultimo_reporte_texto = ""

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV y crear máscara de blancos
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 100, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('WhiteMask', mask_white)  # DEBUG

    # Encontrar contornos en la máscara blanca
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Ajusta según el tamaño real de las cajas
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, x + w, y + h, 0.99])  # Simula alta confianza
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)  # Azul claro

    # Seguimiento con SORT
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

        # Verificar si el centroide está en el ROI
        if (roi_left[0] <= cx <= roi_left[0] + roi_left[2] and
            roi_left[1] <= cy <= roi_left[1] + roi_left[3]):
            roi_label = "IN"
            color = (0, 255, 0)
        else:
            roi_label = "OUT"
            color = (128, 128, 128)

        # Dibujar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Dibujar ROI
    cv2.rectangle(frame, (roi_left[0], roi_left[1]),
                  (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]),
                  (255, 0, 0), 2)

    objetos_intervalo.update(ids_presentes)
    frame_count += 1

    if frame_count >= intervalo:
        ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Objetos únicos en 5s: {len(objetos_intervalo)}"
        print(f"[{ultimo_reporte_texto}] (IDs: {sorted(objetos_intervalo)})")
        objetos_intervalo.clear()
        frame_count = 0

    if ultimo_reporte_texto:
        cv2.putText(frame, ultimo_reporte_texto, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
