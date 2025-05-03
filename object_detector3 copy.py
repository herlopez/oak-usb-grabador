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
roi_left = (1100, 0, 400, 350)  # x, y, ancho, alto

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 5)
frame_count = 0
objetos_intervalo = set()
ultimo_reporte_texto = ""

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

prev_positions = {}  # Guarda la posición X anterior de cada ID
alertados = set()    # IDs ya alertados

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV y crear máscara de blancos
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 100, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # --- Operación morfológica para separar objetos pegados ---
    kernel = np.ones((5, 5), np.uint8)
    mask_white_sep = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('WhiteMaskSep', mask_white_sep)  # DEBUG

    # Encontrar contornos en la máscara procesada
    contours, _ = cv2.findContours(mask_white_sep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 20 and h >= 20:  # Solo objetos de al menos 20x20 píxeles
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

        # Detectar movimiento de derecha a izquierda
        moving_left = False
        if track_id in prev_positions:
            if cx < prev_positions[track_id]:
                moving_left = True
        prev_positions[track_id] = cx

        # Verificar si el centroide está en el ROI y se mueve a la izquierda
        if (roi_left[0] <= cx <= roi_left[0] + roi_left[2] and
            roi_left[1] <= cy <= roi_left[1] + roi_left[3] and
            moving_left):
            roi_label = "IN"
            color = (0, 255, 0)
            if track_id not in alertados:
                print(f"ALERTA: Objeto ID {track_id} detectado en ROI a las {time.strftime('%H:%M:%S')}")
                alertados.add(track_id)
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
        # print(f"[{ultimo_reporte_texto}] (IDs: {sorted(objetos_intervalo)})")
        objetos_intervalo.clear()
        frame_count = 0

    if ultimo_reporte_texto:
        cv2.putText(frame, ultimo_reporte_texto, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Control de velocidad en tiempo real
    delay = int(1000 / fps) if fps > 0 else 33
    cv2.imshow('Detection', frame)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()