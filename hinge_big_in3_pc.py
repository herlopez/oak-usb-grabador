import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Planta101\rpi7\20250429\14\output_20250429_145501.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

RECORTADO = (1200, 280, 500, 500)  # x, y, w, h
ZONA_ALERTA = (150, 150, 300, 100)  # x, y, w, h


# --- Mostrar solo una foto con el recorte y el ROI ---
ret, frame = cap.read()
if ret:
    # Dibuja el recorte (azul)
    cv2.rectangle(frame, (RECORTADO[0], RECORTADO[1]),
                  (RECORTADO[0]+RECORTADO[2], RECORTADO[1]+RECORTADO[3]),
                  (255, 0, 0), 2)
    # Dibuja el ROI (rojo) relativo al recorte
    cv2.rectangle(frame,
                  (RECORTADO[0]+ZONA_ALERTA[0], RECORTADO[1]+ZONA_ALERTA[1]),
                  (RECORTADO[0]+ZONA_ALERTA[0]+ZONA_ALERTA[2], RECORTADO[1]+ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                  (0, 0, 255), 2)
    cv2.imshow("Frame original con recorte y ROI", frame)
    cv2.waitKey(0)  # Espera a que presiones una tecla
    cv2.destroyWindow("Frame original con recorte y ROI")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia el video para el procesamiento normal
else:
    print("No se pudo leer el frame del video.")




MIN_AREA = 1000
MIN_WIDTH = 30
MIN_HEIGHT = 30
MAX_HEIGHT = 250  # Personas probablemente son más altas
TIEMPO_PERDIDA = 10
VELOCIDAD_UMBRAL = 2.0
FRAMES_QUIETOS = 5

# Máscara para blanco en HSV
COLOR_BAJO = np.array([0, 0, 200])
COLOR_ALTO = np.array([180, 50, 255])

next_id = 1
objetos = {}

paused = False
frame_number = 0

while True:
    if not paused or cv2.waitKey(1) & 0xFF == ord('n'):
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        x, y, w, h = RECORTADO
        zona = frame[y:y+h, x:x+w]

        hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, COLOR_BAJO, COLOR_ALTO)
        mask_color = cv2.GaussianBlur(mask_color, (7, 7), 0)

        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detectados = []

        for c in contours:
            area = cv2.contourArea(c)
            (rx, ry, rw, rh) = cv2.boundingRect(c)
            aspect_ratio = rw / float(rh)
            if (area > MIN_AREA and rw > MIN_WIDTH and rh > MIN_HEIGHT and rh < MAX_HEIGHT
                    and 0.7 < aspect_ratio < 1.5):  # aspecto rectangular
                centro = (rx + rw // 2, ry + rh // 2)
                detectados.append({'centro': centro, 'bbox': (rx, ry, rw, rh)})

        usados = set()
        for obj in detectados:
            centro = obj['centro']
            asignado = False
            for oid, data in objetos.items():
                if np.linalg.norm(np.array(centro) - np.array(data['centro'])) < 50:
                    objetos[oid]['centro'] = centro
                    objetos[oid]['trayectoria'].append(centro)
                    objetos[oid]['frames'] += 1
                    objetos[oid]['no_visto'] = 0
                    obj['id'] = oid
                    usados.add(oid)
                    asignado = True
                    break
            if not asignado:
                objetos[next_id] = {
                    'centro': centro,
                    'trayectoria': [centro],
                    'frames': 1,
                    'no_visto': 0,
                    'alertado': False
                }
                obj['id'] = next_id
                usados.add(next_id)
                next_id += 1

        for oid in list(objetos.keys()):
            if oid not in usados:
                objetos[oid]['no_visto'] += 1

        objetos = {oid: data for oid, data in objetos.items() if data['no_visto'] <= TIEMPO_PERDIDA}

        for obj in detectados:
            rx, ry, rw, rh = obj['bbox']
            centro = obj['centro']
            oid = obj['id']
            data = objetos[oid]

            cv2.rectangle(zona, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.putText(zona, f"ID {oid}", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(zona, centro, 5, (0,0,255), -1)

            zx, zy, zw, zh = ZONA_ALERTA
            trayectoria = data['trayectoria']
            en_zona = zx <= centro[0] <= zx+zw and zy <= centro[1] <= zy+zh

            if not data['alertado'] and en_zona and data['frames'] >= FRAMES_QUIETOS:
                ultimos = trayectoria[-FRAMES_QUIETOS:]
                velocidades = [
                    np.linalg.norm(np.array(ultimos[i]) - np.array(ultimos[i-1]))
                    for i in range(1, len(ultimos))
                ]
                velocidad_prom = sum(velocidades) / len(velocidades)

                if velocidad_prom < VELOCIDAD_UMBRAL:
                    print(f"⚠️ Objeto ID {oid} COLOCADO en la zona de alerta")
                    cv2.putText(zona, "⚠️ ALERTA: OBJETO DETENIDO", (rx, ry+rh+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    objetos[oid]['alertado'] = True

        cv2.rectangle(zona, (ZONA_ALERTA[0], ZONA_ALERTA[1]),
                      (ZONA_ALERTA[0]+ZONA_ALERTA[2], ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                      (0, 0, 255), 2)

        cv2.imshow("Zona Recortada", zona)
        cv2.imshow("Máscara Color", mask_color)

    key = cv2.waitKey(0 if paused else 1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('n'):
        paused = True
    elif key == ord('m'):
        frame_number = max(0, frame_number - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        paused = True

cap.release()
cv2.destroyAllWindows()
