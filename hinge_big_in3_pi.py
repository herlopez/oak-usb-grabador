import depthai as dai
import cv2
import numpy as np

# Configuración de la pipeline de DepthAI
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1920, 1080)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Parámetros de tu lógica
RECORTADO = (1200, 280, 500, 500)  # x, y, w, h (ajusta según la resolución de la cámara)
ZONA_ALERTA = (150, 150, 300, 100)
MIN_AREA = 1000
MIN_WIDTH = 30
MIN_HEIGHT = 30
MAX_HEIGHT = 250
TIEMPO_PERDIDA = 10
VELOCIDAD_UMBRAL = 2.0
FRAMES_QUIETOS = 5
COLOR_BAJO = np.array([0, 0, 200])
COLOR_ALTO = np.array([180, 50, 255])

next_id = 1
objetos = {}
paused = False
frame_number = 0

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        if not paused or cv2.waitKey(1) & 0xFF == ord('n'):
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                cv2.waitKey(10)
                continue
            frame = in_rgb.getCvFrame()

            # Visualiza el recorte y el ROI sobre el frame completo
            cv2.rectangle(frame, (RECORTADO[0], RECORTADO[1]),
                          (RECORTADO[0]+RECORTADO[2], RECORTADO[1]+RECORTADO[3]),
                          (255, 0, 0), 2)
            cv2.rectangle(frame,
                          (RECORTADO[0]+ZONA_ALERTA[0], RECORTADO[1]+ZONA_ALERTA[1]),
                          (RECORTADO[0]+ZONA_ALERTA[0]+ZONA_ALERTA[2], RECORTADO[1]+ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                          (0, 0, 255), 2)

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
                        and 0.7 < aspect_ratio < 1.5):
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

                # ALERTA inmediata si el centro entra en el ROI
                if not data['alertado'] and zx <= centro[0] <= zx+zw and zy <= centro[1] <= zy+zh:
                    print(f"⚠️ Objeto ID {oid} ha entrado en la zona de alerta")
                    cv2.putText(zona, "⚠️ ALERTA: OBJETO EN ZONA", (rx, ry+rh+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    objetos[oid]['alertado'] = True

            cv2.rectangle(zona, (ZONA_ALERTA[0], ZONA_ALERTA[1]),
                          (ZONA_ALERTA[0]+ZONA_ALERTA[2], ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                          (0, 0, 255), 2)

            cv2.imshow("Zona Recortada", zona)
            cv2.imshow("Máscara Color", mask_color)

        key = cv2.waitKey(10 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('n'):
            paused = True
        elif key == ord('m'):
            frame_number = max(0, frame_number - 2)
            paused = True

# cv2.destroyAllWindows()