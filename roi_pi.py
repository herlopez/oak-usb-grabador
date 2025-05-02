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
RECORTADO = (1200, 280, 500, 500)  # x, y, w, h
ZONA_ALERTA = (150, 150, 300, 100)  # x, y, w, h

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
    in_rgb = q_rgb.get()
    frame = in_rgb.getCvFrame()

    # Dibuja el recorte (azul)
    cv2.rectangle(frame, (RECORTADO[0], RECORTADO[1]),
                  (RECORTADO[0]+RECORTADO[2], RECORTADO[1]+RECORTADO[3]),
                  (255, 0, 0), 2)
    # Dibuja el ROI (rojo) relativo al recorte
    cv2.rectangle(frame,
                  (RECORTADO[0]+ZONA_ALERTA[0], RECORTADO[1]+ZONA_ALERTA[1]),
                  (RECORTADO[0]+ZONA_ALERTA[0]+ZONA_ALERTA[2], RECORTADO[1]+ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                  (0, 0, 255), 2)

    cv2.imshow("Frame con recorte y ROI", frame)
    cv2.waitKey(0)  # Espera a que presiones una tecla
    cv2.destroyAllWindows()