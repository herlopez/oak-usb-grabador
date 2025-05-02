import cv2
import depthai as dai

# Paso 1: Crear pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(1920, 1080)
cam_rgb.setInterleaved(False)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.video.link(xout.input)

# Paso 2: Zona que quieres recortar del video (por ejemplo, centro)
RECORTADO = (1000, 600, 800, 400)  # x, y, ancho, alto
ZONA_ALERTA = (500, 200, 100, 100)  # dentro del recorte

trayectoria = []  # memoria del movimiento del objeto

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()

        # Paso 3: Recortar la zona que quieres analizar
        x, y, w, h = RECORTADO
        zona = frame[y:y+h, x:x+w]

        # Paso 4: Buscar contornos (rectángulos)
        gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.contourArea(c) > 500:  # rectángulo grande
                (rx, ry, rw, rh) = cv2.boundingRect(c)
                centro = (rx + rw//2, ry + rh//2)
                trayectoria.append(centro)

                # Dibujar
                cv2.rectangle(zona, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

                # Paso 5: Verificar si el centro está dentro de zona de alerta
                zx, zy, zw, zh = ZONA_ALERTA
                if zx <= centro[0] <= zx+zw and zy <= centro[1] <= zy+zh:
                    print("⚠️ Objeto ha entrado en la zona de alerta")

        # Visualización
        cv2.rectangle(zona, (ZONA_ALERTA[0], ZONA_ALERTA[1]),
                      (ZONA_ALERTA[0]+ZONA_ALERTA[2], ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                      (0, 0, 255), 2)

        cv2.imshow("Zona Recortada", zona)
        if cv2.waitKey(1) == ord('q'):
            break
