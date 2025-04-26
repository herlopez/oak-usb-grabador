import depthai as dai
import cv2
import time
from datetime import datetime

# Crear la pipeline
pipeline = dai.Pipeline()

# Crear cámara RGB
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)

# Crear salida
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

# Ejecutar el dispositivo
with dai.Device(pipeline) as device:
    # Obtener la cola de salida
    q = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    # Configuración del grabador de video
    frame_width = 1920  # Ancho del frame (1080p)
    frame_height = 1080  # Alto del frame (1080p)
    fps = 15  # FPS de la cámara
    segment_duration = 60  # Duración del video en segundos (1 minuto)

    while True:
        # Generar nombre de archivo con fecha y hora
        now = datetime.now()
        filename = now.strftime("/mnt/nvme/output_%Y%m%d_%H%M%S.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        print(f"Grabando: {filename}")

        try:
            while True:
                frame = q.get().getCvFrame()
                out.write(frame)

                # Mostrar el frame (opcional)
                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break

                if time.time() - start_time >= segment_duration:
                    print("Grabación de 1 minuto completada.")
                    break
        finally:
            out.release()
            # cv2.destroyAllWindows()  # Solo si usas imshow

        # Aquí el ciclo continúa y se crea un nuevo archivo con nuevo nombre