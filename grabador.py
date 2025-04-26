import depthai as dai
import cv2
import time
from datetime import datetime
import logging
import os

# Configuración del logging
logging.basicConfig(
    filename='/mnt/nvme/grabador.log',  # Cambia la ruta si lo deseas
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

VIDEO_DIR = "/mnt/nvme"
MAX_USAGE_BYTES = 800 * 1024 * 1024 * 1024  # 800 GB

def manage_disk_usage(directory, max_usage_bytes):
    """Elimina los archivos mp4 más antiguos hasta estar por debajo del límite de espacio."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mp4")]
    files = [(f, os.path.getmtime(f), os.path.getsize(f)) for f in files]
    files.sort(key=lambda x: x[1])  # Ordenar por fecha de modificación (más antiguo primero)
    total_size = sum(f[2] for f in files)
    while total_size > max_usage_bytes and files:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file[0])
            logging.info(f"Archivo eliminado para liberar espacio: {oldest_file[0]}")
        except Exception as e:
            logging.error(f"No se pudo eliminar {oldest_file[0]}: {e}")
        total_size = sum(f[2] for f in files)

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
        # Gestionar espacio antes de grabar
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

        # Generar nombre de archivo con fecha y hora
        now = datetime.now()
        filename = now.strftime(f"{VIDEO_DIR}/output_%Y%m%d_%H%M%S.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        print(f"Grabando: {filename}")
        logging.info(f"Inicio de grabación: {filename}")

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
                    logging.info(f"Fin de grabación: {filename}")
                    break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
        finally:
            out.release()
            # cv2.destroyAllWindows()  # Solo si usas imshow

        # El ciclo continúa y se crea un nuevo archivo con nuevo nombre