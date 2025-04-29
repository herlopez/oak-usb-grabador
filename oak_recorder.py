import depthai as dai
import cv2
import time
from datetime import datetime, timedelta
import logging
import os

# Configuración del logging
logging.basicConfig(
    filename='/mnt/nvme/grabador.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

VIDEO_DIR = "/mnt/nvme"
MAX_USAGE_BYTES = 800 * 1024 * 1024 * 1024  # 800 GB

def manage_disk_usage(directory, max_usage_bytes):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                files.append((full_path, os.path.getmtime(full_path), os.path.getsize(full_path)))
    files.sort(key=lambda x: x[1])
    total_size = sum(f[2] for f in files)
    while total_size > max_usage_bytes and files:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file[0])
            logging.info(f"Archivo eliminado para liberar espacio: {oldest_file[0]}")
        except Exception as e:
            logging.error(f"No se pudo eliminar {oldest_file[0]}: {e}")
        total_size = sum(f[2] for f in files)

def esperar_hasta_proximo_multiplo_5():
    now = datetime.now()
    minutos = now.minute
    segundos = now.second
    microsegundos = now.microsecond
    # Próximo múltiplo de 5 minutos
    minutos_a_sumar = (5 - (minutos % 5)) % 5
    if minutos_a_sumar == 0 and (segundos > 0 or microsegundos > 0):
        minutos_a_sumar = 5
    proximo = (now + timedelta(minutes=minutos_a_sumar)).replace(second=0, microsecond=0)
    espera = (proximo - now).total_seconds()
    print(f"Esperando {espera:.2f} segundos hasta el próximo múltiplo de 5 minutos...")
    time.sleep(espera)

# Crear la pipeline
pipeline = dai.Pipeline()

# Crear cámara RGB
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(10)

# Crear salida
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    frame_width = 1920
    frame_height = 1080
    fps = 10
    segment_duration = 300  # 5 minutos en segundos

    # Esperar hasta el próximo múltiplo de 5 minutos
    esperar_hasta_proximo_multiplo_5()
    
    while True:

        # Gestionar espacio antes de grabar
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

        now = datetime.now()
        day_folder = now.strftime("%Y%m%d")
        hour_folder = now.strftime("%H")
        output_dir = os.path.join(VIDEO_DIR, day_folder, hour_folder)
        os.makedirs(output_dir, exist_ok=True)

        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        filepath = os.path.join(output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        print(f"Grabando: {filepath}")
        logging.info(f"Inicio de grabación: {filepath}")

        try:
            while True:
                frame = q.get().getCvFrame()
                out.write(frame)
                if time.time() - start_time >= segment_duration:
                    print("Grabación de 5 minutos completada.")
                    logging.info(f"Fin de grabación: {filepath}")
                    break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
        finally:
            out.release()