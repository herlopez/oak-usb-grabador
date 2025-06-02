# prueba base estable!
#  Graba archivo de video y lo analiza para contar personas en 3 ROIs
#  y fuera de ellas. Guarda estadísticas en CSV y elimina archivos viejos
#  para mantener el uso del disco por debajo de 800 GB.
#  Requiere la librería SORT para el seguimiento de objetos.
#  Se recomienda usar un SSD NVMe para evitar problemas de escritura.

import depthai as dai
from dotenv import load_dotenv
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter
import logging
import os
from datetime import datetime, timedelta
from collections import deque
import sql_logger
import csv

load_dotenv()
SCRIPT_NAME = os.path.basename(__file__)
DEVICE_NAME = os.getenv("DEVICE_NAME")
DATABASE_FILE = os.getenv("DATABASE_FILE")
TABLE_NAME = os.getenv("TABLE_NAME")
DEBUGGER = os.getenv("DEBUGGER", "1").lower() in ("1", "true", "yes")

# Archivo de log de errores
LOGGER_LOG = os.getenv("LOGGER_LOG")

# Configuración del logging
logging.basicConfig(
    filename=LOGGER_LOG,
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

def esperar_hasta_proximo_multiplo(minuto_multiplo):
    now = datetime.now()
    minutos = now.minute
    segundos = now.second
    microsegundos = now.microsecond
    minutos_a_sumar = (minuto_multiplo - (minutos % minuto_multiplo)) % minuto_multiplo
    if minutos_a_sumar == 0 and (segundos > 0 or microsegundos > 0):
        minutos_a_sumar = minuto_multiplo
    proximo = (now + timedelta(minutes=minutos_a_sumar)).replace(second=0, microsecond=0)
    espera = (proximo - now).total_seconds()
    if DEBUGGER: print(f"Esperando {espera:.2f} segundos hasta el próximo múltiplo de {minuto_multiplo} minutos...")
    time.sleep(espera)

# Función para escalar ROIs
def escalar_roi(roi, shape, orig_shape):
    return (
        int(roi[0] * shape[1] / orig_shape[0]),  # x
        int(roi[1] * shape[0] / orig_shape[1]),  # y
        int(roi[2] * shape[1] / orig_shape[0]),  # w
        int(roi[3] * shape[0] / orig_shape[1])   # h
    )

# --- Configuración de ROIs y pipeline ---
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 250, 350, 300)
original_width = 1920
original_height = 1080

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

model_path = blobconverter.from_zoo(
    name="yolov5n_coco_416x416",
    zoo_type="depthai",
    shaves=6
)

pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# XLinkOut para la imagen original de la cámara (1080p)
xout_cam = pipeline.createXLinkOut()
xout_cam.setStreamName("cam")
cam_rgb.video.link(xout_cam.input)

# Nodo manip para 416x416
manip = pipeline.createImageManip()
manip.initialConfig.setResize(416, 416)
manip.initialConfig.setKeepAspectRatio(False)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.setMaxOutputFrameSize(416 * 416 * 3)
cam_rgb.video.link(manip.inputImage)

# mono_left = pipeline.createMonoCamera()
# mono_right = pipeline.createMonoCamera()
# stereo = pipeline.createStereoDepth()
# mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
# mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_left.out.link(stereo.left)
# mono_right.out.link(stereo.right)

detection_nn = pipeline.createYoloDetectionNetwork()

# Un valor más bajo (ej. 0.3 o 0.4) puede detectar más personas, 
# pero también podría generar más falsos positivos. Encuentra un balance para tu escenario.
detection_nn.setConfidenceThreshold(0.25)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setIouThreshold(0.45)
detection_nn.setBlobPath(model_path)
detection_nn.input.setBlocking(False)
detection_nn.input.setQueueSize(1)
detection_nn.setAnchors([
    10,13, 16,30, 33,23,
    30,61, 62,45, 59,119,
    116,90, 156,198, 373,326
])
detection_nn.setAnchorMasks({
    "side52": [0,1,2],
    "side26": [3,4,5],
    "side13": [6,7,8],
})

manip.out.link(detection_nn.input)

xout_nn = pipeline.createXLinkOut()
xout_manip = pipeline.createXLinkOut()  # Para obtener el frame 416x416
xout_nn.setStreamName("detections")
xout_manip.setStreamName("manip")

manip.out.link(xout_manip.input)
detection_nn.out.link(xout_nn.input)

# --- Grabación segmentada ---
MINUTO_MULTIPLO = 1  # Cambia este valor para grabar cada X minutos
fps = 10
segment_duration = 60 * MINUTO_MULTIPLO

# --- Configuración de directorios y CSV ---
now = datetime.now()
current_day = now.strftime("%Y%m%d")

day_folder = now.strftime("%Y%m%d")
csv_path = os.path.join(VIDEO_DIR, day_folder, f"{day_folder}_stats.csv")
os.makedirs(os.path.join(VIDEO_DIR, day_folder), exist_ok=True)

# Initialize db
ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
conn_test = sql_logger.create_connection(DATABASE_FILE)
if conn_test == -1:
    logging.error("Could not connect to the database. Logging to database will be disabled.")
    exit(1)
else:
    # Reed if table exists
    if not sql_logger.table_exists(conn_test, TABLE_NAME):
        # Create table if it doesn't exist
        if sql_logger.create_table(conn_test, TABLE_NAME) == -1:
            logging.error("Could not create event_log table in the database. Logging to database will be disabled.")
            exit(1)
    event_id = sql_logger.insert_event(conn_test, TABLE_NAME, (ts_str, DEVICE_NAME, SCRIPT_NAME, "INFO", 0,0,0,0,0,0,"", "START"))
    if event_id == -1:
        logging.warning("Failed to write initialization event to the database.")


# Esperar hasta que se detecte un dispositivo
max_retries = 10
for i in range(max_retries):
    devices = dai.Device.getAllConnectedDevices()
    if len(devices) > 0:
        break
    print(f"No OAK device found. Retry {i+1}/{max_retries}")
    time.sleep(2)
else:
    raise RuntimeError("No OAK device found after multiple attempts.")

with dai.Device(pipeline) as device:
    cam_queue = device.getOutputQueue("cam", maxSize=4, blocking=False)         # 1080p original
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    manip_queue = device.getOutputQueue("manip", maxSize=4, blocking=False)     # 416x416

    ultimo_minuto_segmento = None
    now = datetime.now()

    day_folder = now.strftime("%Y%m%d")
    hour_folder = now.strftime("%H")
    output_dir = os.path.join(VIDEO_DIR, day_folder, hour_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    if day_folder != current_day:
        # Día nuevo: cierra el CSV anterior y abre uno nuevo
        event_id = sql_logger.insert_event(conn_test, TABLE_NAME, (ts_str, DEVICE_NAME, SCRIPT_NAME, "INFO", 0,0,0,0,0,0,"", "NEW DAY"))
        if event_id == -1:
            logging.warning("Failed to write initialization event to the database.")
    
    # Registro de arranque del programa
    timestamp_1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    event_id = sql_logger.insert_event(conn_test, TABLE_NAME, (ts_str, DEVICE_NAME, SCRIPT_NAME, "INFO", 0,0,0,0,0,0,"", "START"))
    if event_id == -1:
        logging.warning("Failed to write initialization event to the database.")

    while True:
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

        now = datetime.now()
        while ultimo_minuto_segmento == now.minute:
            esperar_hasta_proximo_multiplo(MINUTO_MULTIPLO)
            now = datetime.now()
        ultimo_minuto_segmento = now.minute

        day_folder = now.strftime("%Y%m%d")
        hour_folder = now.strftime("%H")
        output_dir = os.path.join(VIDEO_DIR, day_folder, hour_folder)
        os.makedirs(output_dir, exist_ok=True)

        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        filepath = os.path.join(output_dir, filename)

        # Espera el primer frame para obtener el tamaño real
        in_cam = cam_queue.get()
        in_detections = detections_queue.get()
        frame_1080 = in_cam.getCvFrame()    # 1080p del stream original

        # # Guardar imagen original 1080p
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        img_original_path = os.path.join(img_dir, filename.replace('.mp4', '_1080p.jpg'))
        cv2.imwrite(img_original_path, frame_1080)

        # 1080p con ROIs
        frame_1080_roi = frame_1080.copy()
        roi_left = escalar_roi(roi_left_orig, frame_1080.shape, (original_width, original_height))
        roi_center = escalar_roi(roi_center_orig, frame_1080.shape, (original_width, original_height))
        roi_right = escalar_roi(roi_right_orig, frame_1080.shape, (original_width, original_height))
        cv2.rectangle(frame_1080_roi, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
        cv2.rectangle(frame_1080_roi, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
        cv2.rectangle(frame_1080_roi, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
        img_1080_roi_path = os.path.join(img_dir, filename.replace('.mp4', '_1080p_roi.jpg'))
        cv2.imwrite(img_1080_roi_path, frame_1080_roi)


        frame_height, frame_width = frame_1080.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print(f"Error: No se pudo abrir el archivo de video para escritura: {filepath}")
            logging.error(f"No se pudo abrir el archivo de video para escritura: {filepath}")
            continue  # Salta este segmento

        start_time = time.time()
        print(f"Grabando: {filepath}")
        logging.info(f"Inicio de grabación: {filepath}")

        # Estadísticas acumuladas para el segmento
        frames_in_segment = 0
        roi_left_frames = 0
        roi_center_frames = 0
        roi_right_frames = 0
        person_counts = []
        out_roi_frames = 0

        last_frame_1080 = None

        # Timestamp para el final del segmento
        timestamp_2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        minuto_inicio = now.minute
        try:
            while True:
                if frames_in_segment == 0:
                    current_frame_1080 = frame_1080
                    current_detections = in_detections
                else:
                    in_cam = cam_queue.get()
                    in_detections = detections_queue.get()
                    current_frame_1080 = in_cam.getCvFrame()
                    current_detections = in_detections

                # --- Detección y estadísticas de personas ---
                roi_left = escalar_roi(roi_left_orig, current_frame_1080.shape, (original_width, original_height))
                roi_center = escalar_roi(roi_center_orig, current_frame_1080.shape, (original_width, original_height))
                roi_right = escalar_roi(roi_right_orig, current_frame_1080.shape, (original_width, original_height))

                roi_left_present = False
                roi_center_present = False
                roi_right_present = False
                person_count_this_frame = 0

                # --- Procesar detecciones
                for detection in current_detections.detections:
                    x1 = int(detection.xmin * current_frame_1080.shape[1])
                    y1 = int(detection.ymin * current_frame_1080.shape[0])
                    x2 = int(detection.xmax * current_frame_1080.shape[1])
                    y2 = int(detection.ymax * current_frame_1080.shape[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if detection.label == 0:
                        # Persona: cuenta para ROIs y estadísticas
                        person_count_this_frame += 1

                        color = (0, 255, 0)  # Verde para personas
                        label_text = "Persona"

                        # Calcular centro en RGB
                        cx_rgb = int((x1 + x2) / 2)
                        cy_rgb = int((y1 + y2) / 2)


                        # Dibuja el centro para depuración visual
                        cv2.circle(current_frame_1080, (cx_rgb, cy_rgb), 5, (0,255,255), -1)


                        # Permite que una persona cuente para varios ROIs si corresponde
                        if roi_left[0] <= cx_rgb < roi_left[0] + roi_left[2] and roi_left[1] <= cy_rgb < roi_left[1] + roi_left[3]:
                            roi_left_present = True
                            color = (255, 0, 0)  # Azul para ROI izquierdo
                            label_text = "ROI Left"
                        elif roi_center[0] <= cx_rgb < roi_center[0] + roi_center[2] and roi_center[1] <= cy_rgb < roi_center[1] + roi_center[3]:
                            roi_center_present = True
                            color = (0, 255, 255)  # Amarillo para ROI central
                            label_text = "ROI Center"
                        elif roi_right[0] <= cx_rgb < roi_right[0] + roi_right[2] and roi_right[1] <= cy_rgb < roi_right[1] + roi_right[3]:
                            roi_right_present = True
                            color = (0, 0, 255)  # Rojo para ROI derecho
                            label_text = "ROI Right"

                        # Dibuja el bounding box y el centro
                        cv2.rectangle(current_frame_1080, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(current_frame_1080, (cx, cy), 5, color, -1)
                        cv2.putText(current_frame_1080, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Supón que frame_person_count_total es el número de personas detectadas en este frame
                cv2.putText(
                    current_frame_1080,
                    f"Qty: {person_count_this_frame}",
                    (30, 40),  # posición (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,        # tamaño de fuente
                    (0, 255, 255),  # color (amarillo)
                    3           # grosor
                )       

                # Estadísticas de personas y ROIs
                person_counts.append(person_count_this_frame)
                if roi_left_present:
                    roi_left_frames += 1
                if roi_center_present:
                    roi_center_frames += 1
                if roi_right_present:
                    roi_right_frames += 1
                if (person_count_this_frame > 0) and not (roi_left_present or roi_center_present or roi_right_present):
                    out_roi_frames += 1


                # Dibuja los ROIs sobre el frame
                cv2.rectangle(current_frame_1080, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
                cv2.rectangle(current_frame_1080, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
                cv2.rectangle(current_frame_1080, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)

                out.write(current_frame_1080)
                frames_in_segment += 1

                # Guarda el último frame
                last_frame_1080 = current_frame_1080

                now = datetime.now()
                # Corta exactamente cuando cambia el minuto y el segundo es 0
                if now.minute != minuto_inicio and now.second == 0:
                    timestamp_completo = now.strftime('%Y-%m-%d %H:%M:%S.%f')
                    print(f"Último frame: {timestamp_completo}")
                    print(f"Grabación de {MINUTO_MULTIPLO} minuto(s) completada.")
                    logging.info(f"Fin de grabación: {filepath} | Timestamp: {timestamp_completo}")
                    break
        except KeyboardInterrupt:
            print("Grabación interrumpida por el usuario.")
            logging.info("Grabación interrumpida por el usuario.")
            break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
            print(f"Error durante la grabación: {e}")
        finally:
            out.release()
            # Guardar imagen final de cada stream
            if last_frame_1080 is not None:
                img_final_1080 = os.path.join(img_dir, filename.replace('.mp4', '_1080p_last.jpg'))
                cv2.imwrite(img_final_1080, last_frame_1080)

            # Guardar resumen del segmento en el CSV del día
            pct_left = 100 * roi_left_frames / frames_in_segment if frames_in_segment else 0
            pct_center = 100 * roi_center_frames / frames_in_segment if frames_in_segment else 0
            pct_right = 100 * roi_right_frames / frames_in_segment if frames_in_segment else 0
            pct_out_roi = 100 * out_roi_frames / frames_in_segment if frames_in_segment else 0
            avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0
            max_personas = int(np.max(person_counts)) if person_counts else 0

            event_id = sql_logger.insert_event(conn_test, TABLE_NAME, (ts_str, DEVICE_NAME, SCRIPT_NAME, "INFO", 
                                                f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}", f"{pct_out_roi:.1f}", avg_personas, max_personas,
                                                filename, "DETECTION"))
            if event_id == -1:
                logging.warning("Failed to write initialization event to the database.")                
            print(
                f"%Left={pct_left:.1f} %Center={pct_center:.1f} %Right={pct_right:.1f} "
                f"%!ROI={pct_out_roi:.1f} AVG_P={avg_personas} MAX_P={max_personas} "
                f"VideoFile={filename} "
            )
    cv2.destroyAllWindows()