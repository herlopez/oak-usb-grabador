# prueba base estable!
#  Graba archivo de video y lo analiza para contar personas en 3 ROIs
#  y fuera de ellas. Guarda estadísticas en CSV y elimina archivos viejos
#  para mantener el uso del disco por debajo de 800 GB.
#  Requiere la librería SORT para el seguimiento de objetos.
#  Se recomienda usar un SSD NVMe para evitar problemas de escritura.

import depthai as dai
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter
import logging
import os
from datetime import datetime, timedelta



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
    print(f"Esperando {espera:.2f} segundos hasta el próximo múltiplo de {minuto_multiplo} minutos...")
    while espera > 0:
        time.sleep(min(1, espera))
        espera -= 1

def esperar_hasta_proximo_multiplo_2(minuto_multiplo):
    now = datetime.now()
    if now.second < 10:
        minutos = now.minute
        segundos = now.second
        microsegundos = now.microsecond
        minutos_a_sumar = (minuto_multiplo - (minutos % minuto_multiplo)) % minuto_multiplo
        if minutos_a_sumar == 0 and (segundos > 0 or microsegundos > 0):
            minutos_a_sumar = minuto_multiplo
        proximo = (now + timedelta(minutes=minutos_a_sumar)).replace(second=0, microsecond=0)
        espera = (proximo - now).total_seconds()
        print(f"Esperando {espera:.2f} segundos hasta el próximo múltiplo de {minuto_multiplo} minutos...")
        while espera > 0:
            time.sleep(min(1, espera))
            espera -= 1

def esperar_hasta_segundo_59():
    now = datetime.now()
    espera = 59 - now.second - now.microsecond / 1_000_000
    if espera < 0:
        espera += 60
    print(f"Esperando {espera:.2f} segundos hasta el segundo 59...")
    time.sleep(espera+1)

# Función para escalar ROIs
def escalar_roi(roi, shape, orig_shape):
    return (
        int(roi[0] * shape[1] / orig_shape[0]),  # x
        int(roi[1] * shape[0] / orig_shape[1]),  # y
        int(roi[2] * shape[1] / orig_shape[0]),  # w
        int(roi[3] * shape[0] / orig_shape[1])   # h
    )

# --- Configuración de ROIs y pipeline ---
# Configuración de ROIs (imagen original 1920x1080)
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 250, 350, 300)
roi_hinge_orig  = (1400, 380, 500, 200)
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

mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

detection_nn = pipeline.createYoloDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setIouThreshold(0.5)
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
xout_depth = pipeline.createXLinkOut()
xout_manip = pipeline.createXLinkOut()  # Para obtener el frame 416x416
xout_nn.setStreamName("detections")
xout_depth.setStreamName("depth")
xout_manip.setStreamName("manip")

manip.out.link(xout_manip.input)
detection_nn.out.link(xout_nn.input)
stereo.depth.link(xout_depth.input)


# --- Grabación segmentada ---
MINUTO_MULTIPLO = 1  # Cambia este valor para grabar cada X minutos
fps = 10
segment_duration = 60 * MINUTO_MULTIPLO

with dai.Device(pipeline) as device:
    cam_queue = device.getOutputQueue("cam", maxSize=4, blocking=False)         # 1080p original
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    manip_queue = device.getOutputQueue("manip", maxSize=4, blocking=False)     # 416x416
    # Espera solo antes de iniciar la grabación
    # esperar_hasta_proximo_multiplo(MINUTO_MULTIPLO)
    # esperar_hasta_segundo_59()
    ultimo_minuto_corte = None
    ultimo_minuto_segmento = None

    while True:
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

        now = datetime.now()
        if ultimo_minuto_segmento == now.minute:
            esperar_hasta_proximo_multiplo_2(MINUTO_MULTIPLO)
            now = datetime.now()
            
        day_folder = now.strftime("%Y%m%d")
        hour_folder = now.strftime("%H")
        output_dir = os.path.join(VIDEO_DIR, day_folder, hour_folder)
        os.makedirs(output_dir, exist_ok=True)

        # CSV setup en la carpeta del día (una sola fila por segmento)
        csv_path = os.path.join(VIDEO_DIR, day_folder, f"{day_folder}_stats.csv")
        new_csv = not os.path.exists(csv_path)
        csv_file = open(csv_path, "a", newline="")
        import csv
        csv_writer = csv.writer(csv_file)
        if new_csv:
            csv_writer.writerow([
                "Fecha", "Hora", "Minuto", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Personas", "VideoFile", "Script", "objeto_hinge"
            ])
        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        filepath = os.path.join(output_dir, filename)
    
        ultimo_minuto_segmento = now.minute  # <-- Marca el minuto de este segmento

        # Espera el primer frame para obtener el tamaño real
        in_cam = cam_queue.get()
        in_detections = detections_queue.get()
        in_manip = manip_queue.get()
        frame_1080 = in_cam.getCvFrame()    # 1080p del stream original
        frame_416 = in_manip.getCvFrame()   # 416x416 del manip

        # Guardar imagen original 1080p
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        img_original_path = os.path.join(img_dir, filename.replace('.mp4', '_1080p.jpg'))
        cv2.imwrite(img_original_path, frame_1080)

        # Guardar imagen original 416x416
        img_416_path = os.path.join(img_dir, filename.replace('.mp4', '_416.jpg'))
        cv2.imwrite(img_416_path, frame_416)

        # 1080p con ROIs
        frame_1080_roi = frame_1080.copy()
        roi_left = escalar_roi(roi_left_orig, frame_1080.shape, (original_width, original_height))
        roi_center = escalar_roi(roi_center_orig, frame_1080.shape, (original_width, original_height))
        roi_right = escalar_roi(roi_right_orig, frame_1080.shape, (original_width, original_height))
        roi_hinge_1080 = escalar_roi(roi_hinge_orig, frame_1080.shape, (original_width, original_height))
        cv2.rectangle(frame_1080_roi, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
        cv2.rectangle(frame_1080_roi, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
        cv2.rectangle(frame_1080_roi, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
        cv2.rectangle(frame_1080_roi, (roi_hinge_1080[0], roi_hinge_1080[1]), (roi_hinge_1080[0]+roi_hinge_1080[2], roi_hinge_1080[1]+roi_hinge_1080[3]), (0,128,255), 2)
        img_1080_roi_path = os.path.join(img_dir, filename.replace('.mp4', '_1080p_roi.jpg'))
        cv2.imwrite(img_1080_roi_path, frame_1080_roi)

        # 416x416 con ROIs
        frame_416_roi = frame_416.copy()
        roi_left_416 = escalar_roi(roi_left_orig, frame_416.shape, (original_width, original_height))
        roi_center_416 = escalar_roi(roi_center_orig, frame_416.shape, (original_width, original_height))
        roi_right_416 = escalar_roi(roi_right_orig, frame_416.shape, (original_width, original_height))
        roi_hinge_416 = escalar_roi(roi_hinge_orig, frame_416.shape, (original_width, original_height))
        cv2.rectangle(frame_416_roi, (roi_left_416[0], roi_left_416[1]), (roi_left_416[0]+roi_left_416[2], roi_left_416[1]+roi_left_416[3]), (255,0,0), 2)
        cv2.rectangle(frame_416_roi, (roi_center_416[0], roi_center_416[1]), (roi_center_416[0]+roi_center_416[2], roi_center_416[1]+roi_center_416[3]), (0,255,0), 2)
        cv2.rectangle(frame_416_roi, (roi_right_416[0], roi_right_416[1]), (roi_right_416[0]+roi_right_416[2], roi_right_416[1]+roi_right_416[3]), (0,0,255), 2)
        cv2.rectangle(frame_416_roi, (roi_hinge_416[0], roi_hinge_416[1]), (roi_hinge_416[0]+roi_hinge_416[2], roi_hinge_416[1]+roi_hinge_416[3]), (0,128,255), 2)
        img_416_roi_path = os.path.join(img_dir, filename.replace('.mp4', '_416_roi.jpg'))
        cv2.imwrite(img_416_roi_path, frame_416_roi)

        frame_height, frame_width = frame_1080.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print(f"Error: No se pudo abrir el archivo de video para escritura: {filepath}")
            logging.error(f"No se pudo abrir el archivo de video para escritura: {filepath}")
            csv_file.close()
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
        objeto_hinge_count = 0  # <--- contador de eventos hinge
        objeto_hinge_presente_anterior = False
        
        # Para guardar el último frame de cada stream
        last_frame_1080 = None
        last_frame_416 = None


        try:
            while True:
                # Espera el primer frame para obtener el tamaño real
                if frames_in_segment == 0:
                    current_frame_1080 = frame_1080
                    current_detections = in_detections
                    current_frame_416 = frame_416
                    # Imprime timestamp del primer frame
                    print(f"Primer frame: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
                else:
                    in_cam = cam_queue.get()
                    in_detections = detections_queue.get()
                    in_manip = manip_queue.get()
                    current_frame_1080 = in_cam.getCvFrame()
                    current_detections = in_detections
                    current_frame_416 = in_manip.getCvFrame()

                # Guarda el último frame de cada stream
                last_frame_1080 = current_frame_1080
                last_frame_416 = current_frame_416

                # --- Detección y estadísticas de personas y objeto_hinge ---
                roi_hinge_scaled = escalar_roi(roi_hinge_orig, current_frame_1080.shape, (original_width, original_height))
                roi_hinge_area = roi_hinge_scaled[2] * roi_hinge_scaled[3]
                objeto_hinge_presente = False

                roi_left_present = False
                roi_center_present = False
                roi_right_present = False
                person_count_this_frame = 0

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
                        if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
                            roi_left_present = True
                        elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
                            roi_center_present = True
                        elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
                            roi_right_present = True
                    else:
                        # Objeto NO persona: cuenta para objeto_hinge
                        inter_x1 = max(x1, roi_hinge_scaled[0])
                        inter_y1 = max(y1, roi_hinge_scaled[1])
                        inter_x2 = min(x2, roi_hinge_scaled[0] + roi_hinge_scaled[2])
                        inter_y2 = min(y2, roi_hinge_scaled[1] + roi_hinge_scaled[3])
                        inter_w = max(0, inter_x2 - inter_x1)
                        inter_h = max(0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        if roi_hinge_area > 0 and (inter_area / roi_hinge_area) > 0.2:
                            objeto_hinge_presente = True

                if objeto_hinge_presente and not objeto_hinge_presente_anterior:
                    objeto_hinge_count += 1
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Event HINGE detected (Hinge in ROI)")
                objeto_hinge_presente_anterior = objeto_hinge_presente

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

                out.write(current_frame_1080)
                frames_in_segment += 1

                # Controla el corte del segmento solo una vez por ciclo
                now = datetime.now()
                # Solo cortar si estamos en el primer frame del nuevo minuto y no ya cortamos en este ciclo
                # Usar un flag para asegurar que el corte solo ocurra una vez por segmento
            

                now = datetime.now()
                if (now.second == 00 and frames_in_segment > 0
                    and (ultimo_minuto_corte is None or now.minute != ultimo_minuto_corte)):
                    print(f"Último frame: {now.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    print(f"Grabación de {MINUTO_MULTIPLO} minuto(s) completada.")
                    logging.info(f"Fin de grabación: {filepath}")
                    ultimo_minuto_corte = now.minute
                    break

        except KeyboardInterrupt:
            print("Grabación interrumpida por el usuario.")
            logging.info("Grabación interrumpida por el usuario.")
            break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
            break
        finally:
            out.release()
            # Guardar imagen final de cada stream
            if last_frame_1080 is not None:
                img_final_1080 = os.path.join(img_dir, filename.replace('.mp4', '_1080p_last.jpg'))
                cv2.imwrite(img_final_1080, last_frame_1080)
            if last_frame_416 is not None:
                img_final_416 = os.path.join(img_dir, filename.replace('.mp4', '_416_last.jpg'))
                cv2.imwrite(img_final_416, last_frame_416)
            # Guardar resumen del segmento en el CSV del día
            pct_left = 100 * roi_left_frames / frames_in_segment if frames_in_segment else 0
            pct_center = 100 * roi_center_frames / frames_in_segment if frames_in_segment else 0
            pct_right = 100 * roi_right_frames / frames_in_segment if frames_in_segment else 0
            pct_out_roi = 100 * out_roi_frames / frames_in_segment if frames_in_segment else 0
            avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

            fecha = now.strftime('%Y-%m-%d')
            hora = now.strftime('%H')
            minuto = now.strftime('%M')

            csv_writer.writerow([
                fecha, hora, minuto,
                f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}", f"{pct_out_roi:.1f}", avg_personas,
                filename, "oak_recorder_4.py", objeto_hinge_count
            ])
            print(
                f"%ROI_Left={pct_left:.1f} %ROI_Center={pct_center:.1f} %ROI_Right={pct_right:.1f} "
                f"%Fuera_ROI={pct_out_roi:.1f} Personas={avg_personas} "
                f"VideoFile={filename} objeto_hinge={objeto_hinge_count}"
            )
            csv_file.flush()
            csv_file.close()
            now = datetime.now()
            # if not (now.second < 5):
            #     esperar_hasta_proximo_multiplo_2(MINUTO_MULTIPLO)
    cv2.destroyAllWindows()