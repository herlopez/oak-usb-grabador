#  Graba archivo de 416x416 y lo analiza para contar personas en 3 ROIs
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
    time.sleep(espera)

# --- Configuración de ROIs y pipeline ---
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 250, 350, 300)
roi_hinge  = (1400, 400, 416, 416)
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

xout_rgb = pipeline.createXLinkOut()
xout_nn = pipeline.createXLinkOut()
xout_depth = pipeline.createXLinkOut()
xout_rgb.setStreamName("video")
xout_nn.setStreamName("detections")
xout_depth.setStreamName("depth")

manip.out.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)
stereo.depth.link(xout_depth.input)

# --- Grabación segmentada ---
MINUTO_MULTIPLO = 1  # Cambia este valor para grabar cada X minutos
fps = 10
segment_duration = 60 * MINUTO_MULTIPLO

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    # depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)  # No se usa para solo personas

    esperar_hasta_proximo_multiplo(MINUTO_MULTIPLO)

    while True:
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

        now = datetime.now()
        day_folder = now.strftime("%Y%m%d")
        hour_folder = now.strftime("%H")
        output_dir = os.path.join(VIDEO_DIR, day_folder, hour_folder)
        os.makedirs(output_dir, exist_ok=True)

        # CSV setup en la carpeta del día (una sola fila por segmento)
        csv_path = os.path.join(VIDEO_DIR, day_folder, "person_stats.csv")
        new_csv = not os.path.exists(csv_path)
        csv_file = open(csv_path, "a", newline="")
        import csv
        csv_writer = csv.writer(csv_file)
        if new_csv:
            csv_writer.writerow([
                "Fecha", "Hora", "Minuto", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Personas", "VideoFile", "Script"
            ])
        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        filepath = os.path.join(output_dir, filename)

        # Espera el primer frame para obtener el tamaño real
        in_video = video_queue.get()
        in_detections = detections_queue.get()
        frame = in_video.getCvFrame()
        frame_height, frame_width = frame.shape[:2]

        # Guardar imagen original
        img_original_path = os.path.join(output_dir, filename.replace('.mp4', '_original.jpg'))
        cv2.imwrite(img_original_path, frame)

        # Dibujar ROIs en copia del frame
        frame_roi = frame.copy()
        def escalar_roi(roi):
            return (
                int(roi[0] * frame.shape[1] / original_width),
                int(roi[1] * frame.shape[0] / original_height),
                int(roi[2] * frame.shape[1] / original_width),
                int(roi[3] * frame.shape[0] / original_height)
            )
        roi_left = escalar_roi(roi_left_orig)
        roi_center = escalar_roi(roi_center_orig)
        roi_right = escalar_roi(roi_right_orig)
        cv2.rectangle(frame_roi, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
        cv2.rectangle(frame_roi, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
        cv2.rectangle(frame_roi, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
        # Guardar imagen con ROIs
        img_roi_path = os.path.join(output_dir, filename.replace('.mp4', '_roi.jpg'))
        cv2.imwrite(img_roi_path, frame_roi)

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

        try:
            while True:
                # Usa el primer frame leído antes del bucle
                if frames_in_segment == 0:
                    current_frame = frame
                    current_detections = in_detections
                else:
                    in_video = video_queue.get()
                    in_detections = detections_queue.get()
                    current_frame = in_video.getCvFrame()
                    current_detections = in_detections

                scale_x = current_frame.shape[1] / original_width
                scale_y = current_frame.shape[0] / original_height

                def escalar_roi(roi):
                    return (
                        int(roi[0] * scale_x),
                        int(roi[1] * scale_y),
                        int(roi[2] * scale_x),
                        int(roi[3] * scale_y)
                    )

                roi_left = escalar_roi(roi_left_orig)
                roi_center = escalar_roi(roi_center_orig)
                roi_right = escalar_roi(roi_right_orig)

                # YOLO + SORT solo personas
                detections = []
                for detection in current_detections.detections:
                    if detection.label == 0:
                        x1 = int(detection.xmin * current_frame.shape[1])
                        y1 = int(detection.ymin * current_frame.shape[0])
                        x2 = int(detection.xmax * current_frame.shape[1])
                        y2 = int(detection.ymax * current_frame.shape[0])
                        conf = detection.confidence
                        detections.append([x1, y1, x2, y2, conf])

                tracks = tracker.update(np.array(detections)) if detections else []

                ids_presentes = set()
                roi_left_present = False
                roi_center_present = False
                roi_right_present = False

                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track)
                    ids_presentes.add(track_id)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
                        roi_left_present = True
                    elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
                        roi_center_present = True
                    elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
                        roi_right_present = True

                # Acumular estadísticas para el segmento
                frames_in_segment += 1
                person_counts.append(len(ids_presentes))
                if roi_left_present:
                    roi_left_frames += 1
                if roi_center_present:
                    roi_center_frames += 1
                if roi_right_present:
                    roi_right_frames += 1
                if (len(ids_presentes) > 0) and not (roi_left_present or roi_center_present or roi_right_present):
                    out_roi_frames += 1

                out.write(current_frame)

                if time.time() - start_time >= segment_duration:
                    print(f"Grabación de {MINUTO_MULTIPLO} minuto(s) completada.")
                    logging.info(f"Fin de grabación: {filepath}")
                    break
        except KeyboardInterrupt:
            print("Grabación interrumpida por el usuario.")
            logging.info("Grabación interrumpida por el usuario.")
            break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
        finally:
            out.release()
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
                filename, "oak_recorder_3.py"
            ])
            csv_file.flush()
            csv_file.close()

    cv2.destroyAllWindows()