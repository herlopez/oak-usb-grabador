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
roi_right_orig  = (1200, 300, 350, 250)
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
frame_width = 1920
frame_height = 1080
segment_duration = 60 * MINUTO_MULTIPLO

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    # depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)  # No se usa para solo personas

    # CSV setup
    csv_path = os.path.join(VIDEO_DIR, day_folder, "person_stats.csv")
    new_csv = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    import csv
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Avg"])

    esperar_hasta_proximo_multiplo(MINUTO_MULTIPLO)

    while True:
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

        intervalo = int(fps * 60)  # 1 minuto de estadísticas
        frame_count = 0
        personas_intervalo = set()
        ultimo_reporte_texto = ""

        # Estadísticas por minuto
        frames_in_minute = 0
        roi_left_frames = 0
        roi_center_frames = 0
        roi_right_frames = 0
        person_counts = []
        out_roi_frames = 0

        try:
            while True:
                in_video = video_queue.get()
                in_detections = detections_queue.get()
                frame = in_video.getCvFrame()

                scale_x = frame.shape[1] / original_width
                scale_y = frame.shape[0] / original_height

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
                for detection in in_detections.detections:
                    if detection.label == 0:
                        x1 = int(detection.xmin * frame.shape[1])
                        y1 = int(detection.ymin * frame.shape[0])
                        x2 = int(detection.xmax * frame.shape[1])
                        y2 = int(detection.ymax * frame.shape[0])
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
                        roi_label = "Left"
                        color = (255, 0, 0)
                        roi_left_present = True
                    elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
                        roi_label = "Center"
                        color = (0, 255, 0)
                        roi_center_present = True
                    elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
                        roi_label = "Right"
                        color = (0, 0, 255)
                        roi_right_present = True
                    else:
                        roi_label = ""
                        color = (128, 128, 128)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Dibujar los tres ROIs
                cv2.rectangle(frame, (roi_left[0], roi_left[1]), (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]), (255, 0, 0), 2)
                cv2.rectangle(frame, (roi_center[0], roi_center[1]), (roi_center[0] + roi_center[2], roi_center[1] + roi_center[3]), (0, 255, 0), 2)
                cv2.rectangle(frame, (roi_right[0], roi_right[1]), (roi_right[0] + roi_right[2], roi_right[1] + roi_right[3]), (0, 0, 255), 2)

                personas_intervalo.update(ids_presentes)
                frame_count += 1

                # Estadísticas por minuto
                frames_in_minute += 1
                person_counts.append(len(ids_presentes))
                if roi_left_present:
                    roi_left_frames += 1
                if roi_center_present:
                    roi_center_frames += 1
                if roi_right_present:
                    roi_right_frames += 1
                if (len(ids_presentes) > 0) and not (roi_left_present or roi_center_present or roi_right_present):
                    out_roi_frames += 1

                if frame_count >= intervalo:
                    pct_left = 100 * roi_left_frames / frames_in_minute
                    pct_center = 100 * roi_center_frames / frames_in_minute
                    pct_right = 100 * roi_right_frames / frames_in_minute
                    pct_out_roi = 100 * out_roi_frames / frames_in_minute

                    avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    reporte = (f"{timestamp} | %ROI_Left: {pct_left:.1f} | %ROI_Center: {pct_center:.1f} | "
                               f"%ROI_Right: {pct_right:.1f} | %Fuera_ROI: {pct_out_roi:.1f} | Ocupacion: {avg_personas}")
                    print(reporte)
                    csv_writer.writerow([timestamp, f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}",
                                         f"{pct_out_roi:.1f}", avg_personas])
                    csv_file.flush()

                    # Reset stats
                    personas_intervalo.clear()
                    frame_count = 0
                    frames_in_minute = 0
                    roi_left_frames = 0
                    roi_center_frames = 0
                    roi_right_frames = 0
                    person_counts = []
                    out_roi_frames = 0

                if ultimo_reporte_texto:
                    cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                out.write(frame)

                # Si tienes entorno gráfico, puedes mostrarlo:
                # cv2.imshow('Detection', frame)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'):
                #     raise KeyboardInterrupt

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

    csv_file.close()
    cv2.destroyAllWindows()