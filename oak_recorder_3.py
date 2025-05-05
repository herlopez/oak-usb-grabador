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

# --- Tu código de detección y pipeline ---
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
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1280, 720)

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

        intervalo = int(fps * 5)
        frame_count = 0
        personas_intervalo = set()
        cajas_contador = {"Hinge Big": 0, "Hinge Small": 0, "Outside": 0}
        ultimo_reporte_texto = ""

        try:
            while True:
                in_video = video_queue.get()
                in_detections = detections_queue.get()
                in_depth = depth_queue.get()

                frame = in_video.getCvFrame()
                depth_frame = in_depth.getFrame()
                depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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

                # YOLO + SORT
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
                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track)
                    ids_presentes.add(track_id)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if roi_left[0] <= cx < roi_left[0] + roi_left[2]:
                        roi_label = "Hinge Big"
                        color = (255, 0, 0)
                    elif roi_center[0] <= cx < roi_center[0] + roi_center[2]:
                        roi_label = "Hinge Small"
                        color = (0, 255, 0)
                    elif roi_right[0] <= cx < roi_right[0] + roi_right[2]:
                        roi_label = "Outside"
                        color = (0, 0, 255)
                    else:
                        roi_label = ""
                        color = (128, 128, 128)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Detección de cajas por bordes + profundidad
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                    area = cv2.contourArea(cnt)

                    if len(approx) == 4 and area > 4000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        depth_roi = depth_normalized[y:y+h, x:x+w]
                        avg_depth = np.mean(depth_roi)

                        if 30 < avg_depth < 150:
                            cx = x + w // 2
                            cy = y + h // 2

                            if roi_left[0] <= cx < roi_left[0] + roi_left[2]:
                                roi_label = "Hinge Big"
                                color = (255, 255, 0)
                            elif roi_center[0] <= cx < roi_center[0] + roi_center[2]:
                                roi_label = "Hinge Small"
                                color = (0, 255, 255)
                            elif roi_right[0] <= cx < roi_right[0] + roi_right[2]:
                                roi_label = "Outside"
                                color = (255, 0, 255)
                            else:
                                roi_label = ""
                                color = (100, 100, 100)

                            if roi_label:
                                cajas_contador[roi_label] += 1

                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"Caja {roi_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Dibujar ROIs
                for roi, color in zip([roi_left, roi_center, roi_right], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), color, 2)

                personas_intervalo.update(ids_presentes)
                frame_count += 1

                if frame_count >= intervalo:
                    ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Personas: {len(personas_intervalo)} | Cajas: {cajas_contador}"
                    print(ultimo_reporte_texto)
                    personas_intervalo.clear()
                    cajas_contador = {k: 0 for k in cajas_contador}
                    frame_count = 0

                if ultimo_reporte_texto:
                    cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                out.write(frame)  # <-- Graba el frame procesado

                cv2.imshow('Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
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

    cv2.destroyAllWindows()