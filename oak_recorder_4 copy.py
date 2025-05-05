# prueba base estable! CON PROFUNDIDAD PROMEDIO POR ROI
# Graba archivo de video y lo analiza para contar personas en 3 ROIs,
# mide la profundidad promedio de personas en cada ROI y guarda estadísticas en CSV.

import depthai as dai
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter
import logging
import os
from datetime import datetime, timedelta

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

xout_cam = pipeline.createXLinkOut()
xout_cam.setStreamName("cam")
cam_rgb.video.link(xout_cam.input)

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
xout_manip = pipeline.createXLinkOut()
xout_nn.setStreamName("detections")
xout_depth.setStreamName("depth")
xout_manip.setStreamName("manip")

manip.out.link(xout_manip.input)
detection_nn.out.link(xout_nn.input)
stereo.depth.link(xout_depth.input)

# --- Grabación segmentada ---
MINUTO_MULTIPLO = 1
fps = 10
segment_duration = 60 * MINUTO_MULTIPLO

with dai.Device(pipeline) as device:
    cam_queue = device.getOutputQueue("cam", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    manip_queue = device.getOutputQueue("manip", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    esperar_hasta_proximo_multiplo(MINUTO_MULTIPLO)

    while True:
        manage_disk_usage(VIDEO_DIR, MAX_USAGE_BYTES)

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
                "Fecha", "Hora", "Minuto", "%ROI_Left", "%ROI_Center", "%ROI_Right", "%Fuera_ROI", "Personas", "VideoFile", "Script", "objeto_hinge",
                "Profundidad_ROI_Left", "Profundidad_ROI_Center", "Profundidad_ROI_Right"
            ])
        filename = now.strftime(f"output_%Y%m%d_%H%M%S.mp4")
        filepath = os.path.join(output_dir, filename)

        # Espera el primer frame para obtener el tamaño real
        in_cam = cam_queue.get()
        in_detections = detections_queue.get()
        in_manip = manip_queue.get()
        in_depth = depth_queue.get()
        frame_1080 = in_cam.getCvFrame()
        frame_416 = in_manip.getCvFrame()
        depth_frame = in_depth.getFrame()

        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        img_original_path = os.path.join(img_dir, filename.replace('.mp4', '_1080p.jpg'))
        cv2.imwrite(img_original_path, frame_1080)
        img_416_path = os.path.join(img_dir, filename.replace('.mp4', '_416.jpg'))
        cv2.imwrite(img_416_path, frame_416)

        def escalar_roi(roi, shape, orig_shape):
            return (
                int(roi[0] * shape[1] / orig_shape[0]),
                int(roi[1] * shape[0] / orig_shape[1]),
                int(roi[2] * shape[1] / orig_shape[0]),
                int(roi[3] * shape[0] / orig_shape[1])
            )

        frame_height, frame_width = frame_1080.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print(f"Error: No se pudo abrir el archivo de video para escritura: {filepath}")
            logging.error(f"No se pudo abrir el archivo de video para escritura: {filepath}")
            csv_file.close()
            continue

        start_time = time.time()
        print(f"Grabando: {filepath}")
        logging.info(f"Inicio de grabación: {filepath}")

        frames_in_segment = 0
        roi_left_frames = 0
        roi_center_frames = 0
        roi_right_frames = 0
        person_counts = []
        out_roi_frames = 0
        objeto_hinge_count = 0
        objeto_hinge_presente_anterior = False

        prof_left = []
        prof_center = []
        prof_right = []

        last_frame_1080 = None
        last_frame_416 = None

        try:
            while True:
                if frames_in_segment == 0:
                    current_frame_1080 = frame_1080
                    current_detections = in_detections
                    current_frame_416 = frame_416
                    current_depth = depth_frame
                else:
                    in_cam = cam_queue.get()
                    in_detections = detections_queue.get()
                    in_manip = manip_queue.get()
                    in_depth = depth_queue.get()
                    current_frame_1080 = in_cam.getCvFrame()
                    current_detections = in_detections
                    current_frame_416 = in_manip.getCvFrame()
                    current_depth = in_depth.getFrame()

                last_frame_1080 = current_frame_1080
                last_frame_416 = current_frame_416

                roi_left = escalar_roi(roi_left_orig, current_frame_1080.shape, (original_width, original_height))
                roi_center = escalar_roi(roi_center_orig, current_frame_1080.shape, (original_width, original_height))
                roi_right = escalar_roi(roi_right_orig, current_frame_1080.shape, (original_width, original_height))
                roi_hinge_scaled = escalar_roi(roi_hinge_orig, current_frame_1080.shape, (original_width, original_height))
                roi_hinge_area = roi_hinge_scaled[2] * roi_hinge_scaled[3]
                objeto_hinge_presente = False

                roi_left_present = False
                roi_center_present = False
                roi_right_present = False
                person_count_this_frame = 0

                # --- Visualización sobre el frame ---
                frame_vis = current_frame_1080.copy()

                for detection in current_detections.detections:
                    x1 = int(detection.xmin * current_frame_1080.shape[1])
                    y1 = int(detection.ymin * current_frame_1080.shape[0])
                    x2 = int(detection.xmax * current_frame_1080.shape[1])
                    y2 = int(detection.ymax * current_frame_1080.shape[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if detection.label == 0:
                        person_count_this_frame += 1
                        # Medición de profundidad en el centro del bounding box
                        if 0 <= cy < current_depth.shape[0] and 0 <= cx < current_depth.shape[1]:
                            depth_value = int(current_depth[cy, cx])
                        else:
                            depth_value = 0

                        if roi_left[0] <= cx < roi_left[0] + roi_left[2] and roi_left[1] <= cy < roi_left[1] + roi_left[3]:
                            color = (255, 0, 0)
                            roi_left_present = True
                            prof_left.append(depth_value)
                            roi_label = f"Left {depth_value}mm"
                        elif roi_center[0] <= cx < roi_center[0] + roi_center[2] and roi_center[1] <= cy < roi_center[1] + roi_center[3]:
                            color = (0, 255, 0)
                            roi_center_present = True
                            prof_center.append(depth_value)
                            roi_label = f"Center {depth_value}mm"
                        elif roi_right[0] <= cx < roi_right[0] + roi_right[2] and roi_right[1] <= cy < roi_right[1] + roi_right[3]:
                            color = (0, 0, 255)
                            roi_right_present = True
                            prof_right.append(depth_value)
                            roi_label = f"Right {depth_value}mm"
                        else:
                            color = (0, 255, 255)
                            roi_label = f"Fuera {depth_value}mm"

                        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_vis, roi_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        inter_x1 = max(x1, roi_hinge_scaled[0])
                        inter_y1 = max(y1, roi_hinge_scaled[1])
                        inter_x2 = min(x2, roi_hinge_scaled[0] + roi_hinge_scaled[2])
                        inter_y2 = min(y2, roi_hinge_scaled[1] + roi_hinge_scaled[3])
                        inter_w = max(0, inter_x2 - inter_x1)
                        inter_h = max(0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        if roi_hinge_area > 0 and (inter_area / roi_hinge_area) > 0.2:
                            objeto_hinge_presente = True

                cv2.rectangle(frame_vis, (roi_left[0], roi_left[1]), (roi_left[0]+roi_left[2], roi_left[1]+roi_left[3]), (255,0,0), 2)
                cv2.rectangle(frame_vis, (roi_center[0], roi_center[1]), (roi_center[0]+roi_center[2], roi_center[1]+roi_center[3]), (0,255,0), 2)
                cv2.rectangle(frame_vis, (roi_right[0], roi_right[1]), (roi_right[0]+roi_right[2], roi_right[1]+roi_right[3]), (0,0,255), 2)
                cv2.rectangle(frame_vis, (roi_hinge_scaled[0], roi_hinge_scaled[1]), (roi_hinge_scaled[0]+roi_hinge_scaled[2], roi_hinge_scaled[1]+roi_hinge_scaled[3]), (0,128,255), 2)

                cv2.imshow("Video con ROIs y Profundidad", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Procesamiento interrumpido por el usuario.")
                    break

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

                if time.time() - start_time >= segment_duration:
                    break
        except KeyboardInterrupt:
            print("Grabación interrumpida por el usuario.")
            logging.info("Grabación interrumpida por el usuario.")
            break
        except Exception as e:
            logging.error(f"Error durante la grabación: {e}")
        finally:
            out.release()
            if last_frame_1080 is not None:
                img_final_1080 = os.path.join(img_dir, filename.replace('.mp4', '_1080p_last.jpg'))
                cv2.imwrite(img_final_1080, last_frame_1080)
            if last_frame_416 is not None:
                img_final_416 = os.path.join(img_dir, filename.replace('.mp4', '_416_last.jpg'))
                cv2.imwrite(img_final_416, last_frame_416)
            pct_left = 100 * roi_left_frames / frames_in_segment if frames_in_segment else 0
            pct_center = 100 * roi_center_frames / frames_in_segment if frames_in_segment else 0
            pct_right = 100 * roi_right_frames / frames_in_segment if frames_in_segment else 0
            pct_out_roi = 100 * out_roi_frames / frames_in_segment if frames_in_segment else 0
            avg_personas = int(np.ceil(np.mean(person_counts))) if person_counts else 0

            # Profundidad promedio por ROI
            prof_left_avg = int(np.mean([d for d in prof_left if d > 0])) if prof_left else 0
            prof_center_avg = int(np.mean([d for d in prof_center if d > 0])) if prof_center else 0
            prof_right_avg = int(np.mean([d for d in prof_right if d > 0])) if prof_right else 0

            fecha = now.strftime('%Y-%m-%d')
            hora = now.strftime('%H')
            minuto = now.strftime('%M')

            csv_writer.writerow([
                fecha, hora, minuto,
                f"{pct_left:.1f}", f"{pct_center:.1f}", f"{pct_right:.1f}", f"{pct_out_roi:.1f}", avg_personas,
                filename, "oak_recorder_4.py", objeto_hinge_count,
                prof_left_avg, prof_center_avg, prof_right_avg
            ])
            print(
                f"%ROI_Left={pct_left:.1f} %ROI_Center={pct_center:.1f} %ROI_Right={pct_right:.1f} "
                f"%Fuera_ROI={pct_out_roi:.1f} Personas={avg_personas} "
                f"VideoFile={filename} objeto_hinge={objeto_hinge_count} "
                f"Profundidad_Left={prof_left_avg} Profundidad_Center={prof_center_avg} Profundidad_Right={prof_right_avg}"
            )
            csv_file.flush()
            csv_file.close()

    cv2.destroyAllWindows()