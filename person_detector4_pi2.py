import depthai as dai
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter

# Define los ROIs en la resolución original de la cámara
roi_left_orig   = (100, 550, 300, 250)
roi_center_orig = (880, 400, 100, 150)
roi_right_orig  = (1200, 300, 300, 200)

original_width = 1920
original_height = 1080


# --- Inicializar SORT tracker ---
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# --- Descarga automática del modelo desde la zoo ---
model_path = blobconverter.from_zoo(
    name="yolov5n_coco_416x416",
    zoo_type="depthai",
    shaves=6
)

# --- Pipeline DepthAI ---
pipeline = dai.Pipeline()

# Cámara color (resolución nativa)
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# ImageManip para resize sin mantener aspecto (es lo que espera el modelo YOLOv5n)
manip = pipeline.createImageManip()
manip.initialConfig.setResize(416, 416)
manip.initialConfig.setKeepAspectRatio(False)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.setMaxOutputFrameSize(416 * 416 * 3)  # <--- ¡clave para evitar el error de tamaño!

cam_rgb.video.link(manip.inputImage)

# --- NN ---
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

# Enlaces
manip.out.link(detection_nn.input)

# Salidas
xout_rgb = pipeline.createXLinkOut()
xout_nn = pipeline.createXLinkOut()
xout_rgb.setStreamName("video")
xout_nn.setStreamName("detections")
manip.out.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)

# --- Inicializar dispositivo ---
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1280, 720)

    fps = 10
    intervalo = int(fps * 5)
    frame_count = 0
    personas_intervalo = set()
    ultimo_reporte_texto = ""

    while True:
        in_video = video_queue.get()
        in_detections = detections_queue.get()

        frame = in_video.getCvFrame()        

        # Calcula el escalado para el frame actual
        scale_x = frame.shape[1] / original_width
        scale_y = frame.shape[0] / original_height

        roi_left = (
            int(roi_left_orig[0] * scale_x),
            int(roi_left_orig[1] * scale_y),
            int(roi_left_orig[2] * scale_x),
            int(roi_left_orig[3] * scale_y)
        )
        roi_center = (
            int(roi_center_orig[0] * scale_x),
            int(roi_center_orig[1] * scale_y),
            int(roi_center_orig[2] * scale_x),
            int(roi_center_orig[3] * scale_y)
        )
        roi_right = (
            int(roi_right_orig[0] * scale_x),
            int(roi_right_orig[1] * scale_y),
            int(roi_right_orig[2] * scale_x),
            int(roi_right_orig[3] * scale_y)
        )
        detections = []

        for detection in in_detections.detections:
            print("Label:", detection.label, "Conf:", detection.confidence)
            if detection.label == 0:  # Persona en COCO
                x1 = int(detection.xmin * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                x2 = int(detection.xmax * frame.shape[1])
                y2 = int(detection.ymax * frame.shape[0])
                conf = detection.confidence
                detections.append([x1, y1, x2, y2, conf])

        if len(detections) > 0:
            dets = np.array(detections)
            tracks = tracker.update(dets)
        else:
            tracks = []

        ids_presentes = set()
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            ids_presentes.add(track_id)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Determina en qué ROI está el centroide
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
            cv2.putText(frame, f'ID {track_id} {roi_label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Dibuja los tres ROIs
        cv2.rectangle(frame, (roi_left[0], roi_left[1]), (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (roi_center[0], roi_center[1]), (roi_center[0] + roi_center[2], roi_center[1] + roi_center[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (roi_right[0], roi_right[1]), (roi_right[0] + roi_right[2], roi_right[1] + roi_right[3]), (0, 0, 255), 2)

        personas_intervalo.update(ids_presentes)
        frame_count += 1

        if frame_count >= intervalo:
            ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Personas: {len(personas_intervalo)}"
            print(f"[{ultimo_reporte_texto}] (IDs: {sorted(personas_intervalo)})")
            personas_intervalo.clear()
            frame_count = 0

        if ultimo_reporte_texto:
            cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83 or key == ord('d'):
            pass
        elif key == ord('+'):
            cv2.resizeWindow('Detection', 1920, 1080)
        elif key == ord('-'):
            cv2.resizeWindow('Detection', 640, 360)

    cv2.destroyAllWindows()