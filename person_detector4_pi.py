import depthai as dai
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter

# --- Configuración de ROIs ---
roi_left = (150, 200, 300, 300)
roi_center = (500, 200, 600, 300)
roi_right = (650, 200, 800, 400)

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

# Cámara color
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(416, 416)  # <-- Debe coincidir con el modelo
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)

# --- NN ---
detection_nn = pipeline.createYoloDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setIouThreshold(0.5)
detection_nn.setBlobPath(model_path)
detection_nn.input.setBlocking(False)
detection_nn.input.setQueueSize(1)

# --- Anchors y anchor masks para YOLOv5n 416x416 COCO ---
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
cam_rgb.preview.link(detection_nn.input)
xout_rgb = pipeline.createXLinkOut()
xout_nn = pipeline.createXLinkOut()
xout_rgb.setStreamName("video")
xout_nn.setStreamName("detections")
cam_rgb.preview.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)

# --- Inicializar dispositivo ---
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1280, 720)

    # Parámetros para el reporte cada 5 segundos
    fps = 10
    intervalo = int(fps * 5)
    frame_count = 0
    personas_intervalo = set()
    ultimo_reporte_texto = ""

    while True:
        in_video = video_queue.get()
        in_detections = detections_queue.get()

        frame = in_video.getCvFrame()
        detections = []

        for detection in in_detections.detections:
            print("Label:", detection.label, "Conf:", detection.confidence)
            if detection.label == 0:  # Persona en COCO                x1 = int(detection.xmin * frame.shape[1])
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

        # Acumula los IDs vistos en este intervalo
        personas_intervalo.update(ids_presentes)
        frame_count += 1

        # Reporte cada 5 segundos
        if frame_count >= intervalo:
            ultimo_reporte_texto = f"{time.strftime('%H:%M:%S')} Personas: {len(personas_intervalo)}"
            print(f"[{ultimo_reporte_texto}] (IDs: {sorted(personas_intervalo)})")
            personas_intervalo.clear()
            frame_count = 0

        # Muestra el reporte en la ventana
        if ultimo_reporte_texto:
            cv2.putText(frame, ultimo_reporte_texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83 or key == ord('d'):  # Flecha derecha o 'd'
            pass  # No aplica en tiempo real
        elif key == ord('+'):
            cv2.resizeWindow('Detection', 1920, 1080)
        elif key == ord('-'):
            cv2.resizeWindow('Detection', 640, 360)

    cv2.destroyAllWindows()