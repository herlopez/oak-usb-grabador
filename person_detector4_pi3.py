import depthai as dai
import cv2
import numpy as np
from sort.sort import Sort
import time
import blobconverter

# Define los ROIs en la resolución original de la cámara
roi_left_orig   = (100, 500, 350, 250)
roi_center_orig = (880, 400, 130, 150)
roi_right_orig  = (1200, 300, 350, 250)

original_width = 1920
original_height = 1080

# Inicializar SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Descargar modelo YOLOv5n
model_path = blobconverter.from_zoo(
    name="yolov5n_coco_416x416",
    zoo_type="depthai",
    shaves=6
)

# Pipeline
pipeline = dai.Pipeline()

# Color Camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# ImageManip para redimensionar
manip = pipeline.createImageManip()
manip.initialConfig.setResize(416, 416)
manip.initialConfig.setKeepAspectRatio(False)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.setMaxOutputFrameSize(416 * 416 * 3)
cam_rgb.video.link(manip.inputImage)

# Stereo Depth
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Red neuronal YOLO
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

# Conexiones
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

# Ejecutar
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    detections_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1280, 720)

    fps = 10
    intervalo = int(fps * 5)
    frame_count = 0
    personas_intervalo = set()
    cajas_contador = {"Hinge Big": 0, "Hinge Small": 0, "Outside": 0}
    ultimo_reporte_texto = ""

    while True:
        in_video = video_queue.get()
        in_detections = detections_queue.get()
        in_depth = depth_queue.get()

        frame = in_video.getCvFrame()
        depth_frame = in_depth.getFrame()
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        scale_x = frame.shape[1] / original_width
        scale_y = frame.shape[0] / original_height

        # Reescalar ROIs
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

        cv2.imshow('Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
