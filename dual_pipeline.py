import depthai as dai
import blobconverter

# --- Crear Pipeline ---
pipeline = dai.Pipeline()

# --- Cámara Color ---
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(10)

# --- ImageManip para modelo 1 (YOLOv5n para personas) ---
manip_yolo = pipeline.createImageManip()
manip_yolo.initialConfig.setResize(416, 416)
manip_yolo.initialConfig.setKeepAspectRatio(False)
manip_yolo.setMaxOutputFrameSize(416 * 416 * 3)

# --- ImageManip para modelo 2 (modelo personalizado) ---
manip_custom = pipeline.createImageManip()
manip_custom.initialConfig.setResize(224, 224)  # o lo que necesite tu modelo
manip_custom.initialConfig.setKeepAspectRatio(False)
manip_custom.setMaxOutputFrameSize(224 * 224 * 3)

# --- Enlazar cámara con ambos ImageManip ---
cam_rgb.video.link(manip_yolo.inputImage)
cam_rgb.video.link(manip_custom.inputImage)

# --- YOLOv5n: detección de personas ---
detection_nn_yolo = pipeline.createYoloDetectionNetwork()
detection_nn_yolo.setConfidenceThreshold(0.5)
detection_nn_yolo.setNumClasses(80)
detection_nn_yolo.setCoordinateSize(4)
detection_nn_yolo.setIouThreshold(0.5)
detection_nn_yolo.setBlobPath(blobconverter.from_zoo("yolov5n_coco_416x416", zoo_type="depthai", shaves=6))
detection_nn_yolo.setAnchors([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
detection_nn_yolo.setAnchorMasks({"side52": [0,1,2], "side26": [3,4,5], "side13": [6,7,8]})
manip_yolo.out.link(detection_nn_yolo.input)

# --- Segundo modelo personalizado ---
detection_nn_custom = pipeline.createNeuralNetwork()
detection_nn_custom.setBlobPath("modelos/custom_model.blob")
manip_custom.out.link(detection_nn_custom.input)

# --- Salidas XLink ---
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("video")
cam_rgb.preview.link(xout_rgb.input)

xout_yolo = pipeline.createXLinkOut()
xout_yolo.setStreamName("detections_yolo")
detection_nn_yolo.out.link(xout_yolo.input)

xout_custom = pipeline.createXLinkOut()
xout_custom.setStreamName("detections_custom")
detection_nn_custom.out.link(xout_custom.input)

# --- Fin del pipeline ---
# En tu script principal usas device.getOutputQueue("detections_yolo") y "detections_custom" para cada modelo
