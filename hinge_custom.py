import depthai as dai
import cv2
import numpy as np

# Crear el pipeline
pipeline = dai.Pipeline()

# Nodo de cámara RGB
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 640)  # Usa la resolución de entrenamiento de tu modelo
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Nodo de red neuronal
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("best_openvino_2022.1_6shave.blob")  # Ruta al archivo .blob
nn.input.setBlocking(False)
nn.input.setQueueSize(1)

# Conectar la cámara al modelo
camRgb.preview.link(nn.input)

# Salida de la cámara para mostrarla con OpenCV
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Salida del modelo
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# Ejecutar en el dispositivo
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    nnQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        inRgb = rgbQueue.get()
        inNN = nnQueue.get()

        frame = inRgb.getCvFrame()
        detections = inNN.getFirstLayerFp16()

        # Aquí deberías interpretar el resultado según el modelo que exportaste
        # Este paso cambia si usaste YOLO, MobileNet, etc.
        # Por ejemplo, puedes decodificar bounding boxes si sabes el layout

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) == ord('q'):
            break
