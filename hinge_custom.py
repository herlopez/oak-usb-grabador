import depthai as dai
import cv2
import numpy as np

# Carga de clases
with open("classes.txt", "r") as f:
    class_list = f.read().splitlines()

# Parámetros de entrada
input_size = 416  # ajusta a lo que usaste en entrenamiento

# Crear pipeline
pipeline = dai.Pipeline()

# Cámara RGB
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(input_size, input_size)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Red neuronal
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("best_openvino_2022.1_6shave.blob")
cam.preview.link(nn.input)

# Salidas
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
cam.preview.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# Ejecutar en el dispositivo
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue("rgb", 4, False)
    nnQueue = device.getOutputQueue("nn", 4, False)

    while True:
        inRgb = rgbQueue.get()
        inNN = nnQueue.get()

        frame = inRgb.getCvFrame()
        nn_data = inNN.getFirstLayerFp16()

        # YOLOv5 normalmente produce 25200 predicciones (por defecto en 640x640)
        # Cada predicción tiene al menos 85 valores (x, y, w, h, obj_conf, clases...)
        predictions = np.array(nn_data).reshape((1, -1, len(class_list) + 5))[0]

        for det in predictions:
            conf = det[4]
            if conf > 0.4:  # umbral de confianza
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                score = class_scores[class_id]
                if score > 0.4:
                    label = class_list[class_id]
                    x, y, w, h = det[0:4]

                    # YOLOv5 coords están normalizados (0-1), escala al tamaño real
                    x = int((x - w / 2) * input_size)
                    y = int((y - h / 2) * input_size)
                    w = int(w * input_size)
                    h = int(h * input_size)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("OAK-D YOLOv5", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
