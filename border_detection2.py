#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Variables globales para la ROI y fondo
roi = None
drawing = False
ix, iy = -1, -1
background_roi = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pass  # Puedes agregar visualización dinámica si quieres
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

edgeDetectorLeft = pipeline.create(dai.node.EdgeDetector)
edgeDetectorRight = pipeline.create(dai.node.EdgeDetector)
edgeDetectorRgb = pipeline.create(dai.node.EdgeDetector)

xoutEdgeLeft = pipeline.create(dai.node.XLinkOut)
xoutEdgeRight = pipeline.create(dai.node.XLinkOut)
xoutEdgeRgb = pipeline.create(dai.node.XLinkOut)
xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

edgeLeftStr = "edge left"
edgeRightStr = "edge right"
edgeRgbStr = "edge rgb"
edgeCfgStr = "edge cfg"

xoutEdgeLeft.setStreamName(edgeLeftStr)
xoutEdgeRight.setStreamName(edgeRightStr)
xoutEdgeRgb.setStreamName(edgeRgbStr)
xinEdgeCfg.setStreamName(edgeCfgStr)

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())

# Linking
monoLeft.out.link(edgeDetectorLeft.inputImage)
monoRight.out.link(edgeDetectorRight.inputImage)
camRgb.video.link(edgeDetectorRgb.inputImage)

edgeDetectorLeft.outputImage.link(xoutEdgeLeft.input)
edgeDetectorRight.outputImage.link(xoutEdgeRight.input)
edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

xinEdgeCfg.out.link(edgeDetectorLeft.inputConfig)
xinEdgeCfg.out.link(edgeDetectorRight.inputConfig)
xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output/input queues
    edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 8, False)
    edgeRightQueue = device.getOutputQueue(edgeRightStr, 8, False)
    edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 8, False)
    edgeCfgQueue = device.getInputQueue(edgeCfgStr)

    print("Selecciona una ROI con el mouse en la ventana 'edge rgb'.")
    print("Presiona 'b' para guardar fondo, 'q' para salir.")

    cv2.namedWindow(edgeRgbStr)
    cv2.setMouseCallback(edgeRgbStr, draw_rectangle)

    while True:
        edgeLeft = edgeLeftQueue.get()
        edgeRight = edgeRightQueue.get()
        edgeRgb = edgeRgbQueue.get()

        edgeLeftFrame = edgeLeft.getFrame()
        edgeRightFrame = edgeRight.getFrame()
        edgeRgbFrame = edgeRgb.getFrame()

        # Mostrar la ROI si está seleccionada
        displayFrame = edgeRgbFrame.copy()
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(displayFrame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Detección de movimiento en la ROI
            if background_roi is not None:
                current_roi = displayFrame[y1:y2, x1:x2]
                diff = cv2.absdiff(current_roi, background_roi)
                mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
                if np.sum(mask) > 1000:
                    cv2.putText(displayFrame, "¡Movimiento!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(edgeLeftStr, edgeLeftFrame)
        cv2.imshow(edgeRightStr, edgeRightFrame)
        cv2.imshow(edgeRgbStr, displayFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Guardar fondo con 'b'
        if key == ord('b') and roi:
            x1, y1, x2, y2 = roi
            background_roi = displayFrame[y1:y2, x1:x2].copy()
            print("Fondo guardado.")

        if key == ord('1'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            sobelVerticalKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)

        if key == ord('2'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
            sobelVerticalKernel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)

    cv2.destroyAllWindows()