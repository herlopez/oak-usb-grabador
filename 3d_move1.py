import depthai as dai
import cv2
import numpy as np

# ROI fija (ajusta estos valores a tu necesidad)
x1, y1, x2, y2 = 50, 50, 300, 300  # ejemplo: ROI de 100x100

# Crear pipeline
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    cv2.namedWindow("Depth")

    # Obtener solo un frame
    depthFrame = depthQueue.get().getFrame()
    depthVis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
    depthVis = cv2.applyColorMap(depthVis.astype(np.uint8), cv2.COLORMAP_JET)

    # Dibuja la ROI fija
    cv2.rectangle(depthVis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("Mostrando imagen inicial con ROI. Presiona cualquier tecla para salir.")
    cv2.imshow("Depth", depthVis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()