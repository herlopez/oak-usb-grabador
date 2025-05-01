import depthai as dai
import cv2
import numpy as np

# ROI fija (ajusta estos valores a tu necesidad)
x1, y1, x2, y2 = 200, 150, 300, 250  # ejemplo: ROI de 100x100
roi_selected = True  # Siempre activa

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

    background = None

    print("Presiona 'b' para guardar fondo. 'q' para salir.")

    while True:
        depthFrame = depthQueue.get().getFrame()
        depthVis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depthVis = cv2.applyColorMap(depthVis.astype(np.uint8), cv2.COLORMAP_JET)

        # Dibuja la ROI fija
        cv2.rectangle(depthVis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        key = cv2.waitKey(1)

        if roi_selected and background is not None:
            y_min, y_max = min(y1, y2), max(y1, y2)
            x_min, x_max = min(x1, x2), max(x1, x2)

            current_roi = depthFrame[y_min:y_max, x_min:x_max]
            background_roi = background[y_min:y_max, x_min:x_max]

            diff = cv2.absdiff(current_roi, background_roi)
            mask = (diff > 300).astype(np.uint8) * 255

            if np.sum(mask) > 100:
                cv2.putText(depthVis, "¡Movimiento detectado!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if key == ord('b'):
            background = depthFrame.copy()
            print("✅ Fondo capturado.")

        if key == ord('q'):
            break

        cv2.imshow("Depth", depthVis)

    cv2.destroyAllWindows()