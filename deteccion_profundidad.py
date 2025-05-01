import depthai as dai
import cv2
import numpy as np

# ROI fija: (x, y, w, h)
x, y, w, h = 1100, 300, 300, 300

# Parámetros de validación
FRAMES_PARA_DETECCION = 5  # Número de frames consecutivos para confirmar detección
contador_deteccion = 0
objeto_presente = False

# Crear pipeline de profundidad
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

    print("ROI fija en (1100, 300, 300, 300). Presiona 'q' para salir.")

    while True:
        depthFrame = depthQueue.get().getFrame()
        depthVis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depthVis = cv2.applyColorMap(depthVis.astype(np.uint8), cv2.COLORMAP_JET)

        # Dibuja la ROI fija
        cv2.rectangle(depthVis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extraer la ROI de profundidad
        roi_depth = depthFrame[y:y + h, x:x + w]
        valid_pixels = roi_depth[roi_depth > 0]
        if valid_pixels.size > 0:
            if np.mean(valid_pixels) < 2000:
                contador_deteccion += 1
            else:
                contador_deteccion = 0
        else:
            contador_deteccion = 0

        # Confirmar detección solo si se mantiene varios frames
        if contador_deteccion >= FRAMES_PARA_DETECCION and not objeto_presente:
            print("¡Objeto detectado en la ROI (profundidad)!")
            objeto_presente = True
        if contador_deteccion == 0:
            objeto_presente = False

        # Mostrar alerta en pantalla si objeto presente
        if objeto_presente:
            cv2.putText(depthVis, "¡Objeto en ROI!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Depth", depthVis)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()