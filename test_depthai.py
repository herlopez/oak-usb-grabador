import depthai as dai

# Crear la pipeline
pipeline = dai.Pipeline()

# Crear cámara RGB
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(10)

# Crear salida
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

# Ejecutar el dispositivo
with dai.Device(pipeline) as device:
    # Obtener la cola de salida
    q = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    while True:
        # Captura el frame
        frame = q.get().getCvFrame()

        # Mostrar el frame
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
