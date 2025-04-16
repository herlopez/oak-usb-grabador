import depthai as dai
import cv2
import time

# Crear la pipeline
pipeline = dai.Pipeline()

# Crear cámara RGB
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)

# Crear salida
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

# Ejecutar el dispositivo
with dai.Device(pipeline) as device:
    # Obtener la cola de salida
    q = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    # Configuración del grabador de video
    frame_width = 1920  # Ancho del frame (1080p)
    frame_height = 1080  # Alto del frame (1080p)
    fps = 10  # FPS de la cámara
    segment_duration = 15  # Duración del video en segundos (1 minuto)

    # Crear el archivo de video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    try:
        while True:
            # Captura el frame
            frame = q.get().getCvFrame()

            # Escribir el frame en el archivo de video
            out.write(frame)

            # Mostrar el frame (opcional)
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break

            # Verificar si ha pasado 1 minuto
            if time.time() - start_time >= segment_duration:
                print("Grabación de 1 minuto completada.")
                break
    finally:
        # Liberar recursos
        out.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente.")