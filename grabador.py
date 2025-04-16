import depthai as dai
import cv2
import pygame
import numpy as np

# Inicializa pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))  # Ajusta el tamaño de la pantalla

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

    while True:
        # Captura el frame
        frame = q.get().getCvFrame()

        # Convierte el frame a formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convierte la imagen a superficie de pygame
        frame_surface = pygame.surfarray.make_surface(frame_rgb)

        # Mostrar el frame
        screen.blit(frame_surface, (0, 0))
        pygame.display.update()

        # Verifica si se presionó la tecla de salir
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

