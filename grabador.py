import depthai as dai
import cv2
import time
import os

usb_path = '/media/usb/videos/'
os.makedirs(usb_path, exist_ok=True)

segment_duration = 60  # 1 minutos en segundos

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    file_count = 0
    start_time = time.time()
    filename = os.path.join(usb_path, f"segment_{file_count}.avi")
    writer = None

    while True:
        frame = video_queue.get().getCvFrame()

        if writer is None:
            height, width, _ = frame.shape
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        writer.write(frame)

        if time.time() - start_time >= segment_duration:
            writer.release()
            file_count += 1
            filename = os.path.join(usb_path, f"segment_{file_count}.avi")
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
            start_time = time.time()

        # Puedes quitar esto si no tienes monitor conectado al Pi
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if writer:
        writer.release()
    cv2.destroyAllWindows()
