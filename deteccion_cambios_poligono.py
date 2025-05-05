import cv2
import numpy as np
import depthai as dai

# Define pipeline OAK
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Zona poligonal ROI
roi_polygon = np.array([[200, 150], [400, 150], [450, 300], [250, 350]])

def create_mask(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

# Estado base
base_roi = None
mask = None

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        frame = video.get().getCvFrame()

        if mask is None:
            mask = create_mask(frame, roi_polygon)

        # Aplicar la máscara para extraer solo la ROI
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        if base_roi is None:
            base_roi = roi_frame.copy()
            print("Base ROI definida. Esperando cambios...")
        else:
            # Comparar ROI actual con base
            diff = cv2.absdiff(base_roi, roi_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            change_value = np.sum(gray_diff) / 255  # número de píxeles cambiados

            if change_value > 5000:  # Umbral de cambio (ajustable)
                print("⚠️ Cambio detectado en zona ROI:", change_value)

        # Dibujar la ROI en pantalla
        display = frame.copy()
        cv2.polylines(display, [roi_polygon], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("ZONA MONITOREADA", display)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('b'):
            base_roi = roi_frame.copy()
            print("🔄 Base ROI actualizada")

cv2.destroyAllWindows()
