import torch
import cv2
import pathlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Parchear PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Cargar clases
with open("classes.txt", "r") as f:
    class_list = f.read().splitlines()

video_path = r"C:\Planta101\rpi7\output_20250430_095514.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: No se pudo abrir el video.")
    exit()

cv2.namedWindow("Detección PC", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección PC", 800, 600)

# Leer el primer frame para seleccionar el ROI
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el primer frame del video.")
    exit()

roi = cv2.selectROI("Selecciona el área y presiona ENTER", frame, showCrosshair=True)
x, y, w, h = roi
cv2.destroyWindow("Selecciona el área y presiona ENTER")

if w <= 0 or h <= 0:
    print("Área seleccionada inválida.")
    exit()

# Vuelve al primer frame para procesar todo el video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recorta el frame al ROI antes de la inferencia
    roi_frame = frame[y:y+h, x:x+w]

    roi_resized = cv2.resize(roi_frame, (416, 416))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    results = model(roi_rgb)

    print(results.xyxy[0])  # 👈 Aquí imprimes los resultados crudos

    detected = False
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        label = class_list[int(cls)] if int(cls) < len(class_list) else str(int(cls))
        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(roi_frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        detected = True

    if detected:
        print("¡Detección realizada en este frame!")

    cv2.imshow("Detección PC", roi_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()