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

# Selecciona el punto superior izquierdo con el mouse
roi = cv2.selectROI("Selecciona la esquina superior izquierda y presiona ENTER", frame, showCrosshair=True)
x, y, _, _ = roi  # Ignora el tamaño seleccionado
cv2.destroyWindow("Selecciona la esquina superior izquierda y presiona ENTER")

# Fuerza el tamaño del ROI a 140x157 (puedes cambiarlo si quieres otro tamaño base)
w, h = 140, 157

# Ajusta si el ROI se sale de la imagen
if x + w > frame.shape[1]:
    x = frame.shape[1] - w
if y + h > frame.shape[0]:
    y = frame.shape[0] - h
if x < 0 or y < 0:
    print("El ROI está fuera de la imagen.")
    exit()

# Vuelve al primer frame para procesar todo el video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dibuja el ROI sobre el frame original (en rojo)
    frame_with_roi = frame.copy()
    cv2.rectangle(frame_with_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Recorta el frame al ROI de 140x157
    roi_frame = frame[y:y+h, x:x+w]

    # Redimensiona el ROI a 416x416 para la inferencia
    roi_resized = cv2.resize(roi_frame, (416, 416))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    results = model(roi_rgb)

    detected = False
    for *xyxy, conf, cls in results.xyxy[0]:
        # Las coordenadas están en el espacio 416x416, hay que reescalarlas al tamaño original del ROI para dibujar
        x1, y1, x2, y2 = map(int, xyxy)
        # Reescalado inverso para dibujar en el ROI original
        x1o = int(x1 * w / 416)
        y1o = int(y1 * h / 416)
        x2o = int(x2 * w / 416)
        y2o = int(y2 * h / 416)
        label = class_list[int(cls)] if int(cls) < len(class_list) else str(int(cls))
        cv2.rectangle(roi_frame, (x1o, y1o), (x2o, y2o), (0,255,0), 2)
        cv2.putText(roi_frame, f"{label} {conf:.2f}", (x1o, y1o-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        detected = True

    if detected:
        print("¡Detección realizada en este frame!")

    # Muestra el ROI recortado con detecciones
    cv2.imshow("Detección PC", roi_frame)
    # Muestra el frame original con el ROI pintado
    cv2.imshow("Video Original + ROI", frame_with_roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()