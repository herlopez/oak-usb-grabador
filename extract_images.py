import cv2
import os
from tkinter import Tk, filedialog

# Selecciona el archivo de video con un diálogo
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Selecciona el video",
    filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("No se seleccionó ningún archivo.")
    exit()

# Carpeta de salida fija
output_dir = r"C:\Planta101\rpi7\images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_number = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("Video Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Frame", 1024, 768)

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    display = frame.copy()
    # Número de frame
    cv2.putText(display, f"Frame: {frame_number+1}/{total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # Nombre del archivo
    cv2.putText(display, f"{os.path.basename(video_path)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    # Ruta completa del archivo
    cv2.putText(display, f"{video_path}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.imshow("Video Frame", display)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('p'):
        break
    elif key == ord('d'):  # avanzar 1 frame
        frame_number = min(frame_number + 1, total_frames - 1)
    elif key == ord('a'):  # retroceder 1 frame
        frame_number = max(frame_number - 1, 0)
    elif key == ord('e'):  # avanzar 300 frames
        frame_number = min(frame_number + 10, total_frames - 1)
    elif key == ord('s'):  # guardar imagen
        base = os.path.splitext(os.path.basename(video_path))[0]
        filename = os.path.join(output_dir, f"{base}_frame_{frame_number+1:05d}.png")
        cv2.imwrite(filename, display)
        print(f"Guardado: {filename}")

cap.release()
cv2.destroyAllWindows()