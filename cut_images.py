import cv2
import os
from tkinter import Tk, filedialog

# Selecciona la imagen con un diálogo
Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Selecciona la imagen",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)

if not img_path:
    print("No se seleccionó ninguna imagen.")
    exit()

img = cv2.imread(img_path)
if img is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Selecciona el ROI (área de recorte) con el mouse
roi = cv2.selectROI("Selecciona el área y presiona ENTER", img, showCrosshair=True)
x, y, w, h = roi

# Recorta la imagen y guarda en carpeta "extract" en el mismo path
if w > 0 and h > 0:
    recorte = img[y:y+h, x:x+w]
    cv2.imshow("Recorte", recorte)
    # Carpeta "extract" en el mismo directorio de la imagen original
    base_dir = os.path.dirname(img_path)
    extract_dir = os.path.join(base_dir, "extract")
    os.makedirs(extract_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(img_path))
    new_name = f"{base}_cut{ext}"
    new_path = os.path.join(extract_dir, new_name)
    cv2.imwrite(new_path, recorte)
    print(f"\nNombre del archivo: {new_name}")
    print(f"Ruta completa: {os.path.abspath(new_path)}\n")
    cv2.waitKey(0)

cv2.destroyAllWindows()