import cv2
import os
from tkinter import Tk, filedialog

# Selecciona una imagen con un diálogo
Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Selecciona una imagen para definir la carpeta",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)

if not img_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# Carpeta donde buscar imágenes
base_dir = os.path.dirname(img_path)
extract_dir = os.path.join(base_dir, "extract")
os.makedirs(extract_dir, exist_ok=True)

# Lista de imágenes en el directorio
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imagenes = [f for f in os.listdir(base_dir) if f.lower().endswith(exts)]

if not imagenes:
    print("No se encontraron imágenes en el directorio.")
    exit()

# Selecciona el ROI solo una vez usando la primera imagen
img = cv2.imread(img_path)
if img is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

roi = cv2.selectROI("Selecciona el área y presiona ENTER", img, showCrosshair=True)
x, y, w, h = roi

if w <= 0 or h <= 0:
    print("Área seleccionada inválida. No se guardó ninguna imagen.")
    exit()

# Procesa todas las imágenes del directorio
for nombre in imagenes:
    ruta = os.path.join(base_dir, nombre)
    img = cv2.imread(ruta)
    if img is None:
        print(f"No se pudo cargar la imagen: {ruta}")
        continue
    if y+h > img.shape[0] or x+w > img.shape[1]:
        print(f"El ROI excede el tamaño de la imagen: {nombre}")
        continue
    recorte = img[y:y+h, x:x+w]
    base, ext = os.path.splitext(nombre)
    new_name = f"{base}_cut{ext}"
    new_path = os.path.join(extract_dir, new_name)
    cv2.imwrite(new_path, recorte)
    print(f"Guardado: {os.path.abspath(new_path)}")

print("\nProceso terminado.")
cv2.destroyAllWindows()