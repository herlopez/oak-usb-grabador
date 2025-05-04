import torch
import cv2
import matplotlib.pyplot as plt
import pathlib

# Parchear PosixPath para sistemas Windows
pathlib.PosixPath = pathlib.WindowsPath

# Cargar el modelo YOLOv5 desde un archivo .pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Ruta de la imagen a procesar
img_path = r'C:\Users\herlo\Downloads\Anotations_Hinge_Small\images\train\output_20250429_121830_frame_01629_cut.png'
img = cv2.imread(img_path)

# Convertir la imagen de BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Hacer predicción sobre la imagen
results = model(img_rgb)

# Mostrar los resultados en una ventana de OpenCV
results.show()

# Imprimir resultados en consola
print(results.xyxy[0])         # Coordenadas de las cajas
print(results.names)           # Clases detectadas
print(results.pandas().xywh)   # Formato pandas con información detallada

# Guardar la imagen con las detecciones
results.save()  # Guarda la imagen en runs/detect/exp

# Mostrar la imagen con matplotlib (usa 'ims' en vez de 'imgs')
if hasattr(results, 'ims') and len(results.ims) > 0:
    plt.imshow(results.ims[0])
    plt.axis('off')
    plt.show()
else:
    print("No hay imagen procesada para mostrar.")