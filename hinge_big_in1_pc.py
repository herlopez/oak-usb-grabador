import cv2
import numpy as np

# Paso 1: Abrir video grabado
cap = cv2.VideoCapture(r'C:\Planta101\rpi7\20250429\14\output_20250429_145501.mp4')

# Paso 2: Zona que quieres recortar del video (por ejemplo, centro)
RECORTADO = (1200, 280, 500, 500)  # x, y, ancho, alto
ZONA_ALERTA = (100, 150, 300, 300)  # dentro del recorte

trayectoria = []
MIN_MOVIMIENTO = 30  # píxeles mínimos de movimiento para considerar que se mueve

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 3: Recortar la zona que quieres analizar
    x, y, w, h = RECORTADO
    zona = frame[y:y+h, x:x+w]

    # Paso 4: Buscar contornos (rectángulos)
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    # Prueba con OTSU para ajustar el umbral automáticamente
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja todos los contornos para depuración
    cv2.drawContours(zona, contours, -1, (255, 0, 0), 1)

    objeto_en_movimiento = False

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > 200:  # prueba con 200 si 500 es mucho
            (rx, ry, rw, rh) = cv2.boundingRect(c)
            if rw > 30 and rh > 30:  # prueba con 30 si 50 es mucho
                centro = (rx + rw//2, ry + rh//2)
                # Solo guarda el centroide si realmente se mueve
                if not trayectoria or np.linalg.norm(np.array(centro) - np.array(trayectoria[-1])) > 2:
                    trayectoria.append(centro)
                cv2.rectangle(zona, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                zx, zy, zw, zh = ZONA_ALERTA
                # Solo alerta si el objeto se mueve y viene de la izquierda
                if len(trayectoria) > 5:
                    x_ini, y_ini = trayectoria[0]
                    x_fin, y_fin = trayectoria[-1]
                    mov = abs(x_fin - x_ini)
                    viene_izq = x_ini < 50  # cerca del borde izquierdo del recorte
                    if mov > MIN_MOVIMIENTO and viene_izq:
                        if zx <= centro[0] <= zx+zw and zy <= centro[1] <= zy+zh:
                            print("⚠️ Objeto en movimiento ha entrado en la zona de alerta")
                            objeto_en_movimiento = True

    # Limpiar trayectoria si no hay movimiento relevante
    if not objeto_en_movimiento:
        trayectoria = []

    cv2.rectangle(zona, (ZONA_ALERTA[0], ZONA_ALERTA[1]),
                  (ZONA_ALERTA[0]+ZONA_ALERTA[2], ZONA_ALERTA[1]+ZONA_ALERTA[3]),
                  (0, 0, 255), 2)

    # Ventanas de depuración
    cv2.imshow("Zona Recortada", zona)
    cv2.imshow("Gray", gray)
    cv2.imshow("Thresh", thresh)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()