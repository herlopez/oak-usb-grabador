#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
import os
import time

# Crear carpeta de salida si no existe
output_dir = "/media/hlopez/76E8-CACF1/video"
os.makedirs(output_dir, exist_ok=True)

# Crear (o abrir) archivo de log
log_file_path = os.path.join(output_dir, "log.txt")

# Bucle infinito de grabacion
try:
    while True:
        # Obtener timestamp actual
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_mp4 = f"video_{timestamp}.mp4"
        output_path_mp4 = os.path.join(output_dir, output_file_mp4)

        # Definir el comando de grabacion (directo a MP4)
        record_command = [
            "rpicam-vid",
            "-t", "600000",  # 600000 ms = 10 minuto
            "-o", output_path_mp4,
            "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
            "--width", "1920",
            "--height", "1080",
            "--framerate", "10",
            "--nopreview"
        ]

        # Escribir en log antes de grabar
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp} - Grabando {output_file_mp4}\n")

        print(f"Iniciando grabacion: {output_file_mp4}")

        # Ejecutar grabacion
        subprocess.run(record_command, check=True)

        print(f"Grabacion finalizada: {output_file_mp4}")

        # Registrar en el log la grabacion directa
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp} - Grabacion directa a {output_file_mp4}\n")

        # Pequeña pausa antes de la siguiente grabacion
        time.sleep(1)

except KeyboardInterrupt:
    print("Grabacion continua detenida por el usuario.")
except subprocess.CalledProcessError as e:
    print(f"Ocurrio un error durante la grabacion: {e}")