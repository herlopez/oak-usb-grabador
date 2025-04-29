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
        output_file_264 = f"video_{timestamp}.264"
        output_file_mp4 = f"video_{timestamp}.mp4"
        output_path_264 = os.path.join(output_dir, output_file_264)
        output_path_mp4 = os.path.join(output_dir, output_file_mp4)

        # Definir el comando de grabacion
        record_command = [
            "rpicam-vid",
            "-t", "60000",  # 60000 ms = 1 minuto
            "-o", output_path_264,
            "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
            "--width", "1920",
            "--height", "1080",
            "--framerate", "30",
            "--nopreview"
        ]

        # Escribir en log antes de grabar
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp} - Grabando {output_file_264}\n")

        print(f"Iniciando grabacion: {output_file_264}")

        # Ejecutar grabacion
        subprocess.run(record_command, check=True)

        print(f"Grabacion finalizada: {output_file_264}")

        # Convertir .264 a .mp4 usando ffmpeg
        convert_command = [
            "ffmpeg",
            "-framerate", "30",
            "-i", output_path_264,
            "-c:v", "libx264",
            "-preset", "fast",
            output_path_mp4
        ]

        print(f"Convirtiendo a MP4: {output_file_mp4}")
        subprocess.run(convert_command, check=True)

        # (Opcional) Borrar el archivo .264 para ahorrar espacio
        if os.path.exists(output_path_264):
            os.remove(output_path_264)
            print(f"Archivo .264 eliminado: {output_file_264}")

        # Registrar en el log la conversion
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp} - Convertido a {output_file_mp4}\n")

        # Peque�a pausa antes de la siguiente grabacion
        time.sleep(1)

except KeyboardInterrupt:
    print("Grabacion continua detenida por el usuario.")
except subprocess.CalledProcessError as e:
    print(f"Ocurrio un error durante la grabacion o conversion: {e}")
