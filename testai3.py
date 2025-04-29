#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
import os
import time

# Directorio base de salida
base_output_dir = "/media/hlopez/76E8-CACF1/video"

# Crear (o abrir) archivo de log general
log_file_path = os.path.join(base_output_dir, "log.txt")

def esperar_hasta_proximo_minuto():
    now = datetime.datetime.now()
    segundos_hasta_proximo_minuto = 60 - now.second - now.microsecond / 1_000_000
    if segundos_hasta_proximo_minuto > 0:
        print(f"Esperando {segundos_hasta_proximo_minuto:.2f} segundos para iniciar en el siguiente minuto exacto...")
        time.sleep(segundos_hasta_proximo_minuto)

# Esperar hasta el próximo minuto exacto antes de iniciar el bucle (solo al inicio)
esperar_hasta_proximo_minuto()

segment_duration_ms = 60000      # 1 minuto
overlap_ms = 100                # 1 segundo de solapamiento

try:
    proceso_anterior = None
    while True:
        now = datetime.datetime.now()
        fecha = now.strftime("%Y%m%d")
        hora = now.strftime("%H")

        # Crear carpetas por día y hora
        output_dir = os.path.join(base_output_dir, fecha, hora)
        os.makedirs(output_dir, exist_ok=True)

        # Nombre del archivo con timestamp (inicio de grabación)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_file_mp4 = f"video_{timestamp}.mp4"
        output_path_mp4 = os.path.join(output_dir, output_file_mp4)

        # Definir el comando de grabacion
        record_command = [
            "rpicam-vid",
            "-t", str(segment_duration_ms),
            "-o", output_path_mp4,
            # "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
            "--width", "1920",
            "--height", "1080",
            "--framerate", "10",
            "--nopreview"
        ]

        # Escribir en log antes de grabar
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{timestamp} - Grabando {output_file_mp4} en {output_dir}\n")

        print(f"Iniciando grabacion: {output_file_mp4}")

        # Iniciar la grabación en segundo plano
        proceso = subprocess.Popen(record_command)

        # Esperar hasta (duración - solapamiento)
        time.sleep((segment_duration_ms - overlap_ms) / 1000)

        # Esperar a que termine la grabación anterior (si existe)
        if proceso_anterior is not None:
            proceso_anterior.wait()
            print(f"Grabacion finalizada: {output_file_mp4}")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{timestamp} - Grabacion directa a {output_file_mp4}\n")

        # El proceso actual será el anterior en la próxima vuelta
        proceso_anterior = proceso

except KeyboardInterrupt:
    print("Grabacion continua detenida por el usuario.")
except subprocess.CalledProcessError as e:
    print(f"Ocurrio un error durante la grabacion: {e}")