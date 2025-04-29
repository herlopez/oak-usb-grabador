#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
import os
import time

# Configuración
BASE_OUTPUT_DIR = "/media/hlopez/76E8-CACF1/video/grabaciones"
DURACION_SEGUNDOS_ARCHIVO = 300  # 1 hora

def grabar_archivo_continuo():
    """Graba video continuo en bloques de 1 hora."""
    while True:
        now = datetime.datetime.now()
        fecha_hora = now.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_OUTPUT_DIR, now.strftime("%Y%m%d"))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"grabacion_{fecha_hora}.h264")

        print(f"[{fecha_hora}] Grabando bloque continuo: {output_file}")

        comando = [
            "rpicam-vid",
            "-t", str(DURACION_SEGUNDOS_ARCHIVO * 1000),  # en milisegundos
            "-o", output_file,
            "--width", "1920",
            "--height", "1080",
            "--framerate", "10",
            "--profile", "high",
            "--bitrate", "5000000",   # Puedes ajustar bitrate si quieres
            "--nopreview"
        ]

        subprocess.run(comando)

if __name__ == "__main__":
    grabar_archivo_continuo()
