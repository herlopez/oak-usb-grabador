#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
import os
import time
import glob

# Configuración
BASE_GRABACION_DIR = "/media/hlopez/76E8-CACF1/video/grabaciones"
BASE_CORTE_DIR = "/media/hlopez/76E8-CACF1/video/minutos"
DURACION_CORTE = 60  # 60 segundos

# Archivo fuente que estás usando
archivo_actual = None

def encontrar_ultimo_archivo():
    """Busca el archivo de grabación más reciente (.h264)."""
    archivos = glob.glob(f"{BASE_GRABACION_DIR}/**/*.h264", recursive=True)
    if archivos:
        return max(archivos, key=os.path.getctime)
    return None

def cortar_minuto(archivo_grabacion, timestamp_actual):
    """Corta un segmento de 60 segundos desde archivo fuente."""
    fecha_dir = timestamp_actual.strftime("%Y%m%d")
    hora_dir = timestamp_actual.strftime("%H")
    output_dir = os.path.join(BASE_CORTE_DIR, fecha_dir, hora_dir)
    os.makedirs(output_dir, exist_ok=True)

    salida_nombre = f"video_{timestamp_actual.strftime('%Y%m%d_%H%M%S')}.mp4"
    salida_path = os.path.join(output_dir, salida_nombre)

    comando = [
        "ffmpeg",
        "-y",  # Sobrescribir si existe
        "-ss", "00:00:00", 
        "-i", archivo_grabacion,
        "-t", str(DURACION_CORTE),
        "-c", "copy",  # No recodificar
        salida_path
    ]

    subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Corte creado: {salida_path}")

def borrar_archivo(archivo):
    """Elimina un archivo de grabación."""
    if archivo and os.path.exists(archivo):
        os.remove(archivo)
        print(f"Archivo grande eliminado: {archivo}")

if __name__ == "__main__":
    print("Iniciando corte continuo cada minuto...")

    while True:
        ahora = datetime.datetime.now()
        siguiente_minuto = (ahora + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
        espera = (siguiente_minuto - ahora).total_seconds()
        time.sleep(espera)

        nuevo_archivo = encontrar_ultimo_archivo()

        # Si cambió el archivo (ejemplo: pasó de una hora a otra), elimina el anterior
        if archivo_actual and nuevo_archivo != archivo_actual:
            borrar_archivo(archivo_actual)

        # Actualiza archivo actual
        archivo_actual = nuevo_archivo

        # Cortar minuto actual
        if archivo_actual:
            cortar_minuto(archivo_actual, siguiente_minuto - datetime.timedelta(minutes=1))
