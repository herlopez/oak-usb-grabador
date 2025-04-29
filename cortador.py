#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import subprocess
import datetime
import os
import time
import glob
import shutil

# Configuración
BASE_GRABACION_DIR = "/media/hlopez/76E8-CACF1/video/grabaciones"
BASE_CORTE_DIR = "/media/hlopez/76E8-CACF1/video/minutos"
DURACION_CORTE = 60  # 60 segundos
LOG_ERROR_PATH = "/media/hlopez/76E8-CACF1/video/errores.log"

archivo_actual = None

def registrar_error(error_message):
    """Registra un mensaje de error en el archivo errores.log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_ERROR_PATH, "a") as log_file:
        log_file.write(f"[ERROR] {timestamp} - {error_message}\n")
    print(f"[ERROR] {error_message}")

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
        "-y",
        "-ss", "00:00:00",
        "-i", archivo_grabacion,
        "-t", str(DURACION_CORTE),
        "-c:v", "libx264",  # Usa un códec de video como libx264
        "-preset", "fast",  # Opcional: puedes ajustar el preset
        salida_path
    ]

    result = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Verificar si el corte fue exitoso
    if result.returncode != 0:
        print(f"Error al ejecutar ffmpeg: {result.stderr.decode()}")
        error_message = f"Falló el corte del archivo {archivo_grabacion} en el minuto {timestamp_actual.strftime('%Y%m%d_%H%M%S')}"
        registrar_error(error_message)
        return None
    elif not os.path.exists(salida_path) or os.path.getsize(salida_path) < 1000:
        error_message = f"El archivo cortado {salida_path} es inválido (demasiado pequeño)."
        registrar_error(error_message)
        return None
    else:
        print(f"[OK] Corte creado: {salida_path}")
        return salida_path

def crear_thumbnail(video_path):
    """Crea una miniatura (thumbnail .jpg) a partir de un video."""
    thumbnail_path = video_path.replace(".mp4", ".jpg")

    comando = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", "00:00:01",  # Captura en el segundo 1
        "-vframes", "1",
        "-vf", "scale=320:-1",  # Escalar a 320px ancho, altura proporcional
        thumbnail_path
    ]

    result = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0 and os.path.exists(thumbnail_path):
        print(f"[OK] Thumbnail creado: {thumbnail_path}")
    else:
        error_message = f"Falló la creación del thumbnail para {video_path}"
        registrar_error(error_message)

def borrar_archivo(archivo):
    """Elimina un archivo de grabación."""
    if archivo and os.path.exists(archivo):
        os.remove(archivo)
        print(f"Archivo grande eliminado: {archivo}")

def verificar_espacio_disco():
    """Verifica el espacio disponible en disco y alerta si es menos del 10%."""
    total, usado, libre = shutil.disk_usage("/")
    porcentaje_libre = (libre / total) * 100

    if porcentaje_libre < 10:
        error_message = f"ALERTA: Espacio libre en disco bajo ({porcentaje_libre:.2f}% restante)"
        registrar_error(error_message)

if __name__ == "__main__":
    print("Iniciando corte continuo PRO cada minuto...")

    while True:
        ahora = datetime.datetime.now()
        print(f"[{ahora}] Esperando hasta el siguiente minuto...")
        siguiente_minuto = (ahora + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
        espera = (siguiente_minuto - ahora).total_seconds()
        time.sleep(espera)

        # Verificar espacio libre cada 5 minutos
        if ahora.minute % 5 == 0:
            verificar_espacio_disco()

        nuevo_archivo = encontrar_ultimo_archivo()

        # Si cambió el archivo (ejemplo: nueva hora), borrar anterior
        if archivo_actual and nuevo_archivo != archivo_actual:
            borrar_archivo(archivo_actual)

        archivo_actual = nuevo_archivo

        if archivo_actual:
            corte_path = cortar_minuto(archivo_actual, siguiente_minuto - datetime.timedelta(minutes=1))

            # Solo si se cortó bien, generamos el thumbnail
            if corte_path:
                crear_thumbnail(corte_path)
