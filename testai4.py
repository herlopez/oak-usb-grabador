#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
import os
import time
import shutil

# CONFIGURACIÓN
BASE_OUTPUT_DIR = "/home/hlopez/Videos"
SEGMENT_DURATION_MS = 300000  # 10 minuto
MIN_DISK_FREE_GB = 5          # Espacio mínimo libre en GB para seguir grabando
REINTENTOS_GRABACION = 3      # Número de reintentos si falla grabación
ANCHO = "1920"
ALTO = "1080"
FRAMERATE = "10"

# FUNCIONES

def esperar_hasta_proximo_multiplo_5():
    """Espera hasta el próximo múltiplo de 5 minutos (00, 05, 10, ...)."""
    now = datetime.datetime.now()
    minutos = now.minute
    segundos = now.second
    microsegundos = now.microsecond
    minutos_a_sumar = (5 - (minutos % 5)) % 5
    if minutos_a_sumar == 0 and (segundos > 0 or microsegundos > 0):
        minutos_a_sumar = 5
    proximo = (now + datetime.timedelta(minutes=minutos_a_sumar)).replace(second=0, microsecond=0)
    espera = (proximo - now).total_seconds()
    print(f"Esperando {espera:.2f} segundos para sincronizar al próximo múltiplo de 5 minutos...")
    time.sleep(espera)

def espacio_disponible_gb(path):
    """Calcula espacio libre en GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)

def escribir_log(mensaje):
    """Escribe en el log diario."""
    fecha_log = datetime.datetime.now().strftime("%Y%m%d")
    log_file_path = os.path.join(BASE_OUTPUT_DIR, f"log_{fecha_log}.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as f:
        f.write(f"[{timestamp}] {mensaje}\n")

def grabar_segmento(output_path, duration_ms):
    """Ejecuta rpicam-vid para grabar un segmento."""
    comando = [
        "rpicam-vid",
        "-t", str(duration_ms),
        "-o", output_path,
        "--width", ANCHO,
        "--height", ALTO,
        "--framerate", FRAMERATE,
        "--nopreview"
    ]
    return subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def crear_directorio_salida(base_dir):
    """Crea la carpeta de salida basada en fecha/hora."""
    now = datetime.datetime.now()
    fecha = now.strftime("%Y%m%d")
    hora = now.strftime("%H")
    ruta = os.path.join(base_dir, fecha, hora)
    os.makedirs(ruta, exist_ok=True)
    return ruta

# PROCESO PRINCIPAL

try:
    print("Inicializando grabación continua...")
    escribir_log("Inicio del sistema de grabación continua.")

    esperar_hasta_proximo_multiplo_5()  # Espera antes de la primera grabación

    while True:
        now = datetime.datetime.now()

        # Verificar espacio en disco
        espacio_gb = espacio_disponible_gb(BASE_OUTPUT_DIR)
        if espacio_gb < MIN_DISK_FREE_GB:
            mensaje = f"Espacio insuficiente: {espacio_gb:.2f} GB libres. Pausando grabaciones."
            print(mensaje)
            escribir_log(mensaje)
            time.sleep(60)  # Esperar un minuto antes de reintentar
            continue

        # Preparar carpeta y nombre de archivo
        output_dir = crear_directorio_salida(BASE_OUTPUT_DIR)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_file = f"video_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_file)

        escribir_log(f"Iniciando grabación: {output_file}")
        print(f"Iniciando grabación: {output_file}")

        # Intentar grabar con reintentos
        exito = False
        for intento in range(REINTENTOS_GRABACION):
            resultado = grabar_segmento(output_path, SEGMENT_DURATION_MS)
            if resultado.returncode == 0:
                exito = True
                break
            else:
                escribir_log(f"Error en grabación. Reintentando ({intento+1}/{REINTENTOS_GRABACION})...")
                time.sleep(2)  # Pequeña pausa antes de reintentar

        if exito:
            escribir_log(f"Grabación exitosa: {output_file}")
            print(f"Grabación finalizada: {output_file}")
        else:
            escribir_log(f"Grabación fallida: {output_file}")
            print(f"Error: grabación fallida después de {REINTENTOS_GRABACION} intentos.")

except KeyboardInterrupt:
    print("\nGrabación detenida manualmente.")
    escribir_log("Sistema detenido manualmente por el usuario.")
except Exception as e:
    print(f"\nError fatal: {e}")
    escribir_log(f"Error fatal: {e}")
