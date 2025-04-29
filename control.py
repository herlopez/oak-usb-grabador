#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import subprocess
import time
import os

def ejecutar_grabador():
    """Ejecuta el grabador de video continuo."""
    try:
        print("Iniciando grabador...")
        subprocess.Popen(["python3", "grabadordevideo.py"])  # Asegúrate de que el nombre de tu archivo sea correcto
        print("Grabador iniciado.")
    except Exception as e:
        print(f"Error al iniciar el grabador: {e}")
        with open("errores.log", "a") as log_file:
            log_file.write(f"Error al iniciar el grabador: {e}\n")

def ejecutar_cortador():
    """Ejecuta el cortador de video."""
    try:
        print("Iniciando cortador...")
        subprocess.Popen(["python3", "cortadordevideo.py"])  # Asegúrate de que el nombre de tu archivo sea correcto
        print("Cortador iniciado.")
    except Exception as e:
        print(f"Error al iniciar el cortador: {e}")
        with open("errores.log", "a") as log_file:
            log_file.write(f"Error al iniciar el cortador: {e}\n")

def main():
    print("Iniciando grabación y corte en paralelo...")
    ejecutar_grabador()  # Inicia el grabador en segundo plano
    time.sleep(2)  # Espera un poco para asegurar que el grabador esté en ejecución

    # Verificar si el grabador se está ejecutando correctamente
    if not any(p.name() == "python3" and "grabadordevideo.py" in p.cmdline() for p in subprocess.process_iter()):
        print("Error: El grabador no se inició correctamente.")
        with open("errores.log", "a") as log_file:
            log_file.write("Error: El grabador no se inició correctamente.\n")
        return

    ejecutar_cortador()  # Inicia el cortador en segundo plano

    try:
        while True:
            time.sleep(60)  # Mantiene el script principal corriendo sin hacer nada más
    except KeyboardInterrupt:
        print("Proceso detenido por el usuario.")
        with open("errores.log", "a") as log_file:
            log_file.write("Proceso detenido por el usuario.\n")

if __name__ == "__main__":
    main()
