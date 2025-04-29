#!/bin/bash

# Ruta base donde guardar los videos
BASE_DIR="/media/hlopez/76E8-CACF1/video/minutos"

# Crear loop infinito para grabar videos de 5 minutos
while true; do
  # Obtener fecha y hora actual
  DATE=$(date '+%Y%m%d')
  HOUR=$(date '+%H')
  TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

  # Crear carpeta si no existe
  mkdir -p "$BASE_DIR/$DATE/$HOUR"

  # Ruta de salida
  OUTPUT="$BASE_DIR/$DATE/$HOUR/video_${TIMESTAMP}.mp4"

  echo "[INFO] Grabando: $OUTPUT"

  # Grabar 5 minutos (300000 ms) y guardar en .mp4
  libcamera-vid -t 300000 --codec h264 --width 1920 --height 1080 --nopreview -o - | \
  ffmpeg -loglevel error -y -i - -c copy "$OUTPUT"

  echo "[INFO] Esperando siguiente grabación..."
done
