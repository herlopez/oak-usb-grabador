#!/bin/bash

# Ruta base de salida
BASE_DIR="/media/hlopez/76E8-CACF1/video/minutos"
mkdir -p "$BASE_DIR"

# Loop infinito para grabar videos de 5 minutos
while true; do
    FECHA=$(date +%Y%m%d)
    HORA=$(date +%H)
    MINUTO=$(date +%M)
    SEGUNDO=$(date +%S)
    TIMESTAMP="${FECHA}_${HORA}${MINUTO}${SEGUNDO}"

    DIR_SALIDA="${BASE_DIR}/${FECHA}/${HORA}"
    mkdir -p "$DIR_SALIDA"

    ARCHIVO_SALIDA="${DIR_SALIDA}/video_${TIMESTAMP}.mp4"
    echo "[INFO] Grabando: $ARCHIVO_SALIDA"

    # Graba por 5 minutos (300000 ms) sin mostrar logs
    libcamera-vid \
        -t 300000 \
        --width 1920 --height 1080 \
        --codec h264 -o - \
        --nopreview 2>/dev/null | \
    ffmpeg -hide_banner -loglevel error -f h264 -i - -c copy "$ARCHIVO_SALIDA"

    echo "[INFO] Esperando siguiente grabación..."
done
