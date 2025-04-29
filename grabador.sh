#!/bin/bash

# Ruta base
BASE_DIR="/media/hlopez/76E8-CACF1/video/minutos"

# Duración del video en segundos (5 minutos)
DURACION=300

while true; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FECHA=$(date +"%Y%m%d")
    HORA=$(date +"%H")
    DIR="$BASE_DIR/$FECHA/$HORA"

    mkdir -p "$DIR"

    ARCHIVO="$DIR/video_${TIMESTAMP}.mp4"
    echo "[INFO] Grabando: $ARCHIVO"

    libcamera-vid -t $((DURACION * 1000)) --width 1920 --height 1080 --codec libav --libav-format mp4 -o "$ARCHIVO"

    echo "[INFO] Esperando siguiente grabación..."
done
