#!/bin/bash

# Ruta base de almacenamiento
BASE_DIR="/media/hlopez/76E8-CACF1/video/minutos"

# Crear carpeta del día actual
FECHA=$(date +"%Y%m%d")
HORA=$(date +"%H")
MINUTOS=$(date +"%M")

DIR_HORA="$BASE_DIR/$FECHA/$HORA"
mkdir -p "$DIR_HORA"

# Nombre del archivo con marca de tiempo
ARCHIVO="video_${FECHA}_${HORA}${MINUTOS}.mp4"
RUTA_ARCHIVO="$DIR_HORA/$ARCHIVO"

echo "[INFO] Grabando: $RUTA_ARCHIVO"

# Comando de grabación: 5 minutos (300000 ms), directo en mp4
libcamera-vid \
    -t 300000 \
    --width 1920 \
    --height 1080 \
    --codec libav \
    --libav-format mp4 \
    -o "$RUTA_ARCHIVO"

echo "[INFO] Grabación terminada: $RUTA_ARCHIVO"
