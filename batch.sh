#!/bin/bash

#INPUT_DIR="/home/chicken/Videos/split_adaptive/sword_in_the_stone/"
INPUT_DIR="/home/chicken/Videos/input/"
OUTPUT_DIR="/home/chicken/Videos/output"
SCRIPT_DIR="/home/chicken/Documents/GitHub/GSAM"
PYTHON_SCRIPT="$SCRIPT_DIR/run_ground.py"

echo "[INFO] Starting batch video processing."
mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*; do
    filename=$(basename "$video")
    output_file="$OUTPUT_DIR/$filename"
    echo "[INFO] Processing: $video"
    echo "        Output: $output_file"
    python3 "$PYTHON_SCRIPT" --input "$video" --output "$output_file"
    if [ $? -eq 0 ]; then
        echo "[INFO] Finished processing $filename"
    else
        echo "[ERROR] Failed to process $filename"
    fi
done

echo "[INFO] Batch processing complete."