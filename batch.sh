#!/bin/bash

# Default settings
OVERRIDE=false

#INPUT_DIR="/home/chicken/Videos/split_adaptive/sword_in_the_stone/"
#INPUT_DIR="/home/chicken/Videos/input/"
#OUTPUT_DIR="/home/chicken/Videos/output"

INPUT_DIR="/home/chicken/Videos/split_adaptive/cinderella/"
OUTPUT_DIR="/home/chicken/Videos/output/cinderella"

SCRIPT_DIR="/home/chicken/Documents/GitHub/GSAM"
MASK_SCRIPT="$SCRIPT_DIR/run_ground_continuous.py"

echo "[INFO] Starting batch video processing."
echo "[INFO] Override mode: $OVERRIDE"
mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*; do
    filename=$(basename "$video")
    output_file="$OUTPUT_DIR/$filename"

    # Check if output file already exists and override is not enabled
    if [ -f "$output_file" ] && [ "$OVERRIDE" = false ]; then
        echo "[INFO] Skipping $filename - output file already exists"
        continue
    fi

    echo "[INFO] Processing: $output_file"
    echo "        Output: $output_file"
    python3 "$MASK_SCRIPT" --input "$video" --output "$output_file"
    if [ $? -eq 0 ]; then
        echo "[INFO] Finished processing $filename"
    else
        echo "[ERROR] Failed to process $filename"
    fi
done

echo "[INFO] Batch processing complete."