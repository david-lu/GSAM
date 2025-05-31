#!/bin/bash

#INPUT_DIR="/home/chicken/Videos/split_adaptive/sword_in_the_stone/"
#INPUT_DIR="/home/chicken/Videos/input/"
#OUTPUT_DIR="/home/chicken/Videos/output"
#OUTPUT_CROPPED_DIR="/home/chicken/Videos/output_cropped"

INPUT_DIR="/home/chicken/Videos/split_adaptive/alice"
OUTPUT_DIR="/home/chicken/Videos/output/alice"
OUTPUT_CROPPED_DIR="/home/chicken/Videos/output/alice_cropped"

SCRIPT_DIR="/home/chicken/Documents/GitHub/GSAM"
MASK_SCRIPT="$SCRIPT_DIR/run_ground_continuous.py"
POST_PROCESS_SCRIPT="$SCRIPT_DIR/run_post_process.py"

echo "[INFO] Starting batch video processing."
mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*; do
    filename=$(basename "$video")
    output_file="$OUTPUT_DIR/$filename"

    echo "[INFO] Processing: $output_file"
    echo "        Output: $output_file"
    python3 "$MASK_SCRIPT" --input "$video" --output "$output_file"
    if [ $? -eq 0 ]; then
        echo "[INFO] Finished processing $filename"
    else
        echo "[ERROR] Failed to process $filename"
    fi

    output_cropped_file="$OUTPUT_CROPPED_DIR/$filename"
    echo "[INFO] Post Processing: $output_file"
    python3 "$POST_PROCESS_SCRIPT" --input "$output_file" --output "$output_cropped_file"
    if [ $? -eq 0 ]; then
        echo "[INFO] Finished post processing $filename"
    else
        echo "[ERROR] Failed to post process $filename"
    fi
done

echo "[INFO] Batch processing complete."