#!/bin/bash

INPUT_DIR=~/Videos/output/robin_hood_continuous
TEMP_DIR=~/Videos/output_cropped
OUTPUT_DIR=~/Videos/vqvae

# Create output and temporary directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Process each video file
for video_file in "$INPUT_DIR"/*.mp4 "$INPUT_DIR"/*.avi "$INPUT_DIR"/*.mov "$INPUT_DIR"/*.mkv; do
    # Skip if no files match the pattern
    [ -e "$video_file" ] || continue

    echo "$video_file"
    
    # Get the filename without extension
    filename=$(basename -- "$video_file")
    filename_noext="${filename%.*}"
    
    echo "Processing: $filename"
    
    # Step 1: Run post-processing on the video
    echo "  Step 1: Running post-processing..."
    post_processed_file="$TEMP_DIR/${filename_noext}_post_processed.mp4"
    
    # Randomly select an alignment option
    aligns=("center" "top" "bottom")
    random_index=$(($RANDOM % ${#aligns[@]}))
    align_value=${aligns[$random_index]}
    echo "  Using alignment: $align_value"
    
    python run_post_process.py \
        --input "$video_file" \
        --output "$post_processed_file" \
        --align "$align_value"

    # Check if post-processing was successful
    if [ ! -f "$post_processed_file" ]; then
        echo "  Error: Post-processing failed for $filename. Skipping to next video."
        continue
    fi
    
    # Step 2: Run ground continuous on the post-processed video
    echo "  Step 2: Running ground continuous processing..."
    python run_convert_to_image.py \
        --video_path "$post_processed_file" \
        --output_dir "$OUTPUT_DIR"
    
    echo "Completed: $filename -> ${filename_noext}_processed.mp4"
done
