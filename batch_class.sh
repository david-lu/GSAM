#!/bin/bash

# Source directory containing all files
SOURCE_DIR=~/Videos/vqvae

# Make sure source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

echo "Organizing files into class folders..."

# Process each file in the source directory
for file in "$SOURCE_DIR"/*; do
    # Skip if not a file
    [ -f "$file" ] || continue
    
    # Get just the filename without the path
    filename=$(basename "$file")
    
    # Extract class name (everything before pattern of [number]_post_processed)
    # For example, from "anastasia_0065_post_processed_frame_0007", extract "anastasia"
    class_name=$(echo "$filename" | sed -E 's/(.*)_[0-9]+_post_processed.*/\1/')
    
    # Create the class directory if it doesn't exist
    class_dir="$SOURCE_DIR/$class_name"
    mkdir -p "$class_dir"
    
    # Move the file to its class directory
    echo "Moving $filename to $class_name/"
    mv "$file" "$class_dir/"
done

echo "Organization complete. Files have been moved to their class folders."

# Display summary of created folders
echo -e "\nSummary of created class folders:"
ls -la "$SOURCE_DIR" | grep "^d"