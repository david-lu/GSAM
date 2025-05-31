import cv2
import traceback  # For detailed error messages in the example
import argparse  # For command-line arguments
import os  # For file operations
import numpy as np  # For creating the white background frame

# Define the video codec as a constant
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # Common codec for .mp4 files


def extract_cel_bounding_box(video_path, threshold_value=240, min_contour_area=100):
    """
    Extracts a single bounding box encompassing the maximum area where an animation cel
    appears in a video with a predominantly white background.

    Args:
        video_path (str): Path to the video file.
        threshold_value (int): Grayscale value (0-255) used for thresholding.
        min_contour_area (float): Minimum area (in pixels) for the largest detected contour.

    Returns:
        tuple: (x, y, width, height) of the aggregated bounding box.
               None if video can't be opened or no cel is detected.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    overall_min_x, overall_min_y = float('inf'), float('inf')
    overall_max_x, overall_max_y = 0, 0
    cel_detected_in_any_frame = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                overall_min_x = min(overall_min_x, x)
                overall_min_y = min(overall_min_y, y)
                overall_max_x = max(overall_max_x, x + w)
                overall_max_y = max(overall_max_y, y + h)
                cel_detected_in_any_frame = True

    cap.release()

    if not cel_detected_in_any_frame:
        print("No cel detected based on parameters.")
        return None

    final_width = overall_max_x - overall_min_x
    final_height = overall_max_y - overall_min_y

    if final_width <= 0 or final_height <= 0:
        print(f"Warning: Calculated bounding box has non-positive dimensions. W:{final_width}, H:{final_height}")
        return None

    return (int(overall_min_x), int(overall_min_y), int(final_width), int(final_height))


def crop_video_with_bounding_box(input_video_path, output_video_path, bounding_box):
    """
    Crops a video using a given bounding box and saves it.

    Args:
        input_video_path (str): Path to the source video.
        output_video_path (str): Path to save the cropped video.
        bounding_box (tuple): (x, y, width, height) for cropping.

    Returns:
        bool: True if successful, False otherwise.
    """
    x, y, w, h = bounding_box
    if w <= 0 or h <= 0:
        print(f"Error: Bounding box has invalid dimensions. W:{w}, H:{h}")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, VIDEO_FOURCC, original_fps, (w, h))

    if not out.isOpened():
        print(f"Error: Could not open video writer for: {output_video_path}")
        cap.release()
        return False

    print(f"Cropping video. Input: '{input_video_path}', Output: '{output_video_path}', Box: {bounding_box}")
    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y: y + h, x: x + w]
        out.write(cropped_frame)
        frames_processed += 1

    print(f"Successfully cropped video. Processed {frames_processed} frames.")
    cap.release()
    out.release()
    return True


def resize_video_contain_with_padding(input_video_path, output_video_path, target_width=512, target_height=512,
                                      padding_color=(255, 255, 255), alignment="bottom"):
    """
    Resizes a video to target dimensions using a "contain" strategy with alignment.
    The original content is scaled to fit entirely within the target dimensions,
    maintaining its aspect ratio. Specified color padding is added.

    Args:
        input_video_path (str): Path to the source video.
        output_video_path (str): Path to save the final resized video file.
        target_width (int): Desired width of the output video frames.
        target_height (int): Desired height of the output video frames.
        padding_color (tuple): BGR color for the padding (default is white).
        alignment (str): Alignment of the video within the padded frame.
                         Options: "center", "top", "bottom", "left", "right",
                                  "top-left", "top-right", "bottom-left", "bottom-right".
                         If a single direction (e.g., "bottom") is given,
                         the other axis defaults to "center". Default is "bottom".

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    if target_width <= 0 or target_height <= 0:
        print(f"Error: Target width ({target_width}) or height ({target_height}) is invalid.")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video file: {input_video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_w_orig == 0 or frame_h_orig == 0:
        print(f"Error: Could not read original frame dimensions from {input_video_path}")
        cap.release()
        return False

    out = cv2.VideoWriter(output_video_path, VIDEO_FOURCC, original_fps, (target_width, target_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for output file: {output_video_path}")
        cap.release()
        return False

    print(
        f"Resizing (contain with padding, align: {alignment}) video. Input: '{input_video_path}', Output: '{output_video_path}', Target: {target_width}x{target_height}")

    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale_w = target_width / frame_w_orig
        scale_h = target_height / frame_h_orig
        scale_factor = min(scale_w, scale_h)

        new_w = int(frame_w_orig * scale_factor)
        new_h = int(frame_h_orig * scale_factor)

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        background = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

        # Determine alignment offsets
        align_parts = alignment.lower().split('-')

        # Default vertical alignment: center (if not specified or part of a combined alignment)
        if "top" in align_parts:
            y_offset = 0
        elif "bottom" in align_parts:
            y_offset = target_height - new_h
        else:  # Default to center if no vertical alignment or just "center" is specified
            y_offset = (target_height - new_h) // 2

        # Default horizontal alignment: center (if not specified or part of a combined alignment)
        if "left" in align_parts:
            x_offset = 0
        elif "right" in align_parts:
            x_offset = target_width - new_w
        else:  # Default to center if no horizontal alignment or just "center" is specified
            x_offset = (target_width - new_w) // 2

        # Handle single "center" alignment
        if alignment.lower() == "center":
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2

        # Ensure offsets are not negative (can happen if new_w/h > target_w/h due to rounding, though unlikely with 'contain')
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        background[y_offset: y_offset + new_h, x_offset: x_offset + new_w] = resized_frame

        out.write(background)
        frames_processed += 1

    print(f"Successfully resized (contain with padding) video. Processed {frames_processed} frames.")
    cap.release()
    out.release()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts an animation cel, crops the video to it, and then resizes the cropped video with padding and alignment.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, required=True, help="Path for the final resized output video file.")
    parser.add_argument("--align", type=str, default="bottom",
                        choices=["center", "top", "bottom", "left", "right",
                                 "top-left", "top-right", "bottom-left", "bottom-right"],
                        help="Alignment of the video within the padded frame (default: bottom).")
    # You can add more optional arguments here for threshold, min_area, target_width, target_height, padding_color

    args = parser.parse_args()

    input_file_path = args.input
    final_output_file_path = args.output
    alignment_choice = args.align

    temp_dir = ".tmp"
    temp_filename = "cropped_video.mp4"
    temp_cropped_video_file_path = None

    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_cropped_video_file_path = os.path.join(temp_dir, temp_filename)

        print(f"\n--- Step 1: Processing video to find cel bounding box: {input_file_path} ---")
        bounding_box = extract_cel_bounding_box(input_file_path, threshold_value=230, min_contour_area=500)

        if bounding_box:
            x_bb, y_bb, w_bb, h_bb = bounding_box
            print(f"Detected overall cel bounding box: X={x_bb}, Y={y_bb}, Width={w_bb}, Height={h_bb}")

            print(f"\n--- Step 2: Cropping video to bounding box (temp file: {temp_cropped_video_file_path}) ---")
            success_crop = crop_video_with_bounding_box(input_file_path, temp_cropped_video_file_path, bounding_box)

            if success_crop:
                print(f"Video cropped to bounding box and saved to temporary file: '{temp_cropped_video_file_path}'")

                print(
                    f"\n--- Step 3: Resizing (contain with white padding, align: {alignment_choice}) the cropped video ---")
                success_resize = resize_video_contain_with_padding(
                    temp_cropped_video_file_path,
                    final_output_file_path,
                    alignment=alignment_choice
                )

                if success_resize:
                    print(f"Video resized successfully and saved to '{final_output_file_path}'")
                else:
                    print(f"Failed to resize video from '{temp_cropped_video_file_path}'.")
            else:
                print(f"Failed to crop video to bounding box. Skipping resize step.")
        else:
            print(f"Could not detect the cel's bounding box in '{input_file_path}'. Further processing skipped.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    # The temporary file at temp_cropped_video_file_path will persist after script execution.

