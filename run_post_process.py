import cv2
import traceback
import argparse
import os
import numpy as np
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the video codec as a constant for the FINAL output
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')


def extract_cel_bounding_box(video_path, threshold_value=240, min_contour_area=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
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
        logger.warning("No cel detected based on parameters.")
        return None
    final_width = overall_max_x - overall_min_x
    final_height = overall_max_y - overall_min_y
    if final_width <= 0 or final_height <= 0:
        logger.warning(f"Calculated bounding box has non-positive dimensions. W:{final_width}, H:{final_height}")
        return None
    return (int(overall_min_x), int(overall_min_y), int(final_width), int(final_height))


def crop_video_with_bounding_box(input_video_path, output_video_path, bounding_box, threshold_value_for_masking):
    """
    Crops a video using a given bounding box. For each frame, it identifies
    the cel within the cropped region (using threshold_value_for_masking)
    and places it on a new pure white background of the bounding box dimensions.
    This function will use 'MJPG' codec for the output to better preserve pure white.

    Args:
        input_video_path (str): Path to the source video.
        output_video_path (str): Path to save the cropped video (temporary file).
        bounding_box (tuple): (x, y, width, height) for cropping.
        threshold_value_for_masking (int): Threshold value to generate the cel mask.

    Returns:
        bool: True if successful, False otherwise.
    """
    bb_x, bb_y, bb_w, bb_h = bounding_box
    if bb_w <= 0 or bb_h <= 0:
        logger.error(f"Bounding box has invalid dimensions. W:{bb_w}, H:{bb_h}")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open input video: {input_video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Use 'MJPG' codec for this intermediate file to better preserve pure white
    temp_video_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    logger.info(f"Using MJPG codec for temporary cropped video: {output_video_path}")

    # Output video has dimensions of the bounding box
    out = cv2.VideoWriter(output_video_path, temp_video_fourcc, original_fps, (bb_w, bb_h))

    if not out.isOpened():
        logger.error(f"Could not open video writer for: {output_video_path} with MJPG codec.")
        cap.release()
        return False

    logger.info(f"Cropping video to bounding box and replacing background with white. Output: '{output_video_path}'")
    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video or error
            break

        actual_cropped_color_frame = frame[bb_y: bb_y + bb_h, bb_x: bb_x + bb_w]
        actual_ch, actual_cw = actual_cropped_color_frame.shape[:2]

        if actual_ch == 0 or actual_cw == 0:
            output_frame = np.full((bb_h, bb_w, 3), (255, 255, 255), dtype=np.uint8)
            out.write(output_frame)
            frames_processed += 1
            continue

        gray_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh_full_frame = cv2.threshold(gray_full_frame, threshold_value_for_masking, 255, cv2.THRESH_BINARY_INV)
        actual_cropped_mask_2d = thresh_full_frame[bb_y: bb_y + bb_h, bb_x: bb_x + bb_w]

        if actual_cropped_mask_2d.shape[0] != actual_ch or actual_cropped_mask_2d.shape[1] != actual_cw:
            actual_cropped_mask_2d = cv2.resize(actual_cropped_mask_2d, (actual_cw, actual_ch),
                                                interpolation=cv2.INTER_NEAREST)

        if len(actual_cropped_mask_2d.shape) == 3:
            actual_cropped_mask_2d = cv2.cvtColor(actual_cropped_mask_2d, cv2.COLOR_BGR2GRAY)
        _, actual_cropped_mask_2d = cv2.threshold(actual_cropped_mask_2d, 127, 255, cv2.THRESH_BINARY)

        output_frame = np.full((bb_h, bb_w, 3), (255, 255, 255), dtype=np.uint8)
        content_canvas_white_bg = np.full((actual_ch, actual_cw, 3), (255, 255, 255), dtype=np.uint8)
        mask_for_content_3ch = cv2.cvtColor(actual_cropped_mask_2d, cv2.COLOR_GRAY2BGR)
        cel_on_white_content = np.where(mask_for_content_3ch == 255, actual_cropped_color_frame,
                                        content_canvas_white_bg)
        output_frame[0:actual_ch, 0:actual_cw] = cel_on_white_content

        out.write(output_frame)
        frames_processed += 1

    logger.info(
        f"Successfully processed video for cel extraction on white background. Processed {frames_processed} frames.")
    cap.release()
    out.release()
    return True


def resize_video_contain_with_padding(input_video_path, output_video_path, target_width=512, target_height=512,
                                      padding_color=(255, 255, 255), alignment="bottom"):
    if target_width <= 0 or target_height <= 0:
        logger.error(f"Target width ({target_width}) or height ({target_height}) is invalid.")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open input video file: {input_video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_w_orig == 0 or frame_h_orig == 0:
        logger.error(f"Could not read original frame dimensions from {input_video_path}")
        cap.release()
        return False

    # Use the global VIDEO_FOURCC for the final output
    out = cv2.VideoWriter(output_video_path, VIDEO_FOURCC, original_fps, (target_width, target_height))
    if not out.isOpened():
        logger.error(f"Could not open video writer for output file: {output_video_path}")
        cap.release()
        return False

    logger.info(
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

        align_parts = alignment.lower().split('-')

        if "top" in align_parts:
            y_offset = 0
        elif "bottom" in align_parts:
            y_offset = target_height - new_h
        else:
            y_offset = (target_height - new_h) // 2

        if "left" in align_parts:
            x_offset = 0
        elif "right" in align_parts:
            x_offset = target_width - new_w
        else:
            x_offset = (target_width - new_w) // 2

        if alignment.lower() == "center":
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2

        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        background[y_offset: y_offset + new_h, x_offset: x_offset + new_w] = resized_frame

        out.write(background)
        frames_processed += 1

    logger.info(f"Successfully resized (contain with padding) video. Processed {frames_processed} frames.")
    cap.release()
    out.release()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts an animation cel, crops the video to it (placing cel on white background), and then resizes the cropped video with padding and alignment.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, required=True, help="Path for the final resized output video file.")
    parser.add_argument("--align", type=str, default="bottom",
                        choices=["center", "top", "bottom", "left", "right",
                                 "top-left", "top-right", "bottom-left", "bottom-right"],
                        help="Alignment of the video within the padded frame (default: bottom).")

    args = parser.parse_args()

    input_file_path = args.input
    final_output_file_path = args.output
    alignment_choice = args.align

    temp_dir = ".tmp"
    temp_filename = "cropped_video.mp4"
    temp_cropped_video_file_path = None

    cel_detection_threshold = 230

    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_cropped_video_file_path = os.path.join(temp_dir, temp_filename)

        logger.info(f"--- Step 1: Processing video to find cel bounding box: {input_file_path} ---")
        bounding_box = extract_cel_bounding_box(input_file_path, threshold_value=cel_detection_threshold,
                                                min_contour_area=500)

        if bounding_box:
            x_bb, y_bb, w_bb, h_bb = bounding_box
            logger.info(f"Detected overall cel bounding box: X={x_bb}, Y={y_bb}, Width={w_bb}, Height={h_bb}")

            logger.info(
                f"--- Step 2: Cropping video to bounding box, placing cel on white background (temp file: {temp_cropped_video_file_path}) ---")
            success_crop = crop_video_with_bounding_box(
                input_file_path,
                temp_cropped_video_file_path,
                bounding_box,
                threshold_value_for_masking=cel_detection_threshold
            )

            if success_crop:
                logger.info(
                    f"Video cropped (cel on white bg) and saved to temporary file: '{temp_cropped_video_file_path}'")

                logger.info(
                    f"--- Step 3: Resizing (contain with white padding, align: {alignment_choice}) the processed video ---")
                success_resize = resize_video_contain_with_padding(
                    temp_cropped_video_file_path,
                    final_output_file_path,
                    alignment=alignment_choice
                )

                if success_resize:
                    logger.info(f"Video resized successfully and saved to '{final_output_file_path}'")
                else:
                    logger.error(f"Failed to resize video from '{temp_cropped_video_file_path}'.")
            else:
                logger.error(f"Failed to crop video (cel on white bg). Skipping resize step.")
        else:
            logger.warning(
                f"Could not detect the cel's bounding box in '{input_file_path}'. Further processing skipped.")

    except Exception as e:
        logger.error("An error occurred in the main execution block:", exc_info=True)

    # The temporary file at temp_cropped_video_file_path will persist as per previous request.

