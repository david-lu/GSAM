import cv2
import os
import shutil  # For copying and deleting directories
import tempfile  # For creating a temporary directory
from skimage.metrics import structural_similarity
import numpy as np
import argparse


def _calculate_ssim_between_files(image_path1, image_path2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    """
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None:
            return None
        if img2 is None:
            return None

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        dynamic_range = 255
        score, _ = structural_similarity(img1, img2, full=True, data_range=dynamic_range)
        return score
    except Exception as e:
        print(f"Error calculating SSIM between {image_path1} and {image_path2}: {e}")
        return None


def extract_frames_to_temp(video_path, temp_output_dir):
    """
    Extracts all frames from a video file and saves them into a temporary directory.
    Args:
        video_path (str): The path to the video file.
        temp_output_dir (str): The temporary directory where ALL extracted frames will be saved.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []
    # temp_output_dir is created by tempfile.mkdtemp()

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    extracted_frame_paths = []
    frame_count = 0
    print(f"Extracting all frames temporarily from {video_path} to {temp_output_dir}...")
    while True:
        success, frame = video_capture.read()
        if not success:
            if frame_count == 0:
                print(f"Warning: No frames could be read from {video_path}.")
            break
        frame_count += 1
        frame_filename = f"{video_filename}_frame_{frame_count:04d}.jpg"
        frame_filepath = os.path.join(temp_output_dir, frame_filename)
        try:
            cv2.imwrite(frame_filepath, frame)
            extracted_frame_paths.append(frame_filepath)
        except Exception as e:
            print(f"Error writing temporary frame {frame_filepath}: {e}")
    video_capture.release()
    print(f"Temporarily extracted {len(extracted_frame_paths)} frames.")
    return sorted(extracted_frame_paths)


def check_frames_for_uniqueness(ordered_frame_paths, ssim_threshold):
    """
    Checks each frame for uniqueness using SSIM.
    A frame is unique if it's the first frame, or if its SSIM to the
    preceding frame is below ssim_threshold.
    """
    uniqueness_results = []
    num_frames = len(ordered_frame_paths)

    if num_frames == 0:
        print("No frames provided to check for uniqueness.")
        return []

    print(f"Checking uniqueness for {num_frames} frames with SSIM Threshold={ssim_threshold:.2f} (vs preceding frame)")

    for i in range(num_frames):
        current_frame_path = ordered_frame_paths[i]
        result = {
            'frame_path': current_frame_path,  # Path in the temporary directory
            'is_unique': False,
            'ssim_to_prev': None,
            'is_diff_from_prev': True,
            'reason_prev': "N/A",
            'final_reason': ''
        }

        if i > 0:
            prev_frame_path = ordered_frame_paths[i - 1]
            ssim_val = _calculate_ssim_between_files(current_frame_path, prev_frame_path)
            result['ssim_to_prev'] = ssim_val

            is_different = False
            reason = "High SSIM or comparison error"

            if ssim_val is not None:
                if ssim_val < ssim_threshold:
                    is_different = True
                    reason = f"Low SSIM ({ssim_val:.2f} < {ssim_threshold:.2f})"
                else:
                    is_different = False
                    reason = f"High SSIM ({ssim_val:.2f} >= {ssim_threshold:.2f})"
            else:
                is_different = False
                reason = "SSIM calculation error with preceding frame"

            result['is_diff_from_prev'] = is_different
            result['reason_prev'] = reason
        else:
            result['is_diff_from_prev'] = True
            result['reason_prev'] = "First frame (no preceding frame)"

        if num_frames == 1:
            result['is_unique'] = True
            result['final_reason'] = "Only one frame; considered unique by default."
        elif i == 0:
            result['is_unique'] = True
            result['final_reason'] = "First frame, considered unique (no preceding frame to compare)."
        else:
            result['is_unique'] = result['is_diff_from_prev']
            if result['is_unique']:
                result['final_reason'] = f"Different from preceding frame (Reason: {result['reason_prev']})"
            else:
                result['final_reason'] = f"Similar to preceding frame (Reason: {result['reason_prev']})"

        uniqueness_results.append(result)
    return uniqueness_results


def process_video_wrapper(video_file_path, final_output_dir_unique_frames, ssim_threshold):
    """
    Wrapper function to extract frames temporarily, check uniqueness,
    and save only unique frames as images to the final output directory.
    """
    print(f"--- Starting video processing for: {video_file_path} ---")
    print(f"Final output directory for UNIQUE frames: {final_output_dir_unique_frames}")
    os.makedirs(final_output_dir_unique_frames, exist_ok=True)

    temp_extraction_dir = None
    try:
        # Create a temporary directory to store all frames initially
        temp_extraction_dir = tempfile.mkdtemp(prefix="frame_extractor_temp_")
        print(f"Using temporary directory for all frames: {temp_extraction_dir}")

        extracted_paths_in_temp = extract_frames_to_temp(video_file_path, temp_extraction_dir)

        if not extracted_paths_in_temp:
            print("Frame extraction failed or produced no frames. Aborting uniqueness check.")
            return []

        uniqueness_details = check_frames_for_uniqueness(extracted_paths_in_temp, ssim_threshold)

        print("\n--- Uniqueness Check Complete ---")
        unique_frames_count = 0

        print(f"Copying unique frames to {final_output_dir_unique_frames}...")
        for result_item in uniqueness_details:
            if result_item['is_unique']:
                unique_frames_count += 1
                source_path = result_item['frame_path']  # Path in temp directory
                filename = os.path.basename(source_path)
                destination_path = os.path.join(final_output_dir_unique_frames, filename)
                try:
                    shutil.copy2(source_path, destination_path)
                except Exception as e:
                    print(f"Error copying unique frame {source_path} to {destination_path}: {e}")

        if unique_frames_count > 0:
            print(f"Successfully copied {unique_frames_count} unique frames to {final_output_dir_unique_frames}.")
        else:
            print(f"No unique frames found to copy to {final_output_dir_unique_frames}.")

        print(f"Total frames analyzed: {len(uniqueness_details)}")
        print(f"Number of unique frames identified: {unique_frames_count}")

        return uniqueness_details

    finally:
        # Clean up the temporary directory
        if temp_extraction_dir and os.path.exists(temp_extraction_dir):
            try:
                shutil.rmtree(temp_extraction_dir)
                print(f"Successfully removed temporary directory: {temp_extraction_dir}")
            except Exception as e:
                print(f"Error removing temporary directory {temp_extraction_dir}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts frames from a video, identifies unique frames based on SSIM comparison "
                    "with the preceding frame, and saves ONLY these unique frames as images.")
    parser.add_argument("-v", "--video_path", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save ONLY the unique extracted frames as images. Will be created if it doesn't exist.")
    parser.add_argument("-s", "--ssim_threshold", type=float, default=0.95,
                        help="SSIM threshold for uniqueness (0.0 to 1.0). "
                             "A frame is unique if its SSIM to the PRECEDING frame is *below* this. "
                             "First frame is always unique. Default: 0.85")

    args = parser.parse_args()

    final_unique_frames_output_dir = args.output_dir
    # The directory will be created by process_video_wrapper if it doesn't exist.

    print(f"Input video: {args.video_path}")
    print(f"UNIQUE extracted frames WILL BE SAVED in: {final_unique_frames_output_dir}")
    print(f"SSIM Threshold for uniqueness (vs preceding frame): {args.ssim_threshold:.2f}")

    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found. Please check the path.")
    else:
        analysis_results = process_video_wrapper(args.video_path,
                                                 final_unique_frames_output_dir,
                                                 ssim_threshold=args.ssim_threshold)

        if analysis_results:
            print("\n--- Detailed Uniqueness Results (Printed to Console for verification): ---")
            # Limit printing if too many frames, or just print summary.
            # For now, keeping it for verbosity during testing.
            for result in analysis_results:
                if result['is_unique'] or len(analysis_results) < 20:  # Print details for unique or if few frames
                    frame_name = os.path.basename(result['frame_path'])  # This will be temp path name

                    ssim_p_str = f"{result['ssim_to_prev']:.2f}" if result['ssim_to_prev'] is not None else "N/A"
                    diff_p_status_str = str(result['is_diff_from_prev'])

                    print(f"  Analyzed Frame (from temp): {frame_name} | Unique: {str(result['is_unique']):<5}")
                    print(
                        f"    Prev: SSIM={ssim_p_str}, DifferentFromPrev={diff_p_status_str:<5} (Reason: {result['reason_prev']})")
                    print(f"    Final Uniqueness Reason: {result['final_reason']}")

            if not any(res['is_unique'] for res in analysis_results):  # Check if any unique frames were found
                print("\nNo unique frames were identified to save based on the current SSIM threshold and logic.")

        else:
            print("Video analysis did not complete successfully or no frames were processed.")
