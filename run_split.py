import os
import shutil
import logging
import cv2  # OpenCV for image writing
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector, AdaptiveDetector  # Added AdaptiveDetector here
from scenedetect.scene_detector import \
    SceneDetector  # This is an abstract class, not typically used directly for detection
from scenedetect.video_splitter import split_video_ffmpeg  # For splitting video
from scenedetect.video_manager import VideoManager  # For type hinting video object
from typing import List, Union, Optional, Tuple  # For type hinting
import numpy  # For type hinting frame_data
# cv2 and numpy were imported twice, removed redundant imports
import argparse  # For command-line arguments

# --- Logging Setup ---
# Configure logging once at the module level.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def split_video_into_scenes(
        video_path: str,
        output_dir: str,
        detectors: Optional[List[SceneDetector]] = None,  # Changed ContentDetector to SceneDetector for broader type
) -> List[str]:
    """
    Splits `video_path` into scene clips under `output_dir`.
    Returns list of generated clip filenames (not full paths).
    """
    # Default detector if none provided
    detectors = detectors or [
        AdaptiveDetector(adaptive_threshold=4.0, window_width=3)]  # Changed default to AdaptiveDetector

    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        return []
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {output_dir}")

    # 1) Detect scenes
    try:
        # Using VideoManager as recommended for robust handling
        video = VideoManager([video_path])
        scene_manager = SceneManager()
        for det in detectors:
            scene_manager.add_detector(det)

        # Start video processing
        video.set_downscale_factor(1)  # No downscaling, process at original resolution
        video.start()

        # Perform scene detection
        scene_manager.detect_scenes(frame_source=video, show_progress=True)
        scene_list = scene_manager.get_scene_list()  # No base_timecode needed if using VideoManager default

    except Exception as e:
        logger.error(f"Error during scene detection for {video_path}: {e}")
        if 'video' in locals() and video.is_started():
            video.release()
        return []

    if not scene_list:
        logger.warning(f"No scenes detected in {video_path}.")
        if video.is_started():
            video.release()
        return []

    logger.info(f"Detected {len(scene_list)} scenes in {video_path}.")

    # 2) Split via FFmpeg
    # Ensure video is released before split_video_ffmpeg tries to access it if it's the same path
    # However, split_video_ffmpeg opens its own video handle.
    if video.is_started():
        video.release()

    basename, ext = os.path.splitext(os.path.basename(video_path))
    template = f"{basename}_scene-$SCENE_NUMBER{ext}"  # Changed template for clarity

    try:
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=output_dir,
            output_file_template=template,
            show_progress=True,
            # It's good practice to specify video and audio codecs if known,
            # or let ffmpeg decide if not critical.
            # e.g., vcodec="libx264", acodec="aac"
        )
    except Exception as e:
        logger.error(f"Error during video splitting for {video_path} with FFmpeg: {e}")
        return []

    # 3) Collect filenames
    clips = []
    # Scene numbers in the template are 1-based.
    for i, scene in enumerate(scene_list):
        # Construct the expected filename based on the template
        # $SCENE_NUMBER is usually 1-indexed and padded, e.g., 001, 002
        # PySceneDetect's default padding for $SCENE_NUMBER is 3 digits if total scenes < 1000
        # Let's assume 3-digit padding for robustness, or check PySceneDetect's exact behavior
        # For now, constructing based on typical output.
        # The actual number used by split_video_ffmpeg is scene.scene_number (1-indexed)
        scene_num_str = f"{scene.scene_number:03d}"  # Use scene.scene_number

        fname = template.replace("$SCENE_NUMBER", scene_num_str)
        expected_path = os.path.join(output_dir, fname)
        if os.path.exists(expected_path):
            clips.append(fname)
        else:
            # Fallback for older PySceneDetect versions or different template interpretations
            # This part might need adjustment based on the exact output of split_video_ffmpeg
            # For example, if $SCENE_NUMBER is just the index i+1
            scene_num_str_alt = f"{i + 1:03d}"
            fname_alt = template.replace("$SCENE_NUMBER", scene_num_str_alt)
            expected_path_alt = os.path.join(output_dir, fname_alt)
            if os.path.exists(expected_path_alt):
                clips.append(fname_alt)
                logger.debug(f"Found clip with alternative numbering: {fname_alt}")
            else:
                logger.warning(
                    f"Missing expected clip: {fname} (or {fname_alt}) at {expected_path} (or {expected_path_alt})")

    if clips:
        logger.info(f"Successfully split {video_path} into {len(clips)} scenes in {output_dir}.")
    else:
        logger.warning(f"No clips were generated or found for {video_path} in {output_dir}.")

    return clips


def get_scene_folder(input_video_file: str):
    """
    Generates a default output folder name based on the input video file.
    Example: for "myvideo.mp4", returns "myvideo_scenes".
    """
    base = os.path.splitext(os.path.basename(input_video_file))[0]
    return os.path.join(
        os.path.dirname(input_video_file) or '.',  # Use current dir if no dirname
        f"{base}_scenes"
    )


def fetch_all_videos(directory="data/movies"):
    """
    Fetches all video files from a specified directory.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    found_videos = []
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return found_videos

    for f in os.listdir(directory):
        if f.lower().endswith(video_extensions) and os.path.isfile(os.path.join(directory, f)):
            found_videos.append(os.path.join(directory, f))
    return found_videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Splits a video into scenes using PySceneDetect.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the scene clips.")
    # Optional: add arguments for detector type, threshold, etc.
    # Example: parser.add_argument("--threshold", type=float, default=4.0, help="AdaptiveDetector threshold.")

    args = parser.parse_args()

    input_video = args.input_video
    output_directory = args.output_dir

    logger.info(f"Starting scene splitting for: {input_video}")
    logger.info(f"Output will be saved to: {output_directory}")

    # Example using AdaptiveDetector, can be made configurable
    # Note: The `detectors` argument in `split_video_into_scenes` expects a list of SceneDetector instances.
    custom_detectors = [AdaptiveDetector(adaptive_threshold=2.0, window_width=3)]

    generated_clips = split_video_into_scenes(input_video, output_directory, detectors=custom_detectors)

    if generated_clips:
        logger.info("Scene splitting complete. Generated clips:")
        for clip_name in generated_clips:
            logger.info(f" - {clip_name}")
    else:
        logger.warning("Scene splitting finished, but no clips were generated or found.")

    # The previous logic for processing all videos in a directory is now replaced by command-line arguments.
    # If you want to reinstate that, you would typically not use argparse or make it optional.
    # Example:
    # videos = fetch_all_videos("path/to/your/videos") # Replace with your video directory
    # if not videos:
    #     logger.info("No videos found to process.")
    # else:
    #     for video_file in videos:
    #         logger.info(f"Processing video: {video_file}")
    #         # Determine output directory for each video, e.g., using get_scene_folder
    #         vid_output_dir = get_scene_folder(video_file)
    #         split_video_into_scenes(video_file, vid_output_dir, detectors=custom_detectors)
