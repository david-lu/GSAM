import os

from scenedetect import AdaptiveDetector
import os
import shutil
import logging
import cv2 # OpenCV for image writing
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.video_splitter import split_video_ffmpeg # For splitting video
from scenedetect.video_manager import VideoManager # For type hinting video object
from typing import List, Union, Optional, Tuple # For type hinting
import numpy # For type hinting frame_data
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_video_into_scenes(
    video_path: str,
    output_dir: str,
    detectors: List[ContentDetector] = None,
) -> List[str]:
    """
    Splits `video_path` into scene clips under `output_dir`.
    Returns list of generated clip filenames (not full paths).
    """
    detectors = detectors or [ContentDetector(threshold=12.0, min_scene_len=15)]
    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        return []
    os.makedirs(output_dir, exist_ok=True)

    # 1) Detect scenes
    vm = VideoManager([video_path])
    sm = SceneManager()
    for det in detectors:
        sm.add_detector(det)
    vm.start()
    sm.detect_scenes(vm, show_progress=True)
    scene_list = sm.get_scene_list(vm.get_base_timecode())
    vm.release()

    if not scene_list:
        logger.warning("No scenes detected.")
        return []

    # 2) Split via FFmpeg
    basename, ext = os.path.splitext(os.path.basename(video_path))
    template = f"{basename}_$SCENE_NUMBER{ext}"
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=output_dir,
        output_file_template=template,
        show_progress=True,
    )

    # 3) Collect filenames
    clips = []
    for idx in range(len(scene_list)):
        num = f"{idx+1:03d}"
        fname = template.replace("$SCENE_NUMBER", num)
        if os.path.exists(os.path.join(output_dir, fname)):
            clips.append(fname)
        else:
            logger.warning(f"Missing expected clip: {fname}")
    return clips
def get_scene_folder(input_video_file: str):
    return os.path.join(
        os.path.dirname(input_video_file),
        os.path.splitext(os.path.basename(input_video_file))[0]
    )

def fetch_all_videos(directory="data/movies"):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(video_extensions) and os.path.isfile(os.path.join(directory, f))
    ]

if __name__ == '__main__':
    videos = fetch_all_videos()
    print(videos)
    for video in videos:
        print('SPLITTING UP: ', video)
        output_dir = get_scene_folder(video)
        split_video_into_scenes(video, output_dir, [AdaptiveDetector(adaptive_threshold=4, window_width=3)])
    # input_video_file: str = "data/little_nemo.mkv"
    # output_dir = get_scene_folder(input_video_file)
    # print(output_dir)
    # split_video_into_scenes(input_video_file, output_dir, [AdaptiveDetector(window_width=3)])
