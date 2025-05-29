import shutil

import cv2
import os

import torch
from tqdm import tqdm

from run_ground import track_object_in_video


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def extract_frames_from_video(video_path, output_frames_dir):
    os.makedirs(output_frames_dir, exist_ok=True)
    # Clear existing contents
    for f in os.listdir(output_frames_dir):
        os.remove(os.path.join(output_frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_frames_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        frame_idx += 1
    cap.release()


def track_from_video_file(
    text_prompt: str,
    input_video_path: str,
    output_video_path: str,
    prompt_type: str = "box"
) -> str:
    """
    Extracts frames from a video, runs object tracking, and saves the annotated output video.
    Uses local persistent temp folders: ./input_frames and ./output_frames

    Returns the path to the final output video.
    """
    input_frame_dir = ".tmp/input_frames"
    output_frame_dir = ".tmp/output_frames"

    # Ensure input/output frame folders are clean
    for folder in [input_frame_dir, output_frame_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Step 1: Extract video to frames
    extract_frames_from_video(input_video_path, input_frame_dir)

    # Step 2: Run tracking
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        track_object_in_video(
            input_video_dir=input_frame_dir,
            output_video_dir=output_frame_dir,
            text_prompt=text_prompt,
            prompt_type=prompt_type
        )

    # Step 3: Convert annotated frames to final video
    create_video_from_images(output_frame_dir, output_video_path)

    return output_video_path
