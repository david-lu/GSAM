import cv2
import os

from tqdm import tqdm


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


