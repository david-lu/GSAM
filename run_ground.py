# === Imports and Setup ===
import os
import shutil

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


# === Global Configuration ===
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === Load SAM2 Models ===
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(
    sam2_image_model, mask_threshold=0.5, max_hole_area=8, max_sprinkle_area=128)

# === Load Grounding DINO ===
dino_model_id = "IDEA-Research/grounding-dino-base"
# dino_model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(dino_model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

# === Inference Function ===
def track_object_in_video(
    input_video_dir: str,
    output_video_dir: str,
    text_prompt: str,
    prompt_type: str = "box",
) -> list[str]:
    frame_names = [
        p for p in os.listdir(input_video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = video_predictor.init_state(video_path=input_video_dir)

    ann_frame_idx = 0

    img_path = os.path.join(input_video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    image_predictor.set_image(np.array(image.convert("RGB")))
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    assert prompt_type in ["point", "box", "mask"]

    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif prompt_type == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(output_video_dir, exist_ok=True)
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    saved_frame_paths = []

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(input_video_dir, frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = np.concatenate(list(segments.values()), axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        masked_frame = mask_image_with_detections(img.copy(), detections)
        # annotated_frame = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
        # annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        # annotated_frame = sv.MaskAnnotator().annotate(scene=annotated_frame, detections=detections)

        out_path = os.path.join(output_video_dir, f"masked_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, masked_frame)
        saved_frame_paths.append(out_path)

    create_video_from_images(output_video_dir, os.path.join(output_video_dir, "tracked_output.mp4"))
    return saved_frame_paths

def clean_masks_with_closing(
    masks: np.ndarray,
    dilate_iter: int = 2,
    erode_iter: int = 2,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Applies morphological closing (dilate then erode) to clean each binary mask.

    Args:
        masks (np.ndarray): Array of shape (N, H, W) containing N binary masks.
        dilate_iter (int): Number of dilation iterations.
        erode_iter (int): Number of erosion iterations.
        kernel_size (int): Size of the structuring element.

    Returns:
        np.ndarray: Cleaned masks of shape (N, H, W) as boolean array.
    """
    cleaned = []
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for i in range(masks.shape[0]):
        mask = (masks[i] * 255).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = cv2.erode(mask, kernel, iterations=erode_iter)
        cleaned.append(mask > 0)

    return np.stack(cleaned, axis=0)



def mask_image_with_detections(
    image: np.ndarray,
    detections: sv.Detections,
) -> np.ndarray:
    if detections.mask is None or len(detections.mask) == 0:
        return np.ones_like(image, dtype=np.uint8) * 255

    cleaned_masks = clean_masks_with_closing(detections.mask)

    combined_mask = np.any(cleaned_masks, axis=0)
    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    masked_image = np.where(combined_mask[..., None], image, white_bg)
    return masked_image


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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track objects in a video using Grounded-SAM2")

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input video file (e.g., .mp4)"
    )

    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output video file (e.g., output.mp4)"
    )

    parser.add_argument(
        "--prompt", type=str, default="animation characters maybe holding object.",
        help="Text prompt for the object to track (e.g., 'car.')"
    )

    parser.add_argument(
        "--prompt_type", type=str, default="box", choices=["box", "point", "mask"],
        help="Type of SAM2 prompt to use (box, point, or mask)"
    )

    args = parser.parse_args()

    # Run the pipeline
    output = track_from_video_file(
        input_video_path=args.input,
        text_prompt=args.prompt,
        output_video_path=args.output,
        prompt_type=args.prompt_type
    )