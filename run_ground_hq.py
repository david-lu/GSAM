# === Imports and Setup ===
import os
import shutil

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2, build_sam2_hq_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.common_utils import draw_mask_image_with_detections
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images, extract_frames_from_video

# === Global Configuration ===
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === Load SAM2 Models ===
sam2_checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
video_predictor = build_sam2_hq_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(
    sam2_image_model, mask_threshold=-0, max_hole_area=16, max_sprinkle_area=128)

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
    # Get and sort frame names from the input directory
    frame_names = [
        p for p in os.listdir(input_video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize video predictor's inference state
    inference_state = video_predictor.init_state(video_path=input_video_dir)

    ann_frame_idx = 0  # Index of the annotation frame (usually the first frame)

    # Load the first frame for annotation
    img_path = os.path.join(input_video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # Run grounding model to get object boxes/labels from text prompt
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Post-process outputs to get detected boxes and labels
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.5,
        text_threshold=0.35,
        target_sizes=[image.size[::-1]]
    )

    # Prepare SAM2 image predictor with the RGB image
    image_predictor.set_image(np.array(image.convert("RGB")))
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # Run segmentation (SAM2) for each detected box
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Normalize mask dimensions for consistent processing
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    assert prompt_type in ["point", "box", "mask"]

    # Add objects to the video predictor using the chosen prompt type
    if prompt_type == "point":
        # Use sampled points from masks as prompts
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
        # Use detected boxes as prompts
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type == "mask":
        # Use segmentation masks as prompts
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )

    # Propagate object masks through the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Prepare output directory and ID-to-label mapping
    os.makedirs(output_video_dir, exist_ok=True)
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    saved_frame_paths = []

    # For each frame, overlay the masks on the original image and save
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(input_video_dir, frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = np.concatenate(list(segments.values()), axis=0)

        # Create detections object for annotation utilities
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        # Mask the image using the detected object masks
        masked_frame = draw_mask_image_with_detections(img.copy(), [detections])
        # Optionally, you can annotate with boxes/labels/masks (commented out)
        # annotated_frame = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
        # annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        # annotated_frame = sv.MaskAnnotator().annotate(scene=annotated_frame, detections=detections)

        # Save the masked frame
        out_path = os.path.join(output_video_dir, f"masked_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, masked_frame)
        saved_frame_paths.append(out_path)

    # Combine all annotated frames into a final output video
    return saved_frame_paths


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
        "--prompt", type=str, default=
        "animation characters holding props.",
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