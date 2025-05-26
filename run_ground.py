# === Imports and Setup ===
import os
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
import tempfile
import shutil


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
image_predictor = SAM2ImagePredictor(sam2_image_model)

# === Load Grounding DINO ===
dino_model_id = "IDEA-Research/grounding-dino-tiny"
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
    # Get sorted image frame paths
    frame_names = sorted([
        p for p in os.listdir(input_video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = video_predictor.init_state(video_path=input_video_dir)

    ann_frame_idx = 0
    img_path = os.path.join(input_video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # Run Grounding DINO
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
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # Run SAM2 Image Predictor
    image_predictor.set_image(np.array(image.convert("RGB")))
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 3:
        masks = masks[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    # Add prompt to SAM2 video predictor
    assert prompt_type in ["point", "box", "mask"]
    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, points in enumerate(all_sample_points, start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state, ann_frame_idx, object_id, points=points, labels=labels)
    elif prompt_type == "box":
        for object_id, box in enumerate(input_boxes, start=1):
            video_predictor.add_new_points_or_box(
                inference_state, ann_frame_idx, object_id, box=box)
    elif prompt_type == "mask":
        for object_id, mask in enumerate(masks, start=1):
            video_predictor.add_new_mask(inference_state, ann_frame_idx, object_id, mask=mask)

    # Track in video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save annotated frames
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
        annotated = sv.BoxAnnotator().annotate(img.copy(), detections)
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        annotated = sv.MaskAnnotator().annotate(annotated, detections)

        out_path = os.path.join(output_video_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved_frame_paths.append(out_path)

    # Convert to video
    return saved_frame_paths

def extract_frames_from_video(video_path: str, output_dir: str) -> list[str]:
    """
    Extracts all frames from a video file and saves them as high-quality JPEGs.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.

    Returns:
        list[str]: Sorted list of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Set JPEG quality to maximum (100)
        out_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        frame_paths.append(out_path)
        frame_idx += 1

    cap.release()
    return frame_paths


def track_from_video_file(
        video_path: str,
        output_video_path: str,
        text_prompt: str = "animated animation cel",
        prompt_type: str = "box"
) -> str:
    """
    Wrapper function:
    - Extracts frames from video file
    - Runs tracking
    - Assembles annotated frames into final video

    Args:
        video_path (str): Path to input video file (e.g., .mp4).
        text_prompt (str): Text prompt to detect and track (e.g., "car.").
        output_video_path (str): Output path for final tracked video.
        prompt_type (str): "box", "point", or "mask" (for SAM2 video predictor).

    Returns:
        str: Path to final annotated output video
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_frame_dir = os.path.join(tmp_dir, "input")
        output_frame_dir = os.path.join(tmp_dir, "output")

        # Step 1: Extract frames from input video
        extract_frames_from_video(video_path, input_frame_dir)

        # Step 2: Run object tracking on the frame directory
        track_object_in_video(
            input_video_dir=input_frame_dir,
            output_video_dir=output_frame_dir,
            text_prompt=text_prompt,
            prompt_type=prompt_type
        )

        # Step 3: Assemble the output frames into a final video
        create_video_from_images(output_frame_dir, output_video_path)

    return output_video_path
