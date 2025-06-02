# === Imports and Setup ===
import copy
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

from utils.common_utils import draw_mask_image_with_detections, CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images, extract_frames_from_video

# === Global Configuration ===
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === Hyperparam for Ground and Tracking ===
PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
INPUT_FRAME_DIR = ".tmp/input_frames"
OUTPUT_FRAME_DIR = ".tmp/output_frames"
MASK_DATA_DIR = ".tmp/mask_data"
JSON_DATA_DIR = ".tmp/json_data"

# === Load SAM2 Models ===
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(
    model_cfg,
    sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(
    sam2_image_model,
    mask_threshold=0.6,
    max_sprinkle_area=128)

# === Load Grounding DINO ===
dino_model_id = "IDEA-Research/grounding-dino-base"
# dino_model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(dino_model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

# === Inference Function ===
def track_object_in_video(text_prompt: str, step: int = 12, reverse: bool = False):
    """
    Tracks objects in a video using Grounded-SAM2.

    Args:
    - text_prompt (str): Text prompt for the object to track.
    - step (int): Step size for processing frames. Defaults to 24.
    - reverse (bool): Whether to perform reverse tracking. Defaults to False.
    """
    # Get the list of frame names in the input directory
    frame_names = [
        p for p in os.listdir(INPUT_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize the video predictor state
    inference_state = video_predictor.init_state(video_path=INPUT_FRAME_DIR)

    # Initialize the mask dictionary model
    sam2_masks = MaskDictionaryModel()
    objects_count = 0
    frame_object_count = {}

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
        # Prompt Grounding DINO to get the box coordinates on a specific frame
        print(f"========================= FRAME {start_frame_idx} ===================================")
        img_path = os.path.join(INPUT_FRAME_DIR, frame_names[start_frame_idx])
        image = Image.open(img_path).convert("RGB")
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(
            promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

        # Run Grounding DINO on the image
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        # Results from a grounded object detection processor (likely GROUNDING DINO)
        # This processes the model outputs to obtain bounding boxes and labels for detected objects
        results = processor.post_process_grounded_object_detection(
            outputs,                          # Raw model outputs
            inputs.input_ids,                 # Input token IDs
            box_threshold=0.3,                # Confidence threshold for box detection
            text_threshold=0.4,               # Confidence threshold for text detection
            target_sizes=[image.size[::-1]]   # Target size for scaling boxes to image dimensions
        )

        # Set the current image for SAM (Segment Anything Model) predictor
        image_predictor.set_image(np.array(image.convert("RGB")))

        # Extract the detected bounding boxes from results
        input_boxes = results[0]["boxes"]     # Bounding boxes for detected objects
        # print("results[0]",results[0])
        OBJECTS = results[0]["labels"]        # Labels for the detected objects
        if input_boxes.shape[0] != 0:  # If objects were detected
            print(f"Objects {OBJECTS} detected in the frame {start_frame_idx}. detecting masks...")

            # Use SAM 2 to generate masks for each detected object's bounding box
            masks, scores, logits = image_predictor.predict(
                point_coords=None,            # No point coordinates used
                point_labels=None,            # No point labels used
                box=input_boxes,              # Using bounding boxes as prompts
                multimask_output=False,       # Only generate one mask per box
            )
            # Normalize mask shape to (n, H, W) format
            if masks.ndim == 2:
                masks = masks[None]           # Add batch dimension if missing
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)      # Remove unnecessary dimension
            """
            Step 3: Register each object's positive points to video predictor
            """

            # Step 3: Register detected objects' masks to the video predictor
            if mask_dict.promote_type == "mask":
                # Add the current frame's masks, boxes, and labels to mask_dict
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device),  # Convert numpy masks to tensor
                                                   box_list=torch.tensor(input_boxes),        # Convert boxes to tensor
                                                   label_list=OBJECTS)                        # Labels for the objects
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
        else:
            # No objects detected in this frame
            print(f"No object detected in the frame {start_frame_idx}, skip frame merge")
            mask_dict = sam2_masks  # Use previous masks

        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        # Updates mask_dict by merging with tracking annotations based on IoU threshold
        # Returns and updates the count of unique objects tracked so far
        new_mask_dict = copy.deepcopy(sam2_masks)
        objects_count = new_mask_dict.new_update_masks(
            tracking_annotation_dict=mask_dict,
            iou_threshold=0.8,
            objects_count=objects_count)
        mask_dict = new_mask_dict
        # Store the object count for this frame
        frame_object_count[start_frame_idx] = objects_count
        print("objects_count", objects_count)

        if len(mask_dict.labels) == 0:
            # If no objects to track, save empty masks and JSON data
            mask_dict.save_empty_mask_and_json(MASK_DATA_DIR,                # Directory to save mask files
                                               JSON_DATA_DIR,                # Directory to save JSON annotation files
                                               image_name_list=frame_names[start_frame_idx:start_frame_idx + step])
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        else:
            # print('mask_dict', mask_dict)
            video_predictor.reset_state(inference_state)

            # For each detected object, add its mask to the video predictor
            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,          # Current inference state
                    start_frame_idx,          # Starting frame index
                    object_id,                # Unique ID for this object
                    object_info.mask,         # Mask for this object
                )

            # Dictionary to store segmentation results for each frame
            video_segments = {}  # output the following {step} frames tracking masks
            # Propagate object masks to subsequent frames
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state,
                    max_frame_num_to_track=step,      # Maximum frames to track forward
                    start_frame_idx=start_frame_idx): # Starting frame index
                frame_masks = MaskDictionaryModel()   # Container for this frame's masks

                # Process each object's mask for this frame
                for i, out_obj_id in enumerate(out_obj_ids):
                    # Convert logits to binary mask (threshold at 0.0)
                    out_mask = (out_mask_logits[i] > 0.0)  # .cpu().numpy()
                    # Create object info with mask, class name and logit
                    object_info = ObjectInfo(instance_id=out_obj_id, 
                                           mask=out_mask[0],
                                           class_name=mask_dict.get_target_class_name(out_obj_id),
                                           logit=mask_dict.get_target_logit(out_obj_id))
                    object_info.update_box()  # Update bounding box based on mask
                    # Add this object to the current frame's masks
                    frame_masks.labels[out_obj_id] = object_info
                    # Create mask filename based on frame name
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                # Store this frame's masks in the video segments dictionary
                video_segments[out_frame_idx] = frame_masks
                # Update tracking state for next iteration
                sam2_masks = copy.deepcopy(frame_masks)

            print("video_segments:", len(video_segments))
        """
        Step 5: save the tracking masks and json files
        """
        # Save the tracking masks and corresponding JSON files
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            # Create a single mask image where pixel values correspond to object IDs
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            # Convert to numpy array and save to disk
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(MASK_DATA_DIR, frame_masks_info.mask_name), mask_img)

            # Save corresponding JSON metadata
            json_data_path = os.path.join(JSON_DATA_DIR, frame_masks_info.mask_name.replace(".npy", ".json"))
            frame_masks_info.to_json(json_data_path)

    CommonUtils.draw_masks_and_box_with_supervision(
        INPUT_FRAME_DIR, MASK_DATA_DIR, JSON_DATA_DIR, OUTPUT_FRAME_DIR)

    if not reverse:
        CommonUtils.draw_cleaned_masks(
            INPUT_FRAME_DIR, MASK_DATA_DIR, JSON_DATA_DIR, OUTPUT_FRAME_DIR)
        return

    print("============================== REVERSE TRACKING =================================")
    start_object_id = 0
    object_info_dict = {}
    for frame_idx, current_object_count in frame_object_count.items():
        print(f"Reverse tracking frame {frame_idx}. Frame Object count: {frame_object_count}")
        mask_added = False
        if frame_idx != 0:
            video_predictor.reset_state(inference_state)
            image_base_name = frame_names[frame_idx].split(".")[0]
            json_data_path = os.path.join(JSON_DATA_DIR, f"mask_{image_base_name}.json")
            json_data = MaskDictionaryModel().from_json(json_data_path)
            mask_data_path = os.path.join(MASK_DATA_DIR, f"mask_{image_base_name}.npy")
            mask_array = np.load(mask_data_path)
            print(f"json: {json_data}")
            print(f"mask_array: {mask_array}")
            for object_id in range(start_object_id + 1, current_object_count + 1):
                print(f"Reverse tracking object {object_id}")
                object_info_dict[object_id] = json_data.labels[object_id]
                video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
                mask_added = True
        start_object_id = current_object_count

        if not mask_added:
            print(f"No object detected in the frame, Skipping frame {frame_idx}")
            continue
        else:
            print("object detected in the frame".format(frame_idx))

        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state,
                max_frame_num_to_track=step * 2,
                start_frame_idx=frame_idx,
                reverse=True):
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            json_data_path = os.path.join(JSON_DATA_DIR, f"mask_{image_base_name}.json")
            json_data = MaskDictionaryModel().from_json(json_data_path)
            mask_data_path = os.path.join(MASK_DATA_DIR, f"mask_{image_base_name}.npy")
            mask_array = np.load(mask_data_path)
            # merge the reverse tracking masks with the original masks
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu()
                if out_mask.sum() == 0:
                    print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                    continue
                object_info = object_info_dict[out_obj_id]
                object_info.mask = out_mask[0]
                object_info.update_box()
                json_data.labels[out_obj_id] = object_info
                mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                mask_array[object_info.mask] = out_obj_id

            np.save(mask_data_path, mask_array)
            json_data.to_json(json_data_path)

    CommonUtils.draw_cleaned_masks(
        INPUT_FRAME_DIR, MASK_DATA_DIR, JSON_DATA_DIR, OUTPUT_FRAME_DIR)

def track_from_video_file(
    text_prompt: str,
    input_video_path: str,
    output_video_path: str,
) -> str:

    # Ensure input/output frame folders are clean
    for folder in [INPUT_FRAME_DIR, OUTPUT_FRAME_DIR, MASK_DATA_DIR, JSON_DATA_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Step 1: Extract video to frames
    extract_frames_from_video(input_video_path, INPUT_FRAME_DIR)

    # Step 2: Run tracking
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        track_object_in_video(text_prompt=text_prompt, reverse=True)

    # Step 3: Convert annotated frames to final video
    create_video_from_images(OUTPUT_FRAME_DIR, output_video_path)

    return output_video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track objects in a video using Grounded-SAM2")

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
        "animated cartoon character. ",
        help="Text prompt for the object to track (e.g., 'car.')"
    )

    args = parser.parse_args()

    # Run the pipeline
    output = track_from_video_file(
        input_video_path=args.input,
        text_prompt=args.prompt,
        output_video_path=args.output,
    )