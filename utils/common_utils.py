import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random
from typing import List, Tuple, Any


# TODO: Don't use a class for these static functions
class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def get_detection_from_mask(
            mask: np.ndarray,
            json_data_labels: dict) -> Tuple[sv.Detections | None, List[Tuple[Any, Any]] | None]:
        # color map
        unique_ids = np.unique(mask)

        all_object_masks = []
        for uid in unique_ids:
            if uid == 0:  # skip background id
                continue
            else:
                object_mask = (mask == uid)
                all_object_masks.append(object_mask[None])

        if len(all_object_masks) == 0:
            return None, None

        # get n masks: (n, h, w)
        all_object_masks = np.concatenate(all_object_masks, axis=0)

        all_object_boxes = []
        all_object_ids = []
        all_class_names = []
        object_id_to_name = {}

        for obj_id, obj_item in json_data_labels:
            # box id
            instance_id = obj_item["instance_id"]
            if instance_id not in unique_ids:  # not a valid box
                continue
            # box coordinates
            x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
            all_object_boxes.append([x1, y1, x2, y2])
            # box name
            class_name = obj_item["class_name"]

            # build id list and id2name mapping
            all_object_ids.append(instance_id)
            all_class_names.append(class_name)
            object_id_to_name[instance_id] = class_name

        # Adjust object id and boxes to ascending order
        paired_id_and_box = zip(all_object_ids, all_object_boxes)
        sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])

        # Because we get the mask data as ascending order, so we also need to ascend box and ids
        all_object_ids = [pair[0] for pair in sorted_pair]
        all_object_boxes = [pair[1] for pair in sorted_pair]

        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )

        # custom label to show both id and class name
        labels = list(zip(all_object_ids, all_class_names))

        return detections, labels

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)

            with open(file_path, "r") as file:
                json_data = json.load(file)
                json_data_labels = json_data["labels"].items()

                detections, labels = CommonUtils.get_detection_from_mask(mask, json_data_labels)
                if detections or labels is None:
                    continue

                label_texts = [f"{key}: {name}" for key, name in labels]

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(
                    scene=image.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(
                    annotated_frame, detections=detections, labels=label_texts)
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(
                    scene=annotated_frame, detections=detections)

                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, annotated_frame)
                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def draw_cleaned_masks(
            raw_image_path: str,
            mask_path: str,
            json_path: str,
            output_path: str):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:

            # load image
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")

            file_path = os.path.join(json_path, "mask_" + raw_image_name.split(".")[0] + ".json")
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy")
            mask = np.load(mask_npy_path)

            with open(file_path, "r") as file:
                json_data = json.load(file)
                json_data_labels = json_data["labels"].items()

                detections, labels = CommonUtils.get_detection_from_mask(mask, json_data_labels)

                masked_frame = draw_mask_image_with_detections(image.copy(), [detections])
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, masked_frame)


    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def clean_masks_with_closing(
    masks: np.ndarray,
    dilate_iter: int = 2,
    erode_iter: int = 1,
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


def draw_mask_image_with_detections(
    image: np.ndarray,
    detections_list: List[sv.Detections],
) -> np.ndarray:

    # Create white background
    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    
    # If no detections or all detections have no masks, return white image
    if not detections_list:
        return white_bg
    
    # Collect all masks from all detections
    all_masks = []
    for detections in detections_list:
        if detections is not None and detections.mask is not None and len(detections.mask) > 0:
            cleaned_masks = clean_masks_with_closing(detections.mask)
            all_masks.append(cleaned_masks)
    
    # If no valid masks were found, return white image
    if not all_masks:
        return white_bg
    
    # Combine all masks into a single mask
    # First concatenate all mask arrays, then take the logical OR across all masks
    all_masks_combined = np.concatenate(all_masks, axis=0)
    combined_mask = np.any(all_masks_combined, axis=0)
    
    # Apply the combined mask to the image
    masked_image = np.where(combined_mask[..., None], image, white_bg)
    return masked_image
