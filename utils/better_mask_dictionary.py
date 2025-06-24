from typing import Tuple

import numpy as np
import json
import torch
import copy
import os
import cv2
from dataclasses import dataclass, field

BoundingBox = Tuple[int, int, int, int]

def calculate_bbox(mask) -> BoundingBox | None:
    nonzero_indices = torch.nonzero(mask)

    if nonzero_indices.size(0) == 0:
        # print("nonzero_indices", nonzero_indices)
        return None

    y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
    y_max, x_max = torch.max(nonzero_indices, dim=0)[0]

    return x_min.item(), y_min.item(), x_max.item(), y_max.item()

def calculate_iou(mask1, mask2):
    # Convert masks to float tensors for calculations
    mask1 = mask1.to(torch.float32)
    mask2 = mask2.to(torch.float32)

    # Calculate intersection and union
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection

    # Calculate IoU
    iou = intersection / union
    return iou

def bbox_contains(
    outer_bbox: BoundingBox | None,
    inner_bbox: BoundingBox | None,
    threshold: float = 0.0
) -> bool:
    """
    Checks if an outer bounding box contains an inner bounding box, with an optional threshold.

    A bounding box is defined as (x_min, y_min, x_max, y_max).

    Args:
        outer_bbox (BoundingBox): The outer bounding box (potential container).
        inner_bbox (BoundingBox): The inner bounding box (potential contained).
        threshold (float): A value between 0.0 and 1.0 (inclusive).
                           If threshold > 0, inner_bbox is considered contained
                           if it's within outer_bbox plus a margin.
                           The margin is calculated as a percentage of outer_bbox's
                           width and height.
                           For example, a threshold of 0.1 means inner_bbox can
                           extend up to 10% of outer_bbox's dimension outside
                           outer_bbox and still be considered contained.
                           A negative threshold would make containment stricter,
                           requiring inner_bbox to be smaller than outer_bbox.

    Returns:
        bool: True if outer_bbox contains inner_bbox (with respect to the threshold),
              False otherwise.
    """
    if outer_bbox is None or inner_bbox is None:
        return False

    ox_min, oy_min, ox_max, oy_max = outer_bbox
    ix_min, iy_min, ix_max, iy_max = inner_bbox

    # Calculate dimensions of the outer bounding box
    outer_width = ox_max - ox_min
    outer_height = oy_max - oy_min

    # Calculate margins based on threshold and outer_bbox's dimensions
    # A positive margin expands the outer_bbox for the check.
    margin_x = threshold * outer_width
    margin_y = threshold * outer_height

    # Check if inner_bbox is contained within outer_bbox, considering the margin
    # For containment, inner_bbox's min coordinates must be >= outer_bbox's min - margin
    # and inner_bbox's max coordinates must be <= outer_bbox's max + margin.
    is_contained = (
        (ix_min >= ox_min - margin_x) and
        (iy_min >= oy_min - margin_y) and
        (ix_max <= ox_max + margin_x) and
        (iy_max <= oy_max + margin_y)
    )

    return is_contained


def bbox_vore(mask_a, mask_b):
    bbox_a = calculate_bbox(mask_a)
    bbox_b = calculate_bbox(mask_b)
    return bbox_contains(bbox_a, bbox_b, 0.1) or bbox_contains(bbox_b, bbox_a, 0.1) 

@dataclass
class BetterMaskDictionary:
    mask_name: str = None
    mask_height: int = None
    mask_width: int = None
    promote_type: str = "mask"
    labels: dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            name = label
            box = box # .numpy().tolist()
            new_annotation = MaskObject(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3])
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d


    # TODO: Convert bool operation to float logits
    def update_masks(
            self,
            new_mask_dict,
            iou_threshold=0.8,
            objects_count=0) -> int:
        updated_masks = {}

        tracking_mask_ids = list(new_mask_dict.labels.keys())
        print(f'Merging masks PREV: {list(self.labels.keys())} NEW: {tracking_mask_ids}')

        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
            if seg_mask.mask.sum() == 0:
                continue

            new_mask_copy = MaskObject(
                instance_id=seg_obj_id,
                mask=seg_mask.mask.bool(),
                class_name=seg_mask.class_name)

            for object_id, object_info in new_mask_dict.labels.items():  # grounded_sam masks
                iou = calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
                print(f"iou {seg_obj_id} {seg_mask.class_name} - {object_id} {object_info.class_name} : {iou}")

                # bbox_old = calculate_bbox(seg_mask.mask)
                # bbox_new = calculate_bbox(object_info.mask)
                # old_contains_new = bbox_contains(bbox_old, bbox_new, 0.1)
                # new_contains_old = bbox_contains(bbox_new, bbox_old, 0.1)
                # contains = ((old_contains_new or new_contains_old) and seg_mask.class_name == object_info.class_name)
                # print(f"old_contains_new: {old_contains_new}, new_contains_old: {new_contains_old}")

                if iou > iou_threshold:
                    new_mask_copy.mask = new_mask_copy.mask | object_info.mask.bool()
                    print(f'combining masks for object {seg_obj_id} and {object_id} with iou {iou}')
                    object_id in tracking_mask_ids and tracking_mask_ids.remove(object_id)

            updated_masks[seg_obj_id] = new_mask_copy

        print('Remaining tracking objects', tracking_mask_ids)
        for tracking_id in tracking_mask_ids:
            objects_count += 1
            new_tracking_id = objects_count

            print(f'adding new mask object with id {new_tracking_id}')
            new_mask_copy = MaskObject(
                instance_id=new_tracking_id,
                mask=new_mask_dict.labels[tracking_id].mask.bool(),
                class_name=new_mask_dict.labels[tracking_id].class_name)

            if new_mask_copy.mask.sum() == 0:
                continue

            updated_masks[new_tracking_id] = new_mask_copy

        print('updated_masks', list(updated_masks.keys()))

        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit

    def to_dict(self):
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): MaskObject(**v) for k, v in data["labels"].items()}
        return self




@dataclass
class MaskObject:
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def update_box(self):
        bbox = calculate_bbox(self.mask)
        if bbox is not None:
            self.x1 = bbox[0]
            self.y1 = bbox[1]
            self.x2 = bbox[2]
            self.y2 = bbox[3]
    
    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }