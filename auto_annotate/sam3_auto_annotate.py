#!/usr/bin/env python3
"""
SAM3 Auto-Annotation Script

This script processes images with SAM3 to generate instance segmentation masks
for cars, license plates, and persons. It saves results in the GVA metadata format.

Usage:
    python sam3_auto_annotate.py --input_top_dir <input_dir> --out_top_dir <output_dir>
"""

import argparse
import glob
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Force unbuffered output for real-time progress bar display
os.environ['PYTHONUNBUFFERED'] = '1'

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# For JSON manipulation
import json


class ModelResult:
    """Simple container for model inference results"""
    def __init__(self, class_name: str, confidence: float, mask: np.ndarray):
        self.class_name = class_name
        self.confidence = confidence
        self.mask = mask


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bounding box from a mask.

    Returns:
        Tuple of (x1, y1, x2, y2) or None if mask is empty
    """
    if not np.any(mask):
        return None

    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()

    return (int(x1), int(y1), int(x2), int(y2))


def encode_binary_masks(masks: List[np.ndarray],
                        width: int,
                        height: int) -> Tuple[np.ndarray, List[int]]:
    """
    Encode multiple binary NumPy arrays (boolean masks) into a single NumPy array (uint8),
    assigning each mask an evenly distributed unique value between 0-255.
    Largest masks (by area) are encoded first, smaller masks later.

    Args:
        masks: List of NumPy boolean arrays where False=0 and True=1
        width: Width of image
        height: Height of image

    Returns:
        Tuple containing:
        - combined_image: NumPy array with encoded masks
        - mask_values: List of mask values corresponding to each input mask
    """
    num_masks = len(masks)
    if num_masks > 255:
        raise ValueError("Cannot encode more than 255 masks into a single image")

    combined_image = np.zeros((height, width), dtype=np.uint8)

    if num_masks > 0:
        # Calculate evenly distributed values for each mask
        step_size = 255 // num_masks
        values = [step_size * (i + 1) for i in range(num_masks)]

        # Calculate area of each mask
        areas = [np.sum(mask) for mask in masks]

        # Sort by area (descending)
        indexed_areas = [(i, area) for i, area in enumerate(areas)]
        indexed_areas.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_areas]

        # Process masks in decreasing order of area
        sorted_masks = [masks[i] for i in sorted_indices]

        # Apply unique values for each mask (largest first)
        for i, mask in enumerate(sorted_masks):
            h, w = mask.shape
            if w != width or h != height:
                raise ValueError(f"Unexpected mask resolution {w}x{h}. Expected {width}x{height}")

            combined_image[mask] = values[i]

        # Maintain the original order of mask_values to correspond with input masks
        mask_values = [0] * num_masks
        for i, orig_idx in enumerate(sorted_indices):
            mask_values[orig_idx] = values[i]
    else:
        mask_values = []

    return combined_image, mask_values


def detect_overlapping_masks(masks: List[np.ndarray],
                             iou_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Detect pairs of masks that overlap with IoU greater than the threshold.

    Args:
        masks: List of instance segmentation masks as boolean numpy arrays
        iou_threshold: Threshold for IoU above which masks are considered to overlap

    Returns:
        List of tuples (i, j, iou) of overlapping mask pairs
    """
    n = len(masks)
    overlaps = []

    if n < 2:
        return overlaps

    # Pre-compute mask areas
    areas = np.array([np.sum(mask) for mask in masks])

    for i in range(n):
        mask_i = masks[i]
        area_i = areas[i]

        for j in range(i + 1, n):
            mask_j = masks[j]
            area_j = areas[j]

            # Quick check to skip unnecessary calculations
            smaller_area = min(area_i, area_j)
            larger_area = max(area_i, area_j)
            max_possible_iou = smaller_area / larger_area

            if max_possible_iou < iou_threshold:
                continue

            # Compute intersection
            intersection = np.logical_and(mask_i, mask_j).sum()

            if intersection == 0:
                continue

            # Calculate IoU
            union = area_i + area_j - intersection
            iou = intersection / union

            if iou > iou_threshold:
                overlaps.append((i, j, float(iou)))

    return overlaps


def filter_out_overlapping_masks(inference_result: List[ModelResult]) -> Tuple[List[ModelResult], int]:
    """
    Filter out masks/detections where there is significant overlap.

    Returns:
        Tuple of (filtered_results, num_removed)
    """
    overlaps = detect_overlapping_masks(
        masks=[r.mask for r in inference_result],
        iou_threshold=0.5
    )

    indices_to_remove = set()
    for i, j, overlap in overlaps:
        # Remove one of the overlapping masks (keep the first one)
        indices_to_remove.add(j)

    filtered_result = [item for i, item in enumerate(inference_result) if i not in indices_to_remove]

    return filtered_result, len(indices_to_remove)


def detect_passengers(
    person_results: List[ModelResult],
    car_results: List[ModelResult],
    logger: logging.Logger
) -> List[ModelResult]:
    """
    Detect passengers: persons that are inside cars with closed doors.

    A person is classified as a passenger if:
    1. The person's mask is significantly inside a car's mask (>60% overlap with car)
    2. The person is mostly contained within the car's bounding box
    3. The car door appears closed (person is well inside, not on the edge)

    Args:
        person_results: List of detected persons
        car_results: List of detected cars
        logger: Logger instance

    Returns:
        List of ModelResult objects classified as passengers
    """
    passenger_results = []

    if not person_results or not car_results:
        return passenger_results

    for person_result in person_results:
        person_mask = person_result.mask
        person_area = np.sum(person_mask)

        if person_area == 0:
            continue

        # Get person bounding box
        person_bbox = mask_to_bbox(person_mask.astype(np.uint8))
        if person_bbox is None:
            continue

        px1, py1, px2, py2 = person_bbox
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2

        # Check each car to see if person is inside
        max_containment = 0.0
        best_car_mask = None

        for car_result in car_results:
            car_mask = car_result.mask

            # Get car bounding box
            car_bbox = mask_to_bbox(car_mask.astype(np.uint8))
            if car_bbox is None:
                continue

            cx1, cy1, cx2, cy2 = car_bbox

            # Quick check: is person center inside car bbox?
            if not (cx1 <= person_center_x <= cx2 and cy1 <= person_center_y <= cy2):
                continue

            # Calculate how much of the person is inside the car
            person_inside_car = np.logical_and(person_mask, car_mask)
            intersection_area = np.sum(person_inside_car)
            containment_ratio = intersection_area / person_area

            # Check if person bbox is well inside car bbox (not on edge)
            # Allow some tolerance (10 pixels) from car edges
            tolerance = 10
            is_well_inside = (
                px1 > cx1 + tolerance and
                py1 > cy1 + tolerance and
                px2 < cx2 - tolerance and
                py2 < cy2 - tolerance
            )

            # Person is a passenger if:
            # - >60% of person is inside car mask, OR
            # - >40% inside car and person bbox is well inside car bbox
            if containment_ratio > 0.60 or (containment_ratio > 0.40 and is_well_inside):
                if containment_ratio > max_containment:
                    max_containment = containment_ratio
                    best_car_mask = car_mask

        # If person is inside a car, classify as passenger
        if best_car_mask is not None:
            logger.debug(
                f"Detected passenger with {max_containment*100:.1f}% containment"
            )
            passenger_result = ModelResult(
                class_name="passenger",
                confidence=person_result.confidence,
                mask=person_result.mask
            )
            passenger_results.append(passenger_result)

    return passenger_results


class SAM3Annotator:
    """Wrapper for SAM3 model inference"""

    def __init__(self, device: str = "cuda", logger: logging.Logger = None):
        """
        Initialize SAM3 model

        Args:
            device: Device to run on ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("Loading SAM3 model...")
        self.model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            enable_inst_interactivity=False,
            compile=False
        )
        self.processor = Sam3Processor(self.model, resolution=1008)
        self.logger.info("SAM3 model loaded successfully")

    def segment_image(self, image: np.ndarray, text_prompt: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run SAM3 inference on an image with a text prompt

        Args:
            image: RGB image as numpy array (H, W, 3)
            text_prompt: Text prompt for segmentation (e.g., "car")

        Returns:
            Tuple of (masks, scores) where masks are boolean numpy arrays and scores are confidence values
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Run SAM3 inference
        state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(text_prompt, state)

        # Extract masks and scores
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        # Filter masks by confidence threshold
        confidence_threshold = 0.4
        filtered_masks = []
        filtered_scores = []

        if len(masks) > 0:
            for mask, score in zip(masks, scores):
                if score >= confidence_threshold:
                    # Convert mask to boolean numpy array
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()

                    # Ensure mask is 2D and boolean
                    if len(mask.shape) == 3:
                        mask = mask[0]  # Take first channel if 3D

                    mask = mask.astype(bool)
                    filtered_masks.append(mask)
                    filtered_scores.append(float(score))

        return filtered_masks, filtered_scores


def load_metadata(metadata_path: str) -> dict:
    """Load frame metadata from JSON file"""
    with open(metadata_path, 'r') as fd:
        metadata = json.load(fd)
    return metadata


def save_metadata(metadata: dict, output_path: str) -> None:
    """Save frame metadata to JSON file"""
    with open(output_path, 'w') as output_file:
        json.dump(metadata, output_file, indent=2)


def process_image(
    metadata_path: str,
    sam3_annotator: SAM3Annotator,
    out_top_dir: str,
    input_top_dir: str,
    override_previous: bool,
    logger: logging.Logger
) -> bool:
    """
    Process a single image: run SAM3 inference and save results

    Args:
        metadata_path: Path to input JSON metadata file
        sam3_annotator: SAM3 annotator instance
        out_top_dir: Output top directory
        input_top_dir: Input top directory (for computing relative paths)
        override_previous: Whether to override existing annotations
        logger: Logger instance

    Returns:
        True if processed successfully, False otherwise
    """
    # Load metadata
    try:
        metadata = load_metadata(metadata_path)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return False

    # Find corresponding image file
    img_path = Path(metadata_path).with_suffix(".jpg")
    if not img_path.exists():
        logger.warning(f"No image found at {img_path}")
        return False

    # Compute output paths
    relative_path = os.path.relpath(metadata_path, input_top_dir)
    output_metadata_path = os.path.join(out_top_dir, relative_path)
    output_img_path = Path(output_metadata_path).with_suffix(".jpg")
    output_mask_path = Path(output_metadata_path).with_suffix(".npz")

    if Path(output_metadata_path).exists():
        return True

    # Create output directory
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)

    # Check if already processed
    if not override_previous and output_mask_path.exists():
        logger.debug(f"Skipping {img_path.name} - already processed")
        return True

    # Load image
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Failed to load image from {img_path}")
            return False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {e}")
        return False

    # Run SAM3 inference for each class
    classes_to_segment = ["car", "license plate", "person"]
    all_results = []
    car_results = []
    person_results = []

    for class_name in classes_to_segment:
        logger.debug(f"Segmenting {class_name} in {img_path.name}")
        try:
            masks, scores = sam3_annotator.segment_image(image, class_name)

            # Add results
            for mask, score in zip(masks, scores):
                # Resize mask to original image size if needed
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask.astype(np.uint8) * 255,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    ) > 127

                result = ModelResult(
                    class_name=class_name.replace(" ", "_"),  # "license plate" -> "license_plate"
                    confidence=score,  # Use actual SAM3 confidence score
                    mask=mask
                )
                all_results.append(result)
                logger.debug(f"  Detected {class_name} with confidence {score:.3f}")

                # Keep track of cars and persons separately for passenger detection
                if class_name == "car":
                    car_results.append(result)
                elif class_name == "person":
                    person_results.append(result)

        except Exception as e:
            logger.error(f"Error during SAM3 inference for {class_name}: {e}")
            continue

    # Detect passengers: persons inside cars
    passenger_results = detect_passengers(person_results, car_results, logger)

    # Add passengers to all_results
    all_results.extend(passenger_results)

    # Remove persons that are now classified as passengers
    if passenger_results:
        passenger_masks_set = {id(pr.mask) for pr in passenger_results}
        all_results = [r for r in all_results if not (r.class_name == "person" and id(r.mask) in passenger_masks_set)]

    # Filter overlapping masks
    all_results, num_removed = filter_out_overlapping_masks(all_results)
    if num_removed > 0:
        logger.debug(f"Removed {num_removed} overlapping masks from {img_path.name}")

    # Prepare masks for encoding
    masks = []
    class_names = []
    confidences = []

    for detection in all_results:
        bbox = mask_to_bbox(mask=detection.mask.astype(np.uint8))
        if bbox is None:
            continue
        masks.append(detection.mask)
        class_names.append(detection.class_name)
        confidences.append(detection.confidence)

    # Encode masks
    try:
        combined_masks, mask_values = encode_binary_masks(
            masks=masks,
            width=width,
            height=height
        )
    except Exception as e:
        logger.error(f"Error encoding masks for {img_path.name}: {e}")
        return False

    # Create class_name to mask_values and confidences mappings
    class_name_to_mask_values = {}
    class_name_to_confidences = {}

    for class_name, mask_value, confidence in zip(class_names, mask_values, confidences):
        if class_name not in class_name_to_mask_values:
            class_name_to_mask_values[class_name] = [mask_value]
            class_name_to_confidences[class_name] = [round(confidence, 3)]
        else:
            class_name_to_mask_values[class_name].append(mask_value)
            class_name_to_confidences[class_name].append(round(confidence, 3))

    # Save NPZ file
    try:
        np.savez_compressed(str(output_mask_path), mask=combined_masks)
    except Exception as e:
        logger.error(f"Error saving NPZ file {output_mask_path}: {e}")
        return False

    # Copy JPG file
    try:
        shutil.copy2(str(img_path), str(output_img_path))
    except Exception as e:
        logger.error(f"Error copying image file: {e}")
        return False

    # Update metadata with segmentation info
    annotation_date = datetime.today().strftime('%Y-%m-%d')

    # Create segmentation annotation structure as dictionary
    segmentation_annotation = {
        "annotation_date": annotation_date,
        "considered_classes": ["car", "license_plate", "person", "passenger"],
        "annotator": {
            "annotator_type": "Model",
            "model_id": "SAM3",
            "model_metadata": {
                "model_type": "segmentation",
                "model_name": "SAM3",
                "classes": {
                    "0": "car",
                    "1": "license_plate",
                    "2": "person",
                    "3": "passenger"
                }
            }
        },
        "labels": {
            "annotation_format": "all_masks_on_single_array",
            "annotation_format_version": "1.0",
            "path": str(output_mask_path.name),
            "properties": class_name_to_mask_values,
            "confidences": class_name_to_confidences
        },
        "review": None
    }

    # Update metadata annotation
    if "annotation" not in metadata or metadata["annotation"] is None:
        metadata["annotation"] = {"segmentation": [segmentation_annotation]}
    else:
        # Clear old segmentation annotations
        metadata["annotation"]["segmentation"] = [segmentation_annotation]

    # Save metadata
    try:
        save_metadata(metadata, output_metadata_path)
    except Exception as e:
        logger.error(f"Error saving metadata to {output_metadata_path}: {e}")
        return False

    logger.info(f"Successfully processed {img_path.name} - found {len(masks)} masks")
    return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using SAM3 for car, license plate, and person segmentation"
    )

    parser.add_argument(
        "--input_top_dir",
        required=True,
        help="Path to input directory containing JPG images and JSON metadata files"
    )

    parser.add_argument(
        "--out_top_dir",
        required=True,
        help="Path to output directory where results will be saved"
    )

    parser.add_argument(
        "--override_previous_annotations",
        action="store_true",
        default=False,
        help="Override existing annotations"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run SAM3 on"
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not os.path.isdir(args.input_top_dir):
        logger.error(f"Input directory does not exist: {args.input_top_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.out_top_dir, exist_ok=True)

    logger.info(f"Input directory: {args.input_top_dir}")
    logger.info(f"Output directory: {args.out_top_dir}")
    logger.info(f"Device: {args.device}")

    # Initialize SAM3 annotator
    sam3_annotator = SAM3Annotator(device=args.device, logger=logger)

    # Find all JSON metadata files
    metadata_paths = sorted(glob.glob(
        os.path.join(args.input_top_dir, "**/*.json"),
        recursive=True
    ))

    if not metadata_paths:
        logger.warning(f"No JSON files found in {args.input_top_dir}")
        return

    logger.info(f"Found {len(metadata_paths)} JSON files to process")

    # Process each image
    success_count = 0
    fail_count = 0

    # Temporarily reduce logging level during processing to avoid interference with progress bar
    original_log_level = logger.level
    if logger.level <= logging.INFO:
        logger.setLevel(logging.WARNING)

    # Create progress bar (use stderr to avoid conflicts with logging)
    # Force unbuffered output by setting mininterval=0
    with tqdm(total=len(metadata_paths), desc="Processing images", unit="img",
              ncols=100, file=sys.stderr, dynamic_ncols=False,
              mininterval=0, maxinterval=0.5) as pbar:
        for metadata_path in metadata_paths:
            success = process_image(
                metadata_path=metadata_path,
                sam3_annotator=sam3_annotator,
                out_top_dir=args.out_top_dir,
                input_top_dir=args.input_top_dir,
                override_previous=args.override_previous_annotations,
                logger=logger
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            # Update progress bar immediately and force refresh
            pbar.update(1)
            pbar.set_postfix({'✓': success_count, '✗': fail_count}, refresh=True)
            # Force flush stderr to ensure immediate display
            sys.stderr.flush()

    # Restore logging level
    logger.setLevel(original_log_level)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
