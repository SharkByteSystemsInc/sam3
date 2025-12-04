#!/usr/bin/env python3
"""
Standard SAM3 Model Inference Module

This module provides standard (non-agent) SAM3 inference for segmentation tasks.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils import ModelResult


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

        # Lazy import to avoid circular dependencies
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

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


def run_standard_inference(
    image: np.ndarray,
    classes_to_segment: List[str],
    sam3_annotator: SAM3Annotator,
    logger: logging.Logger
) -> Tuple[List[ModelResult], List[ModelResult], List[ModelResult]]:
    """
    Run standard SAM3 inference for multiple classes

    Args:
        image: RGB image as numpy array (H, W, 3)
        classes_to_segment: List of class names to segment
        sam3_annotator: SAM3 annotator instance
        logger: Logger instance

    Returns:
        Tuple of (all_results, car_results, person_results)
    """
    height, width = image.shape[:2]
    all_results = []
    car_results = []
    person_results = []

    for class_name in classes_to_segment:
        logger.debug(f"Segmenting {class_name}")
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

    return all_results, car_results, person_results
