#!/usr/bin/env python3
"""
SAM3 Agent Inference Module

This module provides agent-based SAM3 inference using an MLLM to iteratively
refine segmentation results.
"""

import logging
import os
from functools import partial
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils import ModelResult


def setup_agent_components(
    sam3_annotator,
    llm_server_url: str,
    llm_model: str,
    llm_api_key: str = "DUMMY_API_KEY"
):
    """
    Setup SAM3 agent components (LLM client and SAM3 client)

    Args:
        sam3_annotator: SAM3 annotator instance
        llm_server_url: URL of the LLM server
        llm_model: Model name/identifier
        llm_api_key: API key for LLM server

    Returns:
        Tuple of (send_generate_request, call_sam_service, llm_config)
    """
    from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
    from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig

    # Create partial functions with fixed parameters
    send_generate_request = partial(
        send_generate_request_orig,
        server_url=llm_server_url,
        model=llm_model,
        api_key=llm_api_key
    )

    call_sam_service = partial(
        call_sam_service_orig,
        sam3_processor=sam3_annotator.processor
    )

    llm_config = {
        "provider": "vllm",
        "model": llm_model,
        "name": llm_model.split("/")[-1] if "/" in llm_model else llm_model,
        "api_key": llm_api_key
    }

    return send_generate_request, call_sam_service, llm_config


def run_agent_inference_for_class(
    image_path: str,
    text_prompt: str,
    send_generate_request,
    call_sam_service,
    llm_config: dict,
    output_dir: str,
    logger: logging.Logger
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run SAM3 agent inference for a single class

    Args:
        image_path: Path to the image file
        text_prompt: Text prompt for segmentation
        send_generate_request: LLM request function
        call_sam_service: SAM3 service function
        llm_config: LLM configuration dict
        output_dir: Output directory for agent artifacts
        logger: Logger instance

    Returns:
        Tuple of (masks, scores) where masks are boolean numpy arrays
    """
    from sam3.agent.inference import run_single_image_inference
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools is required for agent mode. Install with: pip install pycocotools")

    try:
        logger.debug(f"Running agent inference for prompt: '{text_prompt}'")

        # Create class-specific output directory
        class_output_dir = os.path.join(output_dir, "agent_output", text_prompt.replace(" ", "_"))
        os.makedirs(class_output_dir, exist_ok=True)

        # Run agent inference
        output_image_path = run_single_image_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            llm_config=llm_config,
            send_generate_request=send_generate_request,
            call_sam_service=call_sam_service,
            output_dir=class_output_dir,
            debug=False
        )

        # Load the results
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        prompt_for_filename = text_prompt.replace("/", "_").replace(" ", "_")
        base_filename = f"{image_basename}_{prompt_for_filename}_agent_{llm_config['name']}"
        output_json_path = os.path.join(class_output_dir, f"{base_filename}_pred.json")

        if not os.path.exists(output_json_path):
            logger.warning(f"Agent inference did not produce results for '{text_prompt}'")
            return [], []

        import json
        with open(output_json_path, 'r') as f:
            results = json.load(f)

        # Extract masks and scores
        pred_masks_rle = results.get("pred_masks", [])
        pred_scores = results.get("pred_scores", [])
        orig_img_h = results.get("orig_img_h")
        orig_img_w = results.get("orig_img_w")

        # Decode RLE masks to boolean numpy arrays
        masks = []
        scores = []
        for rle_mask, score in zip(pred_masks_rle, pred_scores):
            # Decode RLE to boolean mask using pycocotools
            rle_dict = {"size": [orig_img_h, orig_img_w], "counts": rle_mask}
            mask = mask_utils.decode(rle_dict)
            masks.append(mask.astype(bool))
            scores.append(float(score))

        logger.debug(f"Agent inference found {len(masks)} masks for '{text_prompt}'")
        return masks, scores

    except Exception as e:
        logger.error(f"Error during agent inference for '{text_prompt}': {e}")
        return [], []


def run_agent_inference(
    image_path: str,
    classes_to_segment: List[str],
    sam3_annotator,
    llm_server_url: str,
    llm_model: str,
    output_dir: str,
    logger: logging.Logger
) -> Tuple[List[ModelResult], List[ModelResult], List[ModelResult]]:
    """
    Run SAM3 agent inference for multiple classes

    Args:
        image_path: Path to the image file
        classes_to_segment: List of class names to segment
        sam3_annotator: SAM3 annotator instance
        llm_server_url: URL of the LLM server
        llm_model: Model name/identifier
        output_dir: Output directory for agent artifacts
        logger: Logger instance

    Returns:
        Tuple of (all_results, car_results, person_results)
    """
    # Setup agent components
    send_generate_request, call_sam_service, llm_config = setup_agent_components(
        sam3_annotator=sam3_annotator,
        llm_server_url=llm_server_url,
        llm_model=llm_model
    )

    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    all_results = []
    car_results = []
    person_results = []

    for class_name in classes_to_segment:
        # Use specialized prompt for "person" class
        if class_name == "person":
            text_prompt = "all people, except those displayed on the posters"
        else:
            text_prompt = class_name

        logger.info(f"Running agent inference for: {text_prompt}")

        masks, scores = run_agent_inference_for_class(
            image_path=image_path,
            text_prompt=text_prompt,
            send_generate_request=send_generate_request,
            call_sam_service=call_sam_service,
            llm_config=llm_config,
            output_dir=output_dir,
            logger=logger
        )

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
                class_name=class_name.replace(" ", "_"),
                confidence=score,
                mask=mask
            )
            all_results.append(result)
            logger.debug(f"  Detected {class_name} with confidence {score:.3f}")

            # Keep track of cars and persons separately for passenger detection
            if class_name == "car":
                car_results.append(result)
            elif class_name == "person":
                person_results.append(result)

    return all_results, car_results, person_results
