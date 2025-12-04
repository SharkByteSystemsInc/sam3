#!/usr/bin/env python3
"""
Shared utilities for auto-annotation

This module contains shared classes and utility functions used by both
standard and agent-based inference.
"""

import numpy as np


class ModelResult:
    """Simple container for model inference results"""
    def __init__(self, class_name: str, confidence: float, mask: np.ndarray):
        self.class_name = class_name
        self.confidence = confidence
        self.mask = mask
