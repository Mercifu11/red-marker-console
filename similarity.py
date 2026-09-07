#!/usr/bin/env python3
"""Projection similarity helpers used by the GUI."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def extract_template_projection(image: np.ndarray, red_min=150, green_max=110, blue_max=110) -> list[int]:
    """Extract the red curve from an uploaded projection image."""
    height, width = image.shape[:2]
    projection = [0] * width
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    for x in range(width):
        column_red = red[:, x]
        column_green = green[:, x]
        column_blue = blue[:, x]
        hit = np.where(
            (column_red > red_min)
            & (column_green < green_max)
            & (column_blue < blue_max)
        )[0]
        if hit.size:
            projection[x] = height - int(hit[0])
    return projection


def roi_projection(bgr_frame: np.ndarray, x: int, y: int, width: int, height: int, threshold: int = 180) -> list[int]:
    """Count red pixels per column inside the ROI using the red channel."""
    roi = bgr_frame[y : y + height, x : x + width]
    red = roi[..., 2]
    counts = (red > threshold).sum(axis=0)
    return [int(value) for value in counts]


def resample(values: list[float], target_len: int) -> list[float]:
    if target_len <= 1 or len(values) == target_len:
        return [float(value) for value in values]
    if len(values) == 1:
        return [float(values[0])] * target_len
    result = []
    ratio = (len(values) - 1) / (target_len - 1)
    for index in range(target_len):
        source = index * ratio
        low = int(source)
        high = min(low + 1, len(values) - 1)
        fraction = source - low
        result.append(values[low] * (1 - fraction) + values[high] * fraction)
    return result


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def compare_to_templates(
    test_projection: list[int],
    templates: list[Optional[list[float]]],
    target_len: int,
) -> tuple[list[float], int]:
    test = resample(test_projection, target_len)
    scores = []
    for template in templates:
        if template is None:
            scores.append(0.0)
            continue
        aligned = resample(template, target_len)
        scores.append(cosine_similarity(test, aligned))
    best = int(np.argmax(scores)) if scores else -1
    return scores, best


def similarity_class(score: float, high=0.85, mid=0.60) -> str:
    if score >= high:
        return "high"
    if score >= mid:
        return "mid"
    return "low"
