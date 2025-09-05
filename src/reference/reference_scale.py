#!/usr/bin/env python3
"""
Reference Scale Detection and Calibration

Detects known reference objects (e.g., credit card, spoon) in an image
to compute a pixel-to-millimeter scale that improves volume estimation.

This module provides a simple, pluggable interface and a default
OpenCV-based heuristic implementation that works with bounding-box or
mask measurements. It can be replaced by a learned detector later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2


@dataclass
class ReferenceObjectMeasurement:
    """Measurement for a detected reference object in image space."""

    object_type: str
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    pixel_length: float
    known_real_length_mm: float
    scale_mm_per_pixel: float
    confidence: float


class ReferenceObjectDetectorInterface:
    """Interface for reference object detection to compute image scale."""

    def detect(self, image: np.ndarray) -> List[ReferenceObjectMeasurement]:
        raise NotImplementedError

    def get_supported_objects(self) -> List[str]:
        raise NotImplementedError


class ReferenceObjectCatalog:
    """Known real-world sizes for supported reference objects (in mm)."""

    # Longest dimension used for scale by default
    DEFAULT_SIZES_MM: Dict[str, float] = {
        # ISO/IEC 7810 ID-1 (credit card): 85.60mm × 53.98mm → use long side
        "credit_card": 85.60,
        
        # Specific utensil types (from utensil_measurements.py)
        "dinner_spoon": 180.0,    # Standard dinner spoon
        "dessert_spoon": 140.0,   # Dessert spoon
        "teaspoon": 120.0,        # Tea spoon (smallest)
        "dinner_fork": 200.0,     # Standard dinner fork
        "dessert_fork": 160.0,    # Dessert fork
        
        # Chopsticks
        "chopsticks": 240.0,      # Standard chopsticks
        
        # Generic fallbacks
        "spoon": 160.0,           # Average spoon size
        "fork": 180.0,            # Average fork size
        "tablespoon": 180.0,      # Alias for dinner spoon
        
        # Coins
        "coin_quarter": 24.26,    # US quarter coin diameter
    }

    @classmethod
    def get_length_mm(cls, object_type: str) -> Optional[float]:
        return cls.DEFAULT_SIZES_MM.get(object_type)


class HeuristicReferenceObjectDetector(ReferenceObjectDetectorInterface):
    """
    Heuristic detector using color/edge cues and simple assumptions.

    Notes:
      - This is a lightweight placeholder to derive a usable scale when
        a clear rectangular object (card) or elliptical shiny object (spoon)
        is present. Replace with a trained model for production accuracy.
    """

    def __init__(self, candidate_objects: Optional[List[str]] = None):
        self._objects = candidate_objects or list(ReferenceObjectCatalog.DEFAULT_SIZES_MM.keys())

    def get_supported_objects(self) -> List[str]:
        return list(self._objects)

    def detect(self, image: np.ndarray) -> List[ReferenceObjectMeasurement]:
        if image is None or image.size == 0:
            return []

        detections: List[ReferenceObjectMeasurement] = []

        # Try to find a rectangular object (credit card-like)
        rect = self._detect_prominent_rectangle(image)
        if rect is not None:
            x, y, w, h = rect
            pixel_len = float(max(w, h))
            known_len_mm = ReferenceObjectCatalog.get_length_mm("credit_card")
            if known_len_mm and pixel_len > 10:
                detections.append(
                    ReferenceObjectMeasurement(
                        object_type="credit_card",
                        bbox=(x, y, x + w, y + h),
                        mask=None,
                        pixel_length=pixel_len,
                        known_real_length_mm=known_len_mm,
                        scale_mm_per_pixel=known_len_mm / pixel_len,
                        confidence=0.75,
                    )
                )

        # Try to find an elongated bright ellipse (spoon-like)
        ellipse = self._detect_elongated_ellipse(image)
        if ellipse is not None:
            (cx, cy), (ma, mi), angle = ellipse
            pixel_len = float(max(ma, mi))
            known_len_mm = ReferenceObjectCatalog.get_length_mm("spoon")
            if known_len_mm and pixel_len > 10:
                x1 = int(cx - ma / 2)
                y1 = int(cy - mi / 2)
                x2 = int(cx + ma / 2)
                y2 = int(cy + mi / 2)
                detections.append(
                    ReferenceObjectMeasurement(
                        object_type="spoon",
                        bbox=(x1, y1, x2, y2),
                        mask=None,
                        pixel_length=pixel_len,
                        known_real_length_mm=known_len_mm,
                        scale_mm_per_pixel=known_len_mm / pixel_len,
                        confidence=0.55,
                    )
                )

        # Return detections sorted by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_prominent_rectangle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                # ID-1 card aspect ~ 1.585 (85.6/54.0)
                if 1.3 <= aspect <= 1.9 and area > best_area:
                    best = (x, y, w, h)
                    best_area = area
        return best

    def _detect_elongated_ellipse(self, image: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # Highlight bright specular regions (typical for metal spoon)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_ratio = 0
        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, mi), angle = ellipse
            if ma <= 0 or mi <= 0:
                continue
            elongation = max(ma, mi) / (min(ma, mi) + 1e-6)
            if elongation > 2.2 and elongation > best_ratio:
                best = ellipse
                best_ratio = elongation
        return best


def choose_best_scale(detections: List[ReferenceObjectMeasurement]) -> Optional[float]:
    """Pick the most reliable mm-per-pixel scale from detections."""
    if not detections:
        return None
    # Prefer higher confidence and larger pixel length for robustness
    detections = sorted(detections, key=lambda d: (d.confidence, d.pixel_length), reverse=True)
    return detections[0].scale_mm_per_pixel

