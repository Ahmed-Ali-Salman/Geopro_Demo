"""Image-level text cleanup before OCR.

Pre-processing filters to improve OCR accuracy on MMS-captured images:
- Deskew / perspective correction
- Contrast enhancement (low-light)
- Deblur for motion blur
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_for_ocr(
    image: np.ndarray,
    *,
    deskew: bool = True,
    enhance_contrast: bool = True,
    deblur: bool = False,
) -> np.ndarray:
    """Apply a chain of preprocessing steps to improve OCR accuracy.

    Parameters
    ----------
    image : BGR numpy array (crop of a sign / billboard)
    deskew : correct rotation
    enhance_contrast : apply CLAHE for low-light improvements
    deblur : apply Wiener-like deblur (experimental)

    Returns
    -------
    Preprocessed BGR image
    """
    result = image.copy()

    if enhance_contrast:
        result = apply_clahe(result)

    if deskew:
        result = correct_skew(result)

    if deblur:
        result = sharpen(result)

    return result


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    Works on the L channel of LAB colour space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result


def correct_skew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct slight rotation / skew in a sign crop.

    Uses the Hough Transform to detect dominant lines and
    rotates to correct.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image

    # Compute median angle of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.5:
        return image  # negligible skew

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.debug("Corrected skew by %.1f degrees", median_angle)
    return rotated


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply unsharp masking to counteract motion blur."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened


def crop_detection_region(
    image: np.ndarray,
    bbox: np.ndarray,
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """Crop a detection bounding box from the full image with optional padding.

    Parameters
    ----------
    image : full panoramic image (BGR)
    bbox : [x1, y1, x2, y2] in pixels
    padding_ratio : fractional padding around the bbox

    Returns
    -------
    Cropped BGR sub-image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)

    # Add padding
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image[y1:y2, x1:x2].copy()
