"""Arabic OCR post-processing.

Handles Right-to-Left (RTL) reordering, reshaping, and line merging
for Arabic text output from PaddleOCR (which outputs LTR by default).
"""

from __future__ import annotations

import logging
import re
from typing import List

from geopro.ocr.engine import OCRResult

logger = logging.getLogger(__name__)


def postprocess_arabic(results: List[OCRResult]) -> List[OCRResult]:
    """Apply Arabic-specific post-processing to OCR results.

    1. RTL reordering (PaddleOCR outputs Arabic in LTR order)
    2. Arabic letter reshaping for correct ligatures
    3. Multi-line merging for sign text
    """
    processed = []
    for r in results:
        if r.language == "ar" or _has_arabic(r.text):
            r.text = reshape_and_reorder(r.text)
            r.language = "ar"
        processed.append(r)

    return processed


def reshape_and_reorder(text: str) -> str:
    """Reshape and reorder Arabic text for correct RTL display.

    Uses python-bidi for bidirectional reordering and
    arabic-reshaper for proper letter joining.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except ImportError:
        logger.warning(
            "arabic-reshaper or python-bidi not installed. "
            "Arabic text may not display correctly."
        )
        return text


def merge_multiline_text(
    results: List[OCRResult],
    y_tolerance: float = 20.0,
) -> List[OCRResult]:
    """Merge OCR results that appear on consecutive lines of the same sign.

    Groups results by vertical proximity and concatenates text.

    Parameters
    ----------
    results : list of OCR results from a single sign crop
    y_tolerance : maximum vertical pixel gap between lines

    Returns
    -------
    Merged OCR results (potentially fewer items)
    """
    if len(results) <= 1:
        return results

    # Sort by vertical position (top of bbox)
    sorted_results = sorted(results, key=lambda r: r.bbox[:, 1].min())

    merged: List[OCRResult] = []
    current_group: List[OCRResult] = [sorted_results[0]]

    for i in range(1, len(sorted_results)):
        prev_bottom = current_group[-1].bbox[:, 1].max()
        curr_top = sorted_results[i].bbox[:, 1].min()

        if curr_top - prev_bottom <= y_tolerance:
            current_group.append(sorted_results[i])
        else:
            merged.append(_merge_group(current_group))
            current_group = [sorted_results[i]]

    merged.append(_merge_group(current_group))
    return merged


def _merge_group(group: List[OCRResult]) -> OCRResult:
    """Merge a group of OCR results into a single result."""
    if len(group) == 1:
        return group[0]

    # Concatenate text with newlines
    texts = [r.text for r in group]
    combined_text = "\n".join(texts)

    # Average confidence
    avg_conf = sum(r.confidence for r in group) / len(group)

    # Combined bounding box (convex hull of all corners)
    import numpy as np
    all_points = np.vstack([r.bbox for r in group])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    combined_bbox = np.array([
        min_xy,
        [max_xy[0], min_xy[1]],
        max_xy,
        [min_xy[0], max_xy[1]],
    ], dtype=np.float32)

    # Language: majority vote
    lang = max(set(r.language for r in group), key=lambda l: sum(1 for r in group if r.language == l))

    return OCRResult(
        text=combined_text,
        confidence=avg_conf,
        bbox=combined_bbox,
        language=lang,
    )


def _has_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))
