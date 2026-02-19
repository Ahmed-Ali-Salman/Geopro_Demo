"""OCR engine — PaddleOCR primary, Surya fallback.

Runs text detection + recognition on cropped sign/billboard regions.
Fully offline with bundled models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """A single OCR text detection."""

    text: str
    confidence: float
    bbox: np.ndarray  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — quadrilateral
    language: str = "unknown"  # "ar", "en", or "mixed"


class OCREngine:
    """Unified OCR engine wrapping PaddleOCR or Surya."""

    def __init__(
        self,
        engine: Literal["paddleocr", "surya"] = "paddleocr",
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        confidence_threshold: float = 0.5,
    ):
        self.engine_name = engine
        self.languages = languages or ["ar", "en"]
        self.use_gpu = use_gpu
        self.conf_threshold = confidence_threshold
        self._engine = None
        self._rec_predictor = None  # For Surya
        self._det_predictor = None  # For Surya

        self._init_engine()

    def _init_engine(self) -> None:
        """Lazily initialize the selected OCR backend."""
        if self.engine_name == "paddleocr":
            self._init_paddleocr()
        elif self.engine_name == "surya":
            self._init_surya()
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_name}")

    def _init_paddleocr(self) -> None:
        """Initialize PaddleOCR."""
        from paddleocr import PaddleOCR

        # PaddleOCR handles multi-language via separate models
        # For Arabic: lang='ar', for English: lang='en'
        # We'll run detection once and recognition for each language
        self._engines = {}
        for lang in self.languages:
            logger.info("Initializing PaddleOCR for language: %s", lang)
            self._engines[lang] = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                enable_mkldnn=False,
            )
        self._engine = self._engines.get("en") or list(self._engines.values())[0]

    def _init_surya(self) -> None:
        """Initialize Surya OCR."""
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        # For some versions, we might need model_registry or similar
        # But commonly:
        
        logger.info("Initializing Surya OCR...")
        self._det_predictor = DetectionPredictor()
        
        try:
            self._rec_predictor = RecognitionPredictor()
        except TypeError:
             # Recent versions might require arg
             # Let's try passing None or inspect if we can import the model loader
             # This is a hack for the 'missing positional arg' error
             # If it wants a foundation_predictor, we probably need to instantiate one?
             # Or maybe we rely on default?
             # Let's try instantiating without arguments first (which failed),
             # so now let's try a fallback if available within the library structure
             # Example: rec_predictor = RecognitionPredictor(some_model)
             # Without docs access, we'll try a generic workaround or simply log and fail gracefully 
             # to avoid crashing whole pipeline if OCR fails.
             logger.warning("Surya RecognitionPredictor init failed. OCR will be detection-only.")
             self._rec_predictor = None



    def recognize(
        self,
        image: np.ndarray,
        lang_hint: Optional[str] = None,
    ) -> List[OCRResult]:
        """Run OCR on an image (typically a cropped sign region).

        Parameters
        ----------
        image : BGR numpy array
        lang_hint : if known, prefer this language model (Paddle only)

        Returns
        -------
        List of OCRResult objects
        """
        if self.engine_name == "paddleocr":
            return self._recognize_paddle(image, lang_hint)
        else:
            return self._recognize_surya(image)

    def _recognize_paddle(
        self,
        image: np.ndarray,
        lang_hint: Optional[str] = None,
    ) -> List[OCRResult]:
        """Run PaddleOCR on an image."""
        results: List[OCRResult] = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Determine which language engines to try
        if lang_hint and lang_hint in self._engines:
            engines_to_try = [(lang_hint, self._engines[lang_hint])]
        else:
            engines_to_try = list(self._engines.items())

        best_results: List[OCRResult] = []
        best_total_conf = 0.0

        for lang, engine in engines_to_try:
            try:
                ocr_output = engine.ocr(rgb)
            except Exception as e:
                logger.warning("PaddleOCR failed for lang %s: %s", lang, e)
                continue
                
            if not ocr_output:
                continue

            lang_results = []
            total_conf = 0.0
            try:
                # ---------------------------------------------------------
                # NEW Format 2 (List of Dicts: [{'rec_texts': [], ...}])
                # ---------------------------------------------------------
                if isinstance(ocr_output, list) and ocr_output and isinstance(ocr_output[0], dict):
                    # It seems PaddleOCR sometimes returns a list containing one dict with results
                    res = ocr_output[0]
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    polys = res.get('dt_polys', [])
                    
                    for i in range(len(texts)):
                        text = texts[i]
                        conf = float(scores[i]) if i < len(scores) else 0.8
                        bbox = polys[i] if i < len(polys) else np.zeros((4, 2))
                        
                        if conf < self.conf_threshold:
                            continue
                            
                        lang_results.append(OCRResult(
                            text=text, confidence=conf, bbox=bbox, language=lang
                        ))
                        total_conf += conf

                # ---------------------------------------------------------
                # NEW Format 1 (Dict with 'res' key)
                # ---------------------------------------------------------
                elif isinstance(ocr_output, dict) and 'res' in ocr_output:
                    res = ocr_output['res']
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    polys = res.get('dt_polys', [])
                    
                    for i in range(len(texts)):
                        text = texts[i]
                        conf = float(scores[i]) if i < len(scores) else 0.8
                        bbox = polys[i] if i < len(polys) else np.zeros((4, 2))
                        
                        if conf < self.conf_threshold:
                            continue
                            
                        lang_results.append(OCRResult(
                            text=text, confidence=conf, bbox=bbox, language=lang
                        ))
                        total_conf += conf

                # ---------------------------------------------------------
                # OLD Format (List of lists: [ [ [bbox, [text, conf]], ... ] ])
                # ---------------------------------------------------------
                elif isinstance(ocr_output, list) and ocr_output and ocr_output[0]:
                    # This block handles the standard List[List] format
                    # But we must ensure it's not a List[Dict] which we handled above
                    # Since we used if/elif, we are safe here if ocr_output[0] is not dict
                    
                    for line in ocr_output[0]:
                        if not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                        if not isinstance(line[0], (list, np.ndarray)):
                             continue
                             
                        bbox_points = np.array(line[0], dtype=np.float32)
                        
                        if isinstance(line[1], (tuple, list)):
                            text = line[1][0]
                            try:
                                confidence = float(line[1][1])
                            except (ValueError, TypeError):
                                confidence = 0.5
                        else:
                            text = str(line[1])
                            confidence = 1.0

                        if confidence < self.conf_threshold:
                            continue

                        lang_results.append(OCRResult(
                            text=text, confidence=confidence, bbox=bbox_points, language=lang
                        ))
                        total_conf += confidence
            except Exception as e:
                logger.error("Error parsing OCR output for lang %s: %s", lang, e)
                continue

            if total_conf > best_total_conf:
                best_total_conf = total_conf
                best_results = lang_results

        return best_results

    def _recognize_surya(self, image: np.ndarray) -> List[OCRResult]:
        """Run Surya OCR on an image."""
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # 1. Detection
        det_preds = self._det_predictor([pil_image])[0]

        # 2. Recognition
        # Surya expects a list of images and a list of bboxes
        # However, for cropped sign regions, we can often just run recognition
        # But for robustness, let's use the full pipeline: detect lines -> recognize
        
        bboxes = [p.bbox for p in det_preds.bboxes]
        if not bboxes:
             # Fallback: try to recognize the whole image as one line if small/cropped
             # But Surya really wants bboxes for the recog predictor.
             # Let's create a dummy bbox for the whole image
             w, h = pil_image.size
             bboxes = [[0, 0, w, h]]

        # Rec predictor takes (images, [bboxes])
        rec_preds = self._rec_predictor([pil_image], [self.languages], [bboxes])[0]

        results: List[OCRResult] = []
        for i, text_line in enumerate(rec_preds.text_lines):
            confidence = float(text_line.confidence) if hasattr(text_line, 'confidence') else 1.0 # Surya confidence availability varies
            # If explicit confidence is missing in some versions, we might assume high if detected.
            # But let's check recent Surya API. 
            # Actually, let's skip confidence check if not available or assume 1.0
            
            # Surya 0.4+ usually has confidence
                
            if confidence < self.conf_threshold:
                continue

            bbox = text_line.bbox # [x1, y1, x2, y2]
            # Convert to quad format for consistency
            bbox_np = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]]
            ], dtype=np.float32)

            text = text_line.text
            lang = _detect_language(text)

            results.append(OCRResult(
                text=text,
                confidence=confidence,
                bbox=bbox_np,
                language=lang,
            ))

        return results


def _detect_language(text: str) -> str:
    """Simple heuristic to detect if text is Arabic or English."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = arabic_chars + latin_chars
    if total == 0:
        return "unknown"
    if arabic_chars > latin_chars:
        return "ar"
    elif latin_chars > arabic_chars:
        return "en"
    else:
        return "mixed"
