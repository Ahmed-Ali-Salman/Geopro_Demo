"""YOLO-based 2D object detector (YOLO11 / YOLO26).

Wraps the Ultralytics YOLO library for inference on panoramic images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single 2D detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray           # [x1, y1, x2, y2] in pixels
    mask: Optional[np.ndarray] = None  # segmentation mask, if available
    image_filename: str = ""

    @property
    def bbox_center(self) -> np.ndarray:
        """Center point of the bounding box."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    @property
    def bbox_area(self) -> float:
        """Area of the bounding box in pixels²."""
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class YOLODetector:
    """YOLO object detector for roadside asset detection."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 1280,
        device: str | int = 0,
    ):
        """Initialize the detector.

        Parameters
        ----------
        model_path : path to YOLO .pt weights
        confidence_threshold : minimum confidence to keep a detection
        iou_threshold : NMS IoU threshold
        image_size : inference input size
        device : 'cpu', 0 (GPU), or list of GPU IDs. Defaults to 0 (primary GPU).
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.imgsz = image_size
        self.device = device

        logger.info("Loading YOLO model from %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded. Classes: %s", self.model.names)

    @property
    def class_names(self) -> dict:
        """Mapping of class_id → class_name from the model."""
        return self.model.names

    def detect(
        self,
        image: np.ndarray,
        image_filename: str = "",
    ) -> List[Detection]:
        """Run detection on a single image.

        Parameters
        ----------
        image : BGR numpy array (H, W, 3)
        image_filename : optional filename for traceability

        Returns
        -------
        List of Detection objects
        """
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        # Mapping COCO -> GeoPro Assets
        # In a real scenario, we'd train a custom model.
        # Here we map available classes to requirements.
        CLASS_MAPPING = {
            # Traffic Signs
            "stop sign": "traffic_sign",
            "traffic light": "traffic_light",
            
            # Poles / Infrastructure
            "parking meter": "speed_camera",  # Demo mapping
            
            # Vehicles
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
            "motorcycle": "vehicle",
            
            # Pedestrians
            "person": "pedestrian",
            
            # Vegetation (Demo)
            "potted plant": "tree",
            
            # Misc
            "bench": "street_furniture",
            "chair": "street_furniture",
            "fire hydrant": "utility_asset",
        }

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                
                # Check mapping
                if raw_name in CLASS_MAPPING:
                    final_name = CLASS_MAPPING[raw_name]
                    # Store original type if valid (e.g. stop sign)
                    # We can use this later for attributes
                else:
                    # Skip unmapped classes to reduce noise (e.g. potted plant, tie)
                    # unless we want everything
                    # detections.append(...) 
                    continue

                # Optional segmentation mask
                mask = None
                if result.masks is not None:
                    try:
                        mask = result.masks.data[i].cpu().numpy()
                    except (IndexError, AttributeError):
                        pass

                det = Detection(
                    class_id=cls_id,
                    class_name=final_name,
                    confidence=conf,
                    bbox=bbox,
                    mask=mask,
                    image_filename=image_filename,
                )
                # Monkey-patch raw name for attribute extraction later
                det.raw_class_name = raw_name
                
                detections.append(det)

        logger.debug(
            "Detected %d objects in %s",
            len(detections),
            image_filename or "image",
        )
        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        filenames: Optional[List[str]] = None,
    ) -> List[List[Detection]]:
        """Run detection on a batch of images.

        Returns
        -------
        List of detection lists, one per image
        """
        if filenames is None:
            filenames = [f"image_{i}" for i in range(len(images))]

        results = self.model.predict(
            source=images,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        # Mapping COCO -> GeoPro Assets (duplicated for batch)
        CLASS_MAPPING = {
            "stop sign": "traffic_sign",
            "traffic light": "traffic_light",
            "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
            "person": "pedestrian",
            "bench": "street_furniture",
            "fire hydrant": "utility_asset",
        }

        all_detections: List[List[Detection]] = []
        for result, fname in zip(results, filenames):
            detections: List[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    raw_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    
                    if raw_name in CLASS_MAPPING:
                        final_name = CLASS_MAPPING[raw_name]
                        det = Detection(
                            class_id=cls_id,
                            class_name=final_name,
                            confidence=float(boxes.conf[i].item()),
                            bbox=boxes.xyxy[i].cpu().numpy(),
                            image_filename=fname,
                        )
                        det.raw_class_name = raw_name
                        detections.append(det)
                        
            all_detections.append(detections)

        logger.info("Batch detection: %d images processed", len(images))
        return all_detections
