import io
import logging
from typing import List

from PIL import Image

from app.core.config import get_settings
from app.schemas.scan import DetectionItem

logger = logging.getLogger(__name__)

# COCO class names that correspond to real-world waste items.
# Non-waste COCO classes (airplane, person, car, dog, chair, etc.)
# are intentionally excluded so they never reach classification.
WASTE_RELEVANT_COCO_CLASSES: frozenset = frozenset({
    # Containers / Packaging
    "bottle", "cup", "bowl", "vase",
    # Food / Organic
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    # Paper / Media
    "book",
    # Small household items that become waste
    "scissors", "toothbrush",
})


class ObjectDetector:
    """
    YOLO Object Detection Service — used as FALLBACK only when Gemini fails.
    Sole responsibility: identify physical objects present in the image.
    Results are filtered to waste-relevant COCO classes only.
    Does NOT classify wet/dry, recyclability, or disposal bins.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = None):
        settings = get_settings()
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.CONFIDENCE_THRESHOLD
        )
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model from: {self.model_path}")
                self._model = YOLO(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
                self._model = None

    def detect(self, image_bytes: bytes) -> List[DetectionItem]:
        """
        Runs YOLO object detection on image bytes.
        Returns only detections whose COCO class is in WASTE_RELEVANT_COCO_CLASSES.
        """
        self._load_model()
        if self._model is None:
            logger.warning("[YOLO] Model not loaded. Returning empty detections.")
            return []

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = self._model.predict(
                image,
                conf=self.confidence_threshold,
                verbose=False,
            )

            detections: List[DetectionItem] = []
            if not results:
                return detections

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return detections

            names = result.names or {}

            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                raw_label = names.get(cls_id, f"object_{cls_id}")
                label = raw_label.strip().replace("_", " ").lower()

                # Enforce waste-class whitelist
                if label not in WASTE_RELEVANT_COCO_CLASSES:
                    logger.info(
                        f"[YOLO] Discarding non-waste COCO class '{raw_label}' "
                        f"(confidence={conf:.3f}) — not in waste whitelist."
                    )
                    continue

                xyxy = (
                    [float(coord) for coord in box.xyxy[0].tolist()]
                    if box.xyxy is not None
                    else None
                )

                detections.append(
                    DetectionItem(
                        id=f"obj-{idx + 1}",
                        object_name=label.title(),
                        confidence=round(conf, 3),
                        box=xyxy,
                    )
                )

            return detections
        except Exception as err:
            logger.error(f"[YOLO] Error during detection: {err}")
            return []

