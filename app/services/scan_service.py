import logging
from typing import Optional

from app.core.config import get_settings
from app.schemas.scan import DetectionItem, ScanResponse
from app.services.classifier import (
    GeminiNotConfidentError,
    GeminiPermanentError,
    WasteClassifier,
)
from app.services.detector import ObjectDetector

logger = logging.getLogger(__name__)


class ScanService:
    """
    Orchestration service — Gemini-primary, YOLO-fallback.

    Flow
    ----
    1. Gemini Vision  → identifies + classifies waste item from the raw image.
       On success     → return result immediately (YOLO never runs).
       On failure     → fall through to YOLO.
    2. YOLO           → detects waste-relevant COCO objects (whitelist enforced).
       On success     → classify via deterministic rules.
       On failure     → fall through to clean error response.
    3. Both failed    → return low_confidence=True, empty objects.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        classifier: Optional[WasteClassifier] = None,
    ):
        self.detector = detector or ObjectDetector()
        self.classifier = classifier or WasteClassifier()

    def process_image(
        self, image_bytes: bytes, force_low_confidence: bool = False
    ) -> ScanResponse:
        """Execute end-to-end Gemini-primary detection and classification."""
        logger.info("[SCAN] Starting image classification")

        if force_low_confidence:
            return ScanResponse(
                success=True,
                low_confidence=True,
                message="Low confidence: No clear waste object identified.",
                objects=[],
            )

        # ── Step 1: Gemini Vision (primary) ──────────────────────────────
        logger.info("[GEMINI] Attempting primary vision classification")
        gemini_result = self._try_gemini(image_bytes)
        if gemini_result is not None:
            return gemini_result

        # ── Step 2: YOLO fallback ─────────────────────────────────────────
        logger.info("[FALLBACK] Running YOLO object detection")
        yolo_result = self._try_yolo(image_bytes)
        if yolo_result is not None:
            return yolo_result

        # ── Step 3: Both failed ───────────────────────────────────────────
        logger.warning(
            "[SCAN] Both Gemini and YOLO failed to identify a waste object."
        )
        return ScanResponse(
            success=True,
            low_confidence=True,
            message=(
                "Unable to identify a waste item in this image. "
                "Please upload a clearer photo with a single waste object."
            ),
            objects=[],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_gemini(self, image_bytes: bytes) -> Optional[ScanResponse]:
        """
        Attempt Gemini Vision classification.
        Returns a ScanResponse on success, None on any failure.
        """
        try:
            object_name, classification = self.classifier.scan_image(image_bytes)
            logger.info(f"[GEMINI] Success: {object_name}")
            logger.info("[SCAN] Final source: GEMINI")
            detection = DetectionItem(
                id="obj-1",
                object_name=object_name.strip().title(),
                confidence=1.0,   # Gemini doesn't provide a numeric score
                box=None,
                classification=classification,
            )
            return ScanResponse(success=True, low_confidence=False, objects=[detection])

        except GeminiPermanentError as exc:
            logger.error(f"[GEMINI] Permanent failure (will not retry): {exc}")
        except GeminiNotConfidentError as exc:
            logger.info(f"[GEMINI] Not confident: {exc}")
        except Exception as exc:
            logger.error(f"[GEMINI] Unexpected failure: {exc}")

        return None

    def _try_yolo(self, image_bytes: bytes) -> Optional[ScanResponse]:
        """
        Attempt YOLO detection + rule-based classification.
        Returns a ScanResponse on success, None if no valid waste class found.
        """
        try:
            detections = self.detector.detect(image_bytes)
        except Exception as exc:
            logger.error(f"[YOLO] Detection error: {exc}")
            return None

        if not detections:
            logger.info("[YOLO] No waste-relevant objects detected.")
            return None

        enriched = []
        for item in detections:
            logger.info(
                f"[YOLO] Detection: {item.object_name} confidence={item.confidence}"
            )
            try:
                item.classification = self.classifier.classify_item(item.object_name)
            except Exception as exc:
                logger.error(
                    f"[YOLO] Classification failed for '{item.object_name}': {exc}"
                )
            enriched.append(item)

        if not enriched:
            return None

        logger.info("[SCAN] Final source: YOLO")
        return ScanResponse(success=True, low_confidence=False, objects=enriched)

