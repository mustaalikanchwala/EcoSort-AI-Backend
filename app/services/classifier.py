"""
Waste classification service.

Primary path  : scan_image(image_bytes)  — Gemini Vision (google-genai SDK)
Fallback path : classify_item(name)      — deterministic rule-based classifier
"""
import io
import json
import logging
import time
from typing import Optional, Tuple

from PIL import Image

from app.core.config import get_settings
from app.schemas.waste import ClassificationResult

logger = logging.getLogger(__name__)

# HTTP status codes that must NOT be retried (permanent failures).
_PERMANENT_ERROR_CODES = {400, 401, 403, 404}

# Maximum transient retries (429, 5xx, network errors).
_MAX_TRANSIENT_RETRIES = 2


class GeminiPermanentError(Exception):
    """Raised for permanent Gemini failures (4xx) — do not retry."""


class GeminiNotConfidentError(Exception):
    """Raised when Gemini cannot confidently identify a waste item."""


class WasteClassifier:
    """
    Waste classification service.

    Public API
    ----------
    scan_image(image_bytes)    -> (object_name, ClassificationResult)
        Primary path. Uses Gemini Vision to identify AND classify the waste
        item in a single API call. Raises on failure.

    classify_item(object_name) -> ClassificationResult
        Fallback path (called when Gemini fails). Deterministic rule-based
        classification — never calls any external API.
    """

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL

    # ------------------------------------------------------------------
    # Primary: Gemini Vision
    # ------------------------------------------------------------------

    def scan_image(self, image_bytes: bytes) -> Tuple[str, ClassificationResult]:
        """
        Identify and classify a waste item from raw image bytes using Gemini Vision.

        Returns (object_name, ClassificationResult) on success.
        Raises GeminiPermanentError for unrecoverable API errors (403, 401, etc.).
        Raises GeminiNotConfidentError when no clear waste item is visible.
        Raises other exceptions for unexpected failures (caller falls back to YOLO).
        """
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise GeminiPermanentError("No valid Gemini API key configured.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Re-encode as JPEG so we always send a known MIME type.
        jpeg_bytes = self._to_jpeg(image_bytes)

        prompt = (
    "You are an expert AI system for waste object identification and waste segregation.\n\n"

    "Analyze the entire image carefully and identify EVERY clearly visible waste or garbage "
    "object that can reasonably be classified. Do NOT identify only the most prominent object. "
    "If multiple waste objects are present, identify each object separately.\n\n"

    "For each detected object:\n"
    "- Identify the specific object as accurately as possible.\n"
    "- Describe the object using a concise but meaningful name, including its material/form "
    "when visually clear (for example: 'crushed plastic bottle', 'banana peel', "
    "'cardboard box', 'aluminum can').\n"
    "- Determine whether it is Wet or Dry.\n"
    "- Determine whether it is Recyclable or Non-recyclable.\n"
    "- Determine whether it is Biodegradable or Non-biodegradable.\n"
    "- Recommend the most appropriate disposal bin.\n"
    "- Provide the corresponding bin color.\n"
    "- Give a clear 2-sentence explanation describing the classification and appropriate disposal.\n\n"

    "IMPORTANT DETECTION RULES:\n"
    "1. Inspect the entire image before producing the result.\n"
    "2. Detect all clearly visible waste objects, including objects in the background when "
    "they are sufficiently visible to identify.\n"
    "3. Do not merge separate objects into one object.\n"
    "4. If the same type of object appears multiple times as separate physical objects, "
    "return each physical object separately when they can be distinguished.\n"
    "5. Do not invent objects that are not visibly present.\n"
    "6. Ignore people, hands, furniture, scenery, containers, and other non-waste objects "
    "unless they themselves are clearly waste.\n"
    "7. If an object is partially visible but there is enough visual evidence to identify it, "
    "include it with an appropriate confidence value.\n"
    "8. If an object cannot be identified reliably, do not guess. Mark it as uncertain.\n"
    "9. Classification should be based on the actual visible object/material, not merely its color or shape.\n"
    "10. Treat each detected object independently. Different objects may have completely different "
    "waste classifications.\n\n"

    "BIN RULES:\n"
    "- Recyclable Dry Waste → 'Recyclable Dry Waste (Blue Bin)' → '#2563eb'\n"
    "- Organic / Wet Waste → 'Organic / Wet Waste (Green Bin)' → '#16a34a'\n"
    "- Hazardous Waste → 'Hazardous Waste (Red Bin)' → '#dc2626'\n"
    "- General Waste → 'General Waste (Black Bin)' → '#475569'\n\n"

    "CONFIDENCE:\n"
    "Use 'confident': true only when the object can be identified with reasonable visual certainty. "
    "Use 'confident': false when the object is unclear or cannot be reliably identified. "
    "Do not invent a classification for an unidentified object.\n\n"

    "RETURN FORMAT:\n"
    "Return ONLY valid JSON. Do not return markdown, code fences, explanations outside the JSON, "
    "or additional text.\n\n"

    "The JSON must follow exactly this structure:\n"
    "{\n"
    "  \"objects\": [\n"
    "    {\n"
    "      \"object_name\": \"specific waste item name\",\n"
    "      \"confident\": true,\n"
    "      \"wet_dry\": \"Dry\",\n"
    "      \"recyclable\": \"Recyclable\",\n"
    "      \"biodegradable\": \"Non-biodegradable\",\n"
    "      \"recommended_bin\": \"Recyclable Dry Waste (Blue Bin)\",\n"
    "      \"bin_color\": \"#2563eb\",\n"
    "      \"explanation\": \"Clear 2-sentence explanation of why the object has been classified this way and how it should be disposed of.\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"

    "If no clearly identifiable waste object is present, return:\n"
    "{\n"
    "  \"objects\": []\n"
    "}\n"
        )

        def _call() -> str:
            response = client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text.strip()

        raw = self._call_with_retry(_call)
        return self._parse_gemini_response(raw)

    def _call_with_retry(self, fn):
        """
        Call fn() with bounded retry for transient errors.
        Permanent errors (4xx) are re-raised immediately without retry.
        """
        last_exc = None
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                return fn()
            except GeminiPermanentError:
                raise
            except Exception as exc:
                code = self._extract_http_code(exc)
                if code in _PERMANENT_ERROR_CODES:
                    raise GeminiPermanentError(
                        f"Gemini permanent error (HTTP {code}): {exc}"
                    ) from exc
                last_exc = exc
                if attempt < _MAX_TRANSIENT_RETRIES:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        f"[GEMINI] Transient error (attempt {attempt + 1}), "
                        f"retrying in {wait}s: {exc}"
                    )
                    time.sleep(wait)
        raise last_exc

    @staticmethod
    def _extract_http_code(exc: Exception) -> Optional[int]:
        """
        Best-effort extraction of an HTTP status code from a google-genai exception.
        Inspects .code attribute first, then parses the string representation.
        """
        if hasattr(exc, "code") and isinstance(exc.code, int):
            return exc.code
        for token in str(exc).split():
            token = token.strip(".,;:()")
            if token.isdigit() and 400 <= int(token) < 600:
                return int(token)
        return None

    @staticmethod
    def _to_jpeg(image_bytes: bytes) -> bytes:
        """Re-encode image bytes as JPEG so Gemini always receives a known MIME type."""
        buf = io.BytesIO()
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    @staticmethod
    def _parse_gemini_response(raw: str) -> Tuple[str, ClassificationResult]:
        """
        Parse and validate Gemini's JSON response.
        Raises GeminiNotConfidentError if confident=false.
        Raises ValueError on schema mismatch.
        """
        text = raw.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(text)

        if not data.get("confident", True):
            raise GeminiNotConfidentError(
                "Gemini could not confidently identify a waste item in the image."
            )

        object_name = data.get("object_name", "").strip()
        if not object_name:
            raise ValueError("Gemini response missing object_name.")

        result = ClassificationResult(
            wet_dry=data["wet_dry"],
            recyclable=data["recyclable"],
            biodegradable=data["biodegradable"],
            recommended_bin=data["recommended_bin"],
            bin_color=data.get("bin_color", "#475569"),
            explanation=data["explanation"],
        )
        return object_name, result

    # ------------------------------------------------------------------
    # Fallback: deterministic rule-based classification (no API calls)
    # ------------------------------------------------------------------

    def classify_item(self, object_name: str) -> ClassificationResult:
        """
        Classify a named waste item using deterministic rules only.
        Called by the YOLO fallback path — never calls any external service.
        """
        return self._classify_fallback(object_name)

    def _classify_fallback(self, object_name: str) -> ClassificationResult:
        """Rule-based waste classification. No external calls."""
        lower = object_name.lower()

        # Organic / Wet
        if any(w in lower for w in [
            "banana", "apple", "orange", "food", "fruit", "peel",
            "vegetable", "sandwich", "pizza", "donut", "cake", "broccoli",
            "carrot", "hot dog",
        ]):
            return ClassificationResult(
                wet_dry="Wet",
                recyclable="Non-recyclable",
                biodegradable="Biodegradable",
                recommended_bin="Organic / Wet Waste (Green Bin)",
                bin_color="#16a34a",
                explanation=(
                    f"{object_name} is organic food waste. "
                    "Dispose of it in the green compost/wet bin."
                ),
            )

        # Paper / Cardboard
        if any(w in lower for w in ["paper", "cardboard", "book", "newspaper"]):
            return ClassificationResult(
                wet_dry="Dry",
                recyclable="Recyclable",
                biodegradable="Biodegradable",
                recommended_bin="Paper & Cardboard (Blue Bin)",
                bin_color="#2563eb",
                explanation=(
                    f"{object_name} is clean dry paper material. "
                    "Keep it dry and place in the paper recycling bin."
                ),
            )

        # Plastic / Metal / Glass
        if any(w in lower for w in [
            "bottle", "can", "cup", "plastic", "metal", "glass",
            "fork", "spoon", "vase",
        ]):
            return ClassificationResult(
                wet_dry="Dry",
                recyclable="Recyclable",
                biodegradable="Non-biodegradable",
                recommended_bin="Recyclable Dry Waste (Blue Bin)",
                bin_color="#2563eb",
                explanation=(
                    f"{object_name} is non-biodegradable dry recyclable material. "
                    "Rinse lightly and place in recyclables."
                ),
            )

        # Default general waste
        return ClassificationResult(
            wet_dry="Dry",
            recyclable="Non-recyclable",
            biodegradable="Non-biodegradable",
            recommended_bin="General Waste (Black Bin)",
            bin_color="#475569",
            explanation=(
                f"{object_name} is categorized as general solid waste. "
                "Dispose of it in the regular dry waste bin."
            ),
        )
