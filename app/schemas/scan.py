from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.waste import ClassificationResult


class DetectionItem(BaseModel):
    """
    Object detected by YOLO, enriched with Gemini classification.
    """
    id: Optional[str] = Field(default=None, description="Unique item identifier")
    object_name: str = Field(..., description="Name of the physical object detected by YOLO")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    box: Optional[List[float]] = Field(
        default=None,
        description="Bounding box [x1, y1, x2, y2] from YOLO",
    )
    classification: Optional[ClassificationResult] = Field(
        default=None,
        description="Gemini-generated waste classification",
    )


class ScanResponse(BaseModel):
    """
    Final JSON response returned to the frontend.
    """
    success: bool = True
    low_confidence: bool = False
    message: Optional[str] = None
    objects: List[DetectionItem] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
