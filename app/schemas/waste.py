from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class WetDryEnum(str, Enum):
    DRY = "Dry"
    WET = "Wet"
    UNKNOWN = "Unknown"


class RecyclableEnum(str, Enum):
    RECYCLABLE = "Recyclable"
    NON_RECYCLABLE = "Non-recyclable"
    UNKNOWN = "Unknown"


class BiodegradableEnum(str, Enum):
    BIODEGRADABLE = "Biodegradable"
    NON_BIODEGRADABLE = "Non-biodegradable"
    UNKNOWN = "Unknown"


class ClassificationResult(BaseModel):
    """
    Structured response returned by Gemini for waste classification.
    """
    wet_dry: str = Field(
        ...,
        description="Dry, Wet, or Unknown",
        examples=["Dry", "Wet"],
    )
    recyclable: str = Field(
        ...,
        description="Recyclable, Non-recyclable, or Unknown",
        examples=["Recyclable", "Non-recyclable"],
    )
    biodegradable: str = Field(
        ...,
        description="Biodegradable, Non-biodegradable, or Unknown",
        examples=["Biodegradable", "Non-biodegradable"],
    )
    recommended_bin: str = Field(
        ...,
        description="Recommended disposal bin or category",
        examples=["Recyclable Waste (Blue Bin)", "Organic / Wet Waste (Green Bin)"],
    )
    bin_color: Optional[str] = Field(
        default="#2563eb",
        description="Visual color hex code for bin highlight",
    )
    explanation: str = Field(
        ...,
        description="User-friendly rationale and guidance for correct disposal",
    )
