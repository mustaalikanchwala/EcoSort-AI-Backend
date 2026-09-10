from fastapi import APIRouter, Depends, File, Query, UploadFile
from app.schemas.scan import ScanResponse
from app.services.scan_service import ScanService
from app.utils.file_utils import validate_and_read_image

router = APIRouter(prefix="/api", tags=["Waste Scan"])


def get_scan_service() -> ScanService:
    return ScanService()


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan and classify waste image",
    description="Uploads a waste photo, runs YOLO object detection, and classifies via Gemini.",
)
async def scan_waste(
    file: UploadFile = File(..., description="Waste image to scan and classify"),
    force_low_confidence: bool = Query(
        False, description="Flag for demonstration/testing low-confidence scenario"
    ),
    service: ScanService = Depends(get_scan_service),
) -> ScanResponse:
    # Validate and read file into memory
    image_bytes = await validate_and_read_image(file)

    # Process in-memory with YOLO and Gemini
    return service.process_image(
        image_bytes=image_bytes,
        force_low_confidence=force_low_confidence,
    )
