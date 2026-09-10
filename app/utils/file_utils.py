import io
from fastapi import HTTPException, UploadFile
from PIL import Image
from app.core.config import get_settings

settings = get_settings()


async def validate_and_read_image(file: UploadFile) -> bytes:
    """
    Validates uploaded image file in-memory.
    Checks MIME type, maximum file size, and PIL image integrity.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file was uploaded.")

    # Validate content-type if provided
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in settings.SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{content_type}'. Please upload a JPEG, PNG, or WebP image.",
        )

    # Read image contents into memory
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    # Check file size limit
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_MB}MB.",
        )

    # Verify that image bytes form a valid, readable image using PIL
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is corrupted or not a valid image.",
        ) from exc

    return content
