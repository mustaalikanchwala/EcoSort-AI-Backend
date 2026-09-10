import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.schemas.scan import DetectionItem
from app.schemas.waste import ClassificationResult

client = TestClient(app)


def _create_test_image(format_name: str = "PNG") -> io.BytesIO:
    """Helper to generate in-memory test image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(buf, format=format_name)
    buf.seek(0)
    return buf


def test_health_check():
    """Verify /health endpoint returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data


def test_scan_missing_file():
    """Verify 422 Unprocessable Entity when no file is provided."""
    response = client.post("/api/scan")
    assert response.status_code == 422


def test_scan_unsupported_file_type():
    """Verify 400 when file content-type is not supported image."""
    response = client.post(
        "/api/scan",
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


def test_scan_corrupted_image():
    """Verify 400 when uploaded bytes are not a valid readable image."""
    response = client.post(
        "/api/scan",
        files={"file": ("broken.png", b"not-a-valid-image", "image/png")},
    )
    assert response.status_code == 400
    assert "corrupted or not a valid image" in response.json()["detail"]


def test_scan_force_low_confidence():
    """Verify low-confidence response when force_low_confidence is true."""
    img_buf = _create_test_image("PNG")
    response = client.post(
        "/api/scan?force_low_confidence=true",
        files={"file": ("test.png", img_buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["low_confidence"] is True
    assert len(data["objects"]) == 0


def test_scan_end_to_end_with_mock_detector():
    """Verify successful detection and Gemini classification enrichment."""
    img_buf = _create_test_image("JPEG")

    mock_detection = [
        DetectionItem(
            id="obj-1",
            object_name="Plastic Bottle",
            confidence=0.95,
            box=[10.0, 10.0, 100.0, 200.0],
        )
    ]

    with patch("app.services.detector.ObjectDetector.detect", return_value=mock_detection):
        response = client.post(
            "/api/scan",
            files={"file": ("waste.jpg", img_buf, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["low_confidence"] is False
        assert len(data["objects"]) == 1

        obj = data["objects"][0]
        assert obj["object_name"] == "Plastic Bottle"
        assert obj["confidence"] == 0.95
        assert "classification" in obj
        assert obj["classification"]["wet_dry"] == "Dry"
        assert obj["classification"]["recyclable"] == "Recyclable"
        assert "recommended_bin" in obj["classification"]
