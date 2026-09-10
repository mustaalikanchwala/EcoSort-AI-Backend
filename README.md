# EcoSort AI — Backend

A lightweight, stateless Python + FastAPI backend for the **EcoSort AI** smart waste segregation system.

## Architectural Responsibility Separation

- **YOLO (Ultralytics)**: Answers *"What object is present?"* (extracts label, confidence, bounding box).
- **Google Gemini**: Answers *"How should this object be classified and disposed of?"* (Wet/Dry, Recyclable, Biodegradable, Recommended Bin, and human explanation).
- **Stateless Pipeline**: Zero database dependencies. Uploaded or captured photos are processed in memory and never stored.

## Project Structure

```
backend/
├── app/
│   ├── main.py               # FastAPI entry point, CORS, /health endpoint
│   ├── api/routes/scan.py    # POST /api/scan route
│   ├── core/config.py        # Pydantic Settings & environment variables
│   ├── schemas/
│   │   ├── scan.py           # Detection and ScanResponse schemas
│   │   └── waste.py          # Waste attribute enums & classification schemas
│   ├── services/
│   │   ├── detector.py       # YOLO object detection
│   │   ├── classifier.py     # Google Gemini classification with Pydantic validation
│   │   └── scan_service.py   # In-memory pipeline orchestrator
│   └── utils/file_utils.py   # In-memory image file validation (MIME, size, PIL)
├── tests/test_scan.py        # Pytest test suite
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and configure your Gemini API key:
```bash
cp .env.example .env
```

### 4. Run Server
```bash
uvicorn app.main:app --port 8000 --reload
```
API Documentation will be accessible at `http://localhost:8000/docs`.

### 5. Run Tests
```bash
pytest
```
