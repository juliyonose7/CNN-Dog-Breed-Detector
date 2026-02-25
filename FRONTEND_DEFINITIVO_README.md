# Frontend Integration Guide

## Purpose

This document describes the frontend integration for the 119-class dog breed classifier API.

## Frontend Components

Primary files:
- `simple_frontend_119.html`: static UI for image upload and result rendering
- `styles.css`: frontend styles
- `app.js`: browser-side API communication and result formatting
- `start_frontend.py`: local static server with CORS headers

Alternative frontend:
- `dog-detector-frontend/`: React implementation for extended testing

## Runtime Architecture

- Frontend server: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Main inference endpoint: `POST /predict`

## Startup Procedure

### 1) Start backend API

```bash
python testing_api_119_classes.py
```

### 2) Start frontend server

```bash
python start_frontend.py
```

### 3) Access UI

- Static UI: `http://localhost:3000/simple_frontend_119.html`

## API Response Contract (Example)

```json
{
  "success": true,
  "is_dog": true,
  "processing_time": 0.845,
  "top_predictions": [
    {
      "breed": "Golden Retriever",
      "confidence": 0.8934,
      "class_name": "n02099601-golden_retriever",
      "index": 43
    }
  ],
  "recommendation": {
    "most_likely": "Golden Retriever",
    "confidence": 0.8934,
    "is_confident": true
  }
}
```

## Troubleshooting

### Connection refused
- Confirm backend is running on port `8000`.
- Confirm frontend server is running on port `3000`.

### CORS issues
- Ensure requests are routed through `start_frontend.py`.
- Verify backend CORS configuration for local origins.

### Invalid image errors
- Use supported formats (`.jpg`, `.jpeg`, `.png`, `.webp`).
- Keep file sizes within configured backend limits.

## Operational Recommendations

- Keep backend and frontend logs visible during testing.
- Validate endpoint health before inference tests (`GET /health`).
- Document API contract changes in `CHANGELOG.md`.
