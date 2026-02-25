# Quick Start Guide

## Minimal Startup

### 1) Start backend API

```bash
python testing_api_119_classes.py
```

Wait until the server reports it is listening on port `8000`.

### 2) Start frontend server

```bash
python start_frontend.py
```

Open:
- `http://localhost:3000/simple_frontend_119.html`

## Inference Flow

1. Upload a valid image file.
2. Wait for API response.
3. Review main prediction and top-5 list.

## Basic Troubleshooting

### Cannot connect to API
- Confirm backend process is active on port `8000`.

### Invalid file error
- Use a valid image format and file size within limits.

### Blank frontend page
- Confirm frontend server process is active on port `3000`.

## Quick Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:3000/simple_frontend_119.html
```
