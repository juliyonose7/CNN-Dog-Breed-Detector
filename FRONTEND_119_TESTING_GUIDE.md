# Frontend Testing Guide (119-Class Model)

## Objective

Validate end-to-end integration between the frontend client and the 119-class ResNet-based backend API.

## Components Under Test

- Backend: `testing_api_119_classes.py`
- Frontend (static): `simple_frontend_119.html`, `app.js`, `styles.css`
- Frontend (React): `dog-detector-frontend/`

## Execution Options

### Automated script (Windows)

```powershell
.\start_frontend_119_classes.bat
```

### Manual execution

Backend:

```bash
python testing_api_119_classes.py
```

Frontend (React):

```bash
cd dog-detector-frontend
npm install
npm start
```

Frontend (static):

```bash
python start_frontend.py
```

## Test Endpoints

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Validation Checklist

1. Health endpoint is reachable (`GET /health`).
2. Image upload succeeds from frontend UI.
3. API returns structured predictions.
4. Top-5 predictions are correctly rendered.
5. Error states are handled (invalid file, timeout, backend down).

## Expected Output Quality

- Main prediction and confidence are shown.
- Top-5 list is sorted by confidence descending.
- Processing time is visible.
- No uncaught exceptions in browser console.

## Common Failure Modes

### Backend unavailable
- Symptom: network error in frontend.
- Action: start backend and re-run health check.

### Frontend dependency issues
- Symptom: `npm start` fails.
- Action: remove `node_modules`, reinstall packages, retry.

### Model load failure
- Symptom: backend starts but prediction endpoint fails.
- Action: verify model path and checkpoint compatibility.
