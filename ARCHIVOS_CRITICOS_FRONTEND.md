# Frontend Critical Files Inventory

## Purpose

This document lists frontend files that are considered operationally critical for local inference testing.

## Critical Files

1. `simple_frontend_119.html`
   - Primary static user interface
   - Handles upload workflow and result containers

2. `styles.css`
   - UI styling and layout behavior

3. `app.js`
   - Frontend business logic
   - API request/response mapping

4. `start_frontend.py`
   - Local static server with CORS support

5. `INICIAR_SISTEMA.bat`
   - Windows startup helper for backend/frontend orchestration

6. `iniciar_sistema.sh`
   - Unix startup helper for backend/frontend orchestration

## Supporting Documentation

- `FRONTEND_DEFINITIVO_README.md`
- `FRONTEND_119_TESTING_GUIDE.md`
- `GUIA_USO_RAPIDA.md`

## Integrity Recommendations

- Keep filenames stable unless all references are updated.
- Validate startup scripts after any path changes.
- Track frontend behavior changes in `CHANGELOG.md`.

## Backup Recommendation

Before modifying critical frontend files:
1. Create a local backup copy.
2. Commit current state in a dedicated branch.
3. Validate both static and React frontend flows after change.
