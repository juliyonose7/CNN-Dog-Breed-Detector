# System Overview

## Scope

This document provides a high-level view of the end-to-end system in this repository.

## Pipeline Stages

1. Data ingestion and preprocessing
2. Training and fine-tuning
3. Evaluation and bias analysis
4. Inference optimization
5. API serving and frontend integration

## Main Runtime Modes

- Binary classification mode (`dog` vs `not-dog`)
- Multi-class breed classification mode

## Deployment Pattern (Local)

- FastAPI backend on port `8000`
- Frontend server on port `3000`

## Operational Notes

- Use isolated virtual environments.
- Keep generated outputs and large model files out of source control.
- Track all externally visible changes in `CHANGELOG.md`.
