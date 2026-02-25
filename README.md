# Dog Detection and Breed Classification

Production-oriented deep learning project for:
- binary image classification (`dog` vs `not-dog`), and
- dog breed classification (multi-class pipelines, including 119-class configurations).

The repository includes model training scripts, evaluation and bias-analysis utilities, FastAPI services, and frontend integration assets.

## Current Repository Status

This codebase contains multiple training/evaluation iterations and generated artifacts from experimentation.

For professional publication and maintainability, this repository now follows:
- technical documentation in English,
- a clear versioning policy (`VERSIONING.md`),
- contribution guidelines (`CONTRIBUTING.md`), and
- changelog tracking (`CHANGELOG.md`).

## Core Capabilities

- Transfer learning and fine-tuning pipelines (EfficientNet, ResNet, DenseNet variants)
- Binary and multi-class inference APIs
- Class-balancing and augmentation workflows
- Bias and false-negative analysis tooling
- Frontend integration for inference testing

## Repository Structure

```text
NOTDOG YESDOG/
├── api_server.py                     # FastAPI service (binary pipeline variant)
├── testing_api_119_classes.py        # FastAPI service for 119-class model
├── main_pipeline.py                  # End-to-end pipeline entrypoint
├── model_trainer.py                  # Core training pipeline
├── inference_optimizer.py            # Inference optimization scripts
├── breed_*.py                        # Breed-related training and preprocessing
├── analyze_*.py                      # Analysis and reporting scripts
├── dog-detector-frontend/            # React frontend
├── simple_frontend_119.html          # Static frontend variant
├── requirements.txt                  # Python dependencies
├── CHANGELOG.md                      # Versioned change log
├── CONTRIBUTING.md                   # Contribution workflow
└── VERSIONING.md                     # Semantic versioning policy
```

## Environment Requirements

Minimum recommended baseline:
- Python 3.10+
- 16 GB RAM
- Multi-core CPU
- Optional GPU acceleration (ROCm/CUDA depending on platform)

## Quick Start

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run a baseline API service

```bash
python api_server.py
```

Typical local endpoints:
- `http://localhost:8000`
- `http://localhost:8000/docs`

## Training and Evaluation Workflows

Examples:

```bash
python model_trainer.py
python train_binary.py
python breed_trainer.py
python analyze_bias_119_classes.py
python analyze_false_negatives_119.py
```

Because this repository contains multiple experimental variants, select one training path at a time and keep outputs outside source control.

## Frontend Testing

Options include:
- Static frontend (`simple_frontend_119.html` + local static server)
- React frontend under `dog-detector-frontend/`

See:
- `FRONTEND_DEFINITIVO_README.md`
- `FRONTEND_119_TESTING_GUIDE.md`
- `GUIA_USO_RAPIDA.md`

## Publication Guidance for GitHub

### Required before first public push

1. Initialize Git and create the default branch:

```bash
git init
git checkout -b main
```

2. Make the initial commit:

```bash
git add .
git commit -m "chore: initialize repository with technical documentation baseline"
```

3. Create first public tag:

```bash
git tag -a v0.1.0 -m "Initial public release"
```

4. Push to remote:

```bash
git remote add origin <your-github-repo-url>
git push -u origin main
git push origin v0.1.0
```

### Model artifacts and large files

Do not store large model binaries in standard Git history when possible. Use one of:
- GitHub Releases assets
- Git LFS

## Recommended Next Cleanup (Optional)

- Consolidate duplicate training scripts into clearly versioned modules.
- Move generated reports to a dedicated output directory ignored by Git.
- Add automated checks (lint, minimal tests, API smoke tests) in CI.

## License

Add a project license file (`LICENSE`) before public release if not already present.
