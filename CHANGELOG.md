# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- AWS S3 model hosting for remote deployment:
  - Public S3 bucket (`notdog-yesdog-heavy-814298259360`) in `us-east-1` for model weight storage.
  - Automatic model download with retry logic and timeout handling in `testing_api_119_classes.py`.
  - Environment variable overrides for model URLs, retries, and timeouts.
- Repository governance and publication baseline:
  - `.gitignore` for Python, frontend, models, and generated artifacts.
  - `CONTRIBUTING.md` contribution and pull request workflow.
  - `VERSIONING.md` versioning and release policy.

### Changed
- `README.md` fully rewritten with comprehensive project documentation:
  - Architecture overview with hierarchical classification diagram.
  - Complete tech stack, repository structure, and API endpoint reference.
  - AWS S3 model hosting documentation with environment variable configuration.
  - Training pipeline, evaluation, and cross-validation documentation.
  - Frontend interface setup instructions for both React and static variants.
- All code comments translated from Spanish to English across 15+ files.
- Documentation style normalized to professional technical format without visual decorations.
