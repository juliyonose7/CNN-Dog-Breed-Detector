# Versioning and Release Policy

## Versioning Model

This project uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `MAJOR`: incompatible API or model-interface changes
- `MINOR`: backward-compatible features and quality improvements
- `PATCH`: backward-compatible bug fixes and documentation corrections

## Suggested Initial Version

Use `v0.1.0` for the first public repository publication.

Rationale:
- The codebase is functional and extensive.
- The architecture appears iterative and still evolving.
- `0.x` communicates active development and expected breaking changes.

## Release Process

1. Update `CHANGELOG.md` under `[Unreleased]`.
2. Create release section for target version.
3. Tag release:
   - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
   - `git push origin vX.Y.Z`
4. Publish GitHub Release notes.
5. Attach model artifacts as release assets (or use Git LFS).

## Model Artifact Policy

- Avoid storing large model binaries directly in Git history.
- Prefer one of:
  - GitHub Releases assets, or
  - Git LFS with explicit storage budget controls.
