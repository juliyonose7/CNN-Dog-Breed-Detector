# Contributing

## Scope

This repository contains research and production-oriented assets for:
- Binary classification (`dog` vs `not-dog`)
- Multi-class dog breed classification
- API and frontend integration

## Branching Strategy

- `main`: stable branch, release-ready
- `feature/<name>`: new features
- `fix/<name>`: bug fixes
- `docs/<name>`: documentation-only updates

## Commit Convention

Use Conventional Commits:
- `feat:` new functionality
- `fix:` bug fixes
- `docs:` documentation only
- `refactor:` internal code changes without behavior changes
- `test:` tests and validation updates
- `chore:` maintenance work

Examples:
- `feat(api): add threshold override endpoint`
- `fix(training): prevent class imbalance in fold split`
- `docs(readme): rewrite setup section for Linux and Windows`

## Pull Request Requirements

1. Keep PRs scoped to one concern.
2. Update documentation when behavior changes.
3. Include verification evidence:
   - command output,
   - test summary,
   - or API request/response examples.
4. Do not include large generated artifacts in source control.

## Quality Checklist

Before opening a PR:
- Run local tests that are relevant to your change.
- Verify API startup for changed backend modules.
- Ensure no secrets or private datasets are committed.
- Confirm model artifacts are handled via release assets or Git LFS.
