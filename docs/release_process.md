# Release Process

## Versioning policy

This repository uses Semantic Versioning:

- `MAJOR`: breaking CLI/API changes.
- `MINOR`: backward-compatible features.
- `PATCH`: backward-compatible fixes and tooling/docs improvements.

## Changelog policy

- Keep `CHANGELOG.md` updated under `## [Unreleased]` in every PR.
- Move entries from `Unreleased` to a tagged version section at release time.
- Keep entries user-facing and grouped by Added/Changed/Fixed/Removed.

## Release steps

1. Ensure `main` is green in CI.
2. Confirm `CHANGELOG.md` and version in `pyproject.toml` are updated.
3. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. GitHub Actions `release.yml` verifies quality gates, builds distribution artifacts, and creates a GitHub Release.

## Local pre-release checks

```bash
make check
poetry build
```
