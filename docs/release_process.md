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
3. Confirm `.zenodo.json` and `CITATION.cff` match the release version, author metadata, license, and repository URL.
4. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. GitHub Actions `release.yml` verifies quality gates, builds distribution artifacts, and creates a GitHub Release.
6. If the repository is enabled in Zenodo, Zenodo archives the GitHub Release and mints a DOI for the release.

## Zenodo DOI setup

This repository includes `.zenodo.json` for Zenodo-specific archive metadata and `CITATION.cff` for GitHub citation metadata.

One-time setup:

1. Sign in to Zenodo with the GitHub account that owns or can administer `DiogoRibeiro7/huggingface-finetuning-lab`.
2. Enable the repository in Zenodo's GitHub integration.
3. Create a GitHub Release from a `vX.Y.Z` tag.
4. After Zenodo archives the release, copy the minted DOI into any release notes, badges, or downstream citation material that should reference the archived version.

Zenodo uses `.zenodo.json` in preference to `CITATION.cff` when both files exist, so keep `.zenodo.json` complete for each release.

## Local pre-release checks

```bash
make check
poetry build
```
