# Security Policy

## Supported Versions

Security fixes are applied to the latest released version and the current `main` branch.

| Version | Supported |
| --- | --- |
| 1.x | Yes |
| < 1.0 | No |

## Reporting a Vulnerability

Do not open public GitHub issues for suspected vulnerabilities.

Report security concerns by emailing Diogo Ribeiro at `diogo_dj@hotmail.com` with:

- Affected version, branch, or commit.
- Steps to reproduce.
- Impact and likely exposure.
- Any relevant logs, configuration, or proof-of-concept details.

You should receive an acknowledgement within 7 days. Confirmed vulnerabilities will be triaged, fixed, and disclosed with release notes when appropriate.

## Scope

This project is a fine-tuning and serving template. Security reviews should pay particular attention to:

- Model artifact loading paths.
- Dataset ingestion and file parsing.
- FastAPI serving inputs.
- Container runtime configuration.
- Handling of private datasets, tokens, and model credentials.
