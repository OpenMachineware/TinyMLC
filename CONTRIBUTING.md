# Contributing to TinyMLC

First off, thank you for considering contributing to TinyMLC. This project
is released under the [Apache License 2.0](./LICENSE) and is intended to
remain open source. Contributions of all kinds are welcome — code, bug
reports, documentation, and feature suggestions.

## Contributor License Agreement (CLA)

Before your pull request can be merged, you MUST sign our
[Contributor License Agreement](./CLA.md). The English version
([`CLA.md`](./CLA.md)) is the legally binding document; the Chinese version
([`CLA_zh.md`](./CLA_zh.md)) is provided for reference only.

- We use [CLA assistant](https://cla-assistant.io/) to manage signatures.
- On your **first** pull request, the CLA assistant bot will post a
  **"Sign the CLA"** button in the PR conversation. Click it and follow the
  instructions.
- Signing is required only once — it applies to all your future
  contributions.
- The CLA does **not** change the open-source nature of the project: all
  contributions are licensed to the project under the Apache License 2.0.

## Ways to Contribute

### Reporting Bugs

If you find a bug, please open an issue using the
[Bug Report](./.github/ISSUE_TEMPLATE/bug_report.yml) template. A good bug
report includes:

- A clear, descriptive title.
- The exact steps to reproduce the problem.
- Expected behavior vs. actual behavior.
- Environment information (OS, Python version, target MCU/backend, etc.).
- Logs or minimal code that triggers the issue, when possible.

### Requesting Features

Open a [Feature Request](./.github/ISSUE_TEMPLATE/feature_request.yml)
issue. Describe the problem you are trying to solve and how you expect the
feature to behave, rather than only proposing an implementation.

### Submitting Code

1. **Fork** the repository and clone your fork locally.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
3. **Make your changes** following the project's
   [code style](./docs/dev_en.md#code-style).
4. **Add or update tests** and make sure everything passes:
   ```bash
   python -m pytest tests/
   ```
5. **Commit** your changes with a Signed-off-by line:
   ```bash
   git commit -s
   ```
6. **Push** your branch and open a pull request against `main` using the
   [pull request template](./.github/pull_request_template.md).
7. **Sign the CLA** when prompted by the CLA assistant bot (first PR only).
8. Address review feedback; once approved, your PR will be merged.

## Code Style

- All code comments MUST be in English.
- Strict 80-column limit.
- Follow the existing code style in the project.
- No Chinese characters in code (comments or strings).

## Git Commit Guidelines

- Every commit MUST include a Signed-off-by line (`git commit -s`).
- All commit messages MUST be in English.
- Use present tense: "Fix bug", not "Fixed bug".
- First line: a summary of no more than 50 characters.
- Body: optional details.

## Development Guidelines

See [Development Guidelines](./docs/dev_en.md) for details on code
organization, adding new operators, adding new optimization passes, and
running tests.

## Issues and PRs

- Issues MUST be written in English.
- Pull requests MUST be written in English.
- If you email the maintainer directly and you are a Chinese speaker,
  Chinese is preferred for direct correspondence.

## Getting Help

- Open an issue for questions or discussions.
- Check the [documentation](./docs/guide_en.md) first.
