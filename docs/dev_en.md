Development Guidelines

Code Style

- All code comments MUST be in English
- Strict 80-column limit
- Follow the existing code style in the project
- No Chinese characters in code (comments or strings)

Git Commits

- All commits MUST include a Signed-off-by line using `git commit -s`
- All commit messages MUST be in English
- Use present tense: "Fix bug" not "Fixed bug"
- First line: summary (<= 50 characters)
- Body: details (optional)

Issues and PRs

- Issues MUST be in English
- PRs MUST be in English
- If you are a Chinese user and email me directly,
  Chinese is acceptable.

Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: python -m pytest tests/
5. Submit a PR

Code Organization

TinyMLC/        Core library
ANG/          Network generator
converter/    ONNX / TFLite parsers
transform/    Optimization passes
templates/    Jinja2 templates
ops/            Operators
c/            Pure-C operators
riscv/        RISC-V specific
arm/          ARM specific
TinyGUI/        Qt6 GUI
utils/          Utility functions

Adding a New Operator

1. Add pure-C implementation to ops/c/
2. Add operator name to SUPPORTED_OPS in ops.py
3. Add conversion logic to converter/
4. Add code generation logic to codegen.py

Adding a New Optimization Pass

1. Create new file in TinyMLC/transform/
2. Inherit from Pass base class
3. Implement run() method
4. Add to PassManager.default_pipeline()
5. Add tests

Testing

python -m pytest tests/
