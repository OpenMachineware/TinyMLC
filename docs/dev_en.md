# Development Guidelines

## Code Style

- All code comments MUST be in English
- **Strict 80-column limit**
- Follow the existing code style in the project
- No Chinese characters in code (comments or strings)

## Git Commits

- All commits MUST include a Signed-off-by line using `git commit -s`
- All commit messages MUST be in English
- Use present tense: "Fix bug" not "Fixed bug"
- First line: summary (<= 50 characters)
- Body: details (optional)

## Issues and PRs

- Issues MUST be in English
- PRs MUST be in English
- If you are a Chinese and email me directly, Chinese please.

## Contributing

- Fork the repository 
- Create a feature branch 
- Make your changes 
- Run tests: python -m pytest tests/ 
- Submit a PR

## Code Organization


| Directory | Description |
| :--- | :--- |
| TinyMLC/ | Core library |
| ANG/ | Network generator |
| converter/ | ONNX / TFLite parsers |
| transform/ | Optimization passes |
| templates/ | Jinja2 templates |
| ops/ | Operators |
| c/ | Pure-C operators |
| riscv/ | RISC-V specific |
| arm/ | ARM specific |
| TinyGUI/ | Qt6 GUI |
| utils/ | Utility functions |

## Adding a New Operator

- Add pure-C implementation to ops/c/ 
- Add operator name to SUPPORTED_OPS in ops.py 
- Add conversion logic to converter/ 
- Add code generation logic to codegen.py

## Adding a New Optimization Pass

- Create new file in TinyMLC/transform/ 
- Inherit from Pass base class 
- Implement run() method 
- Add to PassManager.default_pipeline()
- Add tests

##Testing

`python -m pytest tests/`
