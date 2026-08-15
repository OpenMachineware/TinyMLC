# TinyMLC - Tiny Machine Learning Compiler

[![PyPI version](https://img.shields.io/pypi/v/tinymlc.svg)](https://pypi.org/project/tinymlc/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/OpenMachineware/TinyMLC)](https://cla-assistant.io/OpenMachineware/TinyMLC)

[![Chinese Docs](https://img.shields.io/badge/Chinese_Docs-Click_here-blue?style=for-the-badge)](./README_zh.md)

Automatic generation + optimization + deployment of MCU neural
networks.

## Features

- Automatic network generation (ANG)
- ONNX / TFLite model conversion
- 7 optimization passes (constant folding, DCE, CSE, Simplify,
  fusion, algebraic simplify, memory reuse)
- 31 pure-C operators (int8 quantization)
- Multi-backend support (Host / ARM / RISC-V)
- Qt6 GUI
- Accelerator library support (CMSIS-NN / NMSIS)

## Quick Start

```bash
pip install -e .
python main.py generate --task-type classification --max-macs 100000
cd TinyGUI/build && ./TinyGUI
```

## Documentation

- User guide: [docs/guide_en.md](./docs/guide_en.md)
- Develop guide: [docs/dev_en.md](./docs/dev_en.md)
- Porting: [docs/porting_en.md](./docs/porting_en.md)
- Operator status: [docs/ops_en.md](./docs/ops_en.md)

## Contributing

Contributions are welcome. Before your first pull request, you will be asked
to sign our [Contributor License Agreement](./CLA.md) (English version:
[`CLA.md`](./CLA.md), legally binding; Chinese reference:
[`CLA_zh.md`](./CLA_zh.md)). See
[CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

Apache License 2.0
