# TinyMLC - 微型机器学习编译器

[![PyPI version](https://img.shields.io/pypi/v/tinymlc.svg)](https://pypi.org/project/tinymlc/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/OpenMachineware/TinyMLC)](https://cla-assistant.io/OpenMachineware/TinyMLC)

[![英文文档](https://img.shields.io/badge/English_Docs-Click_here-brightgreen?style=for-the-badge)](./README.md)

自动生成 + 优化 + 部署 MCU 神经网络。

## 特性

- 自动网络生成（ANG）
- ONNX / TFLite 模型转换
- 7 个优化 Pass（常量折叠、DCE、CSE、Simplify、
  融合、代数化简、内存复用）
- 31 个纯 C 算子（int8 量化）
- 多后端支持（Host / ARM / RISC-V）
- Qt6 图形界面
- 加速库支持（CMSIS-NN / NMSIS）

## 快速开始

```bash
pip install -e .
python main.py generate --task-type classification --max-macs 100000
cd TinyGUI/build && ./TinyGUI
```

## 文档


- 用户指南：[docs/guide_zh.md](./docs/guide_zh.md)
- 开发指南：[docs/dev_zh.md](./docs/dev_zh.md)
- 移植指南：[docs/porting_zh.md](./docs/porting_zh.md)
- 算子状态：[docs/ops_zh.md](./docs/ops_zh.md)

## 贡献

欢迎贡献。在您的首个拉取请求前，需要签署我们的
[贡献者许可协议](./CLA.md)（英文版 [`CLA.md`](./CLA.md) 具有法律效力，
中文参考版 [`CLA_zh.md`](./CLA_zh.md)）。详见
[CONTRIBUTING_zh.md](./CONTRIBUTING_zh.md)。

## 许可证

Apache License 2.0
