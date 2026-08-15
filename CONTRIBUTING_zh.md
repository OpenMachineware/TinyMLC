# 为 TinyMLC 贡献代码

首先，感谢您考虑为 TinyMLC 作出贡献。本项目依据
[Apache License 2.0](./LICENSE) 发布，并有意保持开源。欢迎各类贡献——
代码、Bug 报告、文档和功能建议。

## 贡献者许可协议（CLA）

在您的拉取请求（Pull Request）被合并之前，您**必须**签署我们的
[贡献者许可协议](./CLA.md)。英文版（[`CLA.md`](./CLA.md)）是具有法律
效力的文件；中文版（[`CLA_zh.md`](./CLA_zh.md)）仅供理解参考。

- 我们使用 [CLA assistant](https://cla-assistant.io/) 管理签署。
- 在您的**首次**拉取请求中，CLA assistant 机器人会在 PR 对话中发布一个
  **"Sign the CLA"**（签署 CLA）按钮，点击并按提示操作即可。
- 只需签署**一次**，之后所有贡献均适用。
- CLA 不会改变项目的开源性质：所有贡献均依据 Apache License 2.0 许可给项目。

## 贡献方式

### 报告 Bug

如果您发现 Bug，请使用 [Bug 报告](./.github/ISSUE_TEMPLATE/bug_report.yml)
模板提交 Issue。一份好的 Bug 报告应包含：

- 清晰、描述性的标题。
- 可复现问题的确切步骤。
- 预期行为与实际行为。
- 环境信息（操作系统、Python 版本、目标 MCU/后端等）。
- 尽可能附上触发问题的日志或最小代码。

### 请求新功能

请通过 [功能请求](./.github/ISSUE_TEMPLATE/feature_request.yml)
提交 Issue。请描述您想解决的问题以及期望的功能行为，而非仅提出一种实现方案。

### 提交代码

1. **Fork** 本仓库并在本地克隆您的 fork。
2. 从 `main` **创建功能分支**：
   ```bash
   git checkout -b feat/my-new-feature
   ```
3. **进行修改**，遵循项目的[代码风格](./docs/dev_zh.md#代码风格)。
4. **添加或更新测试**，并确保全部通过：
   ```bash
   python -m pytest tests/
   ```
5. **提交**您的更改，并附带 Signed-off-by 行：
   ```bash
   git commit -s
   ```
6. **推送**分支，并使用[拉取请求模板](./.github/pull_request_template.md)
   针对 `main` 打开 PR。
7. 当 CLA assistant 机器人提示时**签署 CLA**（仅首次 PR）。
8. 处理评审意见；批准后您的 PR 将被合并。

## 代码风格

- 所有代码注释必须使用英文。
- 严格遵循 80 列宽度限制。
- 遵循项目中已有的代码风格。
- 代码中不得出现中文字符（注释或字符串）。

## Git 提交规范

- 每次提交必须包含 Signed-off-by 行（`git commit -s`）。
- 所有提交信息必须使用英文。
- 使用现在时："Fix bug"，而非 "Fixed bug"。
- 首行：不超过 50 个字符的摘要。
- 正文：可选的详细说明。

## 开发指南

关于代码组织、新增算子、新增优化 Pass 以及运行测试的详细信息，
请参阅[开发指南](./docs/dev_zh.md)。

## Issue 与 PR 语言

- Issue 必须使用英文撰写。
- 拉取请求必须使用英文撰写。
- 如果您是中文使用者且直接通过邮件联系维护者，直接通信建议使用中文。

## 获取帮助

- 如有疑问或讨论，请提交 Issue。
- 请先查阅[文档](./docs/guide_zh.md)。
