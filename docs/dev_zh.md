开发指南

代码风格

- 所有代码注释必须使用英文
- 严格 80 列换行
- 遵循项目现有代码风格
- 代码中不能出现中文字符（注释或字符串）

Git Commit

- 所有 commit 必须使用 git commit -s 添加 Signed-off-by
- 所有 commit 消息必须使用英文
- 使用现在时："Fix bug" 而不是 "Fixed bug"
- 第一行：摘要（不超过 50 字符）
- 正文：详细说明（可选）

Issue 和 PR

- Issue 必须使用英文
- PR 必须使用英文
- 如果你是中国用户直接给我发邮件，可以使用中文

贡献流程

1. Fork 仓库
2. 创建功能分支
3. 修改代码
4. 运行测试：python -m pytest tests/
5. 提交 PR

代码结构

TinyMLC/        核心库
ANG/          网络生成器
converter/    ONNX / TFLite 解析器
transform/    优化 Pass
templates/    Jinja2 模板
ops/            算子
c/            纯 C 算子
riscv/        RISC-V 特定
arm/          ARM 特定
TinyGUI/        Qt6 图形界面
utils/          工具函数

添加新算子

1. 在 ops/c/ 添加纯 C 实现
2. 在 ops.py 的 SUPPORTED_OPS 中添加算子名
3. 在 converter/ 添加转换逻辑
4. 在 codegen.py 添加代码生成逻辑

添加新的优化 Pass

1. 在 TinyMLC/transform/ 创建新文件
2. 继承 Pass 基类
3. 实现 run() 方法
4. 添加到 PassManager.default_pipeline()
5. 添加测试

测试

python -m pytest tests/
