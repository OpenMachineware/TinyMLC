移植指南

支持的平台

平台       状态   说明
Host       ✅ 完整  开发调试用
ARM        ✅ 完整  纯 C 版本
RISC-V     ✅ 完整  纯 C 版本

添加新目标

1. 在 ops/ 下创建目标目录
2. 添加链接脚本 link_xxx.ld
3. 添加启动文件 start.S
4. 添加构建脚本 build_xxx_*.sh

添加新后端

1. 实现算子库（参考 ops/c/）
2. 修改 codegen.py 添加后端选择
3. 修改 copy_files_to_build() 拷贝文件

加速库适配

1. 在 ops/xxx/ 下创建加速库目录
2. 实现加速库的算子 wrapper
3. 修改 build_xxx_*.sh 链接加速库

开发规范

- 所有 commit 必须使用英文
- 所有代码注释必须使用英文
- 严格 80 列换行
- 遵循现有代码风格
- Issue 和 PR：仅限英文
- 中国用户邮件：可以使用中文

测试

python -m pytest tests/