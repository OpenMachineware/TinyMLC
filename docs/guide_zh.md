TinyMLC 用户指南

CLI 命令

generate
生成网络结构。

    python main.py generate \
        --task-type classification \
        --input-shape 1,28,28,1 \
        --output-shape 1,10 \
        --max-macs 100000 \
        --mode debug \
        --run

convert
转换 ONNX / TFLite 模型。

    python main.py convert \
        --model model.onnx \
        --target riscv \
        --run

GUI 使用

Generate: 生成网络
Clear:    清空控制台
Stop:     停止当前进程
Settings: 配置路径
Export Log: 导出日志到文件

配置文件

~/.tinymlc/config.json

项目结构

TinyMLC/      核心库
TinyGUI/      Qt6 图形界面
ops/          算子实现
utils/        工具函数
