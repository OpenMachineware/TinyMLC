TinyMLC User Guide

CLI Commands

generate
Generate a network structure.

    python main.py generate \
        --task-type classification \
        --input-shape 1,28,28,1 \
        --output-shape 1,10 \
        --max-macs 100000 \
        --mode debug \
        --run

convert
Convert ONNX / TFLite models.

    python main.py convert \
        --model model.onnx \
        --target riscv \
        --run

GUI Usage

Generate: Generate a network
Clear:    Clear console output
Stop:     Stop the current process
Settings: Configure paths
Export Log: Export console log to file

Configuration File

~/.tinymlc/config.json

Project Structure

TinyMLC/      Core library
TinyGUI/      Qt6 GUI
ops/          Operator implementations
utils/        Utility functions
