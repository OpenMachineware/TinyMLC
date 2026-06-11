#!/usr/bin/env python3
"""
从 tflite 文件中提取 FC 层的权重和 bias
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path


def extract_fc_weights(model_path):
    """提取 FULLY_CONNECTED 层的权重和 bias"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    weights_data = None
    bias_data = None

    for tensor in tensor_details:
        # FULLY_CONNECTED 的权重和 bias 索引
        # 从之前的 verbose 输出可知：
        # - 权重张量索引是 3 (shape=[10, 560])
        # - bias 张量索引是 2 (shape=[10])
        if tensor['index'] == 3:
            weights_data = interpreter.get_tensor(3)
            print(f"权重: shape={weights_data.shape}, dtype={weights_data.dtype}")
            print(f"权重范围: [{weights_data.min()}, {weights_data.max()}]")
        elif tensor['index'] == 2:
            bias_data = interpreter.get_tensor(2)
            print(f"bias: shape={bias_data.shape}, dtype={bias_data.dtype}")
            print(f"bias 范围: [{bias_data.min()}, {bias_data.max()}]")

    return weights_data, bias_data


def export_weights_to_c(weights, name, output_file):
    """导出 int8 权重"""
    flat = weights.flatten()
    output_file.write(f"static const int8_t {name}[{flat.size}] = {{\n    ")
    for i, val in enumerate(flat):
        output_file.write(f"{int(val)}")
        if i < flat.size - 1:
            output_file.write(", ")
        if (i + 1) % 16 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def export_bias_to_c(bias, name, output_file):
    """导出 int32 bias"""
    output_file.write(f"static const int32_t {name}[{bias.size}] = {{\n    ")
    for i, val in enumerate(bias):
        output_file.write(f"{int(val)}")
        if i < bias.size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def main():
    parser = argparse.ArgumentParser(description='从 tflite 模型提取 FC 层权重')
    parser.add_argument('model', help='TFLite 模型文件路径')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='输出目录 (默认: tinymlc_generated)')
    args = parser.parse_args()

    weights, bias = extract_fc_weights(args.model)

    if weights is None or bias is None:
        print("错误: 未找到 FC 层的权重或 bias")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 fc_weights.h
    output_path = output_dir / 'fc_weights.h'
    with open(output_path, 'w') as f:
        f.write("// 自动从 tflite 提取的 FC 层权重和 bias\n")
        f.write("// 请勿手动修改\n\n")
        export_weights_to_c(weights, "fc_weights", f)
        export_bias_to_c(bias, "fc_bias", f)

    print(f"已生成: {output_path}")
    print(f"权重数量: {weights.size}, bias 数量: {bias.size}")


if __name__ == "__main__":
    main()
