#!/usr/bin/env python3
"""
从 tflite 文件中提取 FC 和 LSTM 层的权重和 bias
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path


def extract_fc_weights(interpreter):
    """提取 FULLY_CONNECTED 层的权重和 bias"""
    weights_data = None
    bias_data = None

    # FC 权重索引 3, bias 索引 2
    try:
        weights_data = interpreter.get_tensor(3)
        print(f"FC 权重: shape={weights_data.shape}, dtype={weights_data.dtype}")
        print(f"FC 权重范围: [{weights_data.min()}, {weights_data.max()}]")
    except:
        print("警告: 无法获取 FC 权重 (index=3)")

    try:
        bias_data = interpreter.get_tensor(2)
        print(f"FC bias: shape={bias_data.shape}, dtype={bias_data.dtype}")
        print(f"FC bias 范围: [{bias_data.min()}, {bias_data.max()}]")
    except:
        print("警告: 无法获取 FC bias (index=2)")

    return weights_data, bias_data


def extract_lstm_weights(interpreter):
    """提取 LSTM 各门的权重和 bias"""
    lstm_weights = {
        'input': {},
        'recurrent': {},
        'bias': {}
    }

    # LSTM 各门的张量索引（从 verbose 输出得知）
    # 输入权重: 15,14,13,12 (shape [20,28])
    # 递归权重: 11,10,9,8  (shape [20,20])
    # 偏置: 7,6,5,4       (shape [20])

    input_indices = {'i': 15, 'f': 14, 'g': 13, 'o': 12}
    recurrent_indices = {'i': 11, 'f': 10, 'g': 9, 'o': 8}
    bias_indices = {'i': 7, 'f': 6, 'g': 5, 'o': 4}

    for gate, idx in input_indices.items():
        try:
            data = interpreter.get_tensor(idx)
            lstm_weights['input'][gate] = data
            print(f"LSTM input_{gate}: shape={data.shape}, dtype={data.dtype}")
        except:
            print(f"警告: 无法获取 LSTM input_{gate} (index={idx})")
            lstm_weights['input'][gate] = None

    for gate, idx in recurrent_indices.items():
        try:
            data = interpreter.get_tensor(idx)
            lstm_weights['recurrent'][gate] = data
            print(f"LSTM recurrent_{gate}: shape={data.shape}, dtype={data.dtype}")
        except:
            print(f"警告: 无法获取 LSTM recurrent_{gate} (index={idx})")
            lstm_weights['recurrent'][gate] = None

    for gate, idx in bias_indices.items():
        try:
            data = interpreter.get_tensor(idx)
            lstm_weights['bias'][gate] = data
            print(f"LSTM bias_{gate}: shape={data.shape}, dtype={data.dtype}")
        except:
            print(f"警告: 无法获取 LSTM bias_{gate} (index={idx})")
            lstm_weights['bias'][gate] = None

    return lstm_weights


def export_weights_to_c(weights, name, output_file):
    """导出 int8 权重"""
    if weights is None:
        output_file.write(f"// {name} 未找到，使用占位符\n")
        output_file.write(f"static const int8_t {name}[1] = {{0}};\n\n")
        return

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
    if bias is None:
        output_file.write(f"// {name} 未找到，使用占位符\n")
        output_file.write(f"static const int32_t {name}[1] = {{0}};\n\n")
        return

    output_file.write(f"static const int32_t {name}[{bias.size}] = {{\n    ")
    for i, val in enumerate(bias):
        output_file.write(f"{int(val)}")
        if i < bias.size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def extract_quant_params(interpreter):
    """提取 LSTM 各门的量化参数"""
    tensor_details = interpreter.get_tensor_details()
    tensor_map = {t['index']: t for t in tensor_details}

    gate_indices = {'i': 15, 'f': 14, 'g': 13, 'o': 12}
    quant_params = {}

    for gate, idx in gate_indices.items():
        tensor = tensor_map.get(idx)
        if tensor and tensor['quantization']:
            scale = tensor['quantization'][0]
            zero_point = tensor['quantization'][1]
            quant_params[gate] = {'scale': scale, 'zero_point': zero_point}
            print(f"LSTM gate {gate}: scale={scale}, zero_point={zero_point}")

    return quant_params


def export_concatenated_weights(weights_list, output_file, array_name,
                                dtype='int8'):
    """导出拼接后的权重数组"""
    # 收集所有数组
    arrays = []
    total_size = 0
    for gate in ['i', 'f', 'g', 'o']:
        w = weights_list[gate]
        if w is None:
            print(f"警告: {gate} 门权重缺失，使用零数组")
            return
        flat = w.flatten()
        arrays.append(flat)
        total_size += len(flat)

    # 拼接
    concatenated = np.concatenate(arrays)

    if dtype == 'int8':
        c_type = 'int8_t'
    else:
        c_type = 'int32_t'

    output_file.write(
        f"static const {c_type} {array_name}[{total_size}] = {{\n    ")
    for i, val in enumerate(concatenated):
        output_file.write(f"{int(val)}")
        if i < total_size - 1:
            output_file.write(", ")
        if (i + 1) % 16 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def export_concatenated_bias(bias_list, output_file, array_name):
    """导出拼接后的 bias 数组"""
    arrays = []
    total_size = 0
    for gate in ['i', 'f', 'g', 'o']:
        b = bias_list[gate]
        if b is None:
            print(f"警告: {gate} 门 bias 缺失，使用零数组")
            return
        flat = b.flatten()
        arrays.append(flat)
        total_size += len(flat)

    concatenated = np.concatenate(arrays)

    output_file.write(
        f"static const int32_t {array_name}[{total_size}] = {{\n    ")
    for i, val in enumerate(concatenated):
        output_file.write(f"{int(val)}")
        if i < total_size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def main():
    parser = argparse.ArgumentParser(description='从 tflite 模型提取 FC 和 LSTM 权重')
    parser.add_argument('model', help='TFLite 模型文件路径')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='输出目录 (默认: tinymlc_generated)')
    args = parser.parse_args()

    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # 提取权重
    fc_weights, fc_bias = extract_fc_weights(interpreter)
    lstm_weights = extract_lstm_weights(interpreter)

    if fc_weights is None and all(v is None for v in lstm_weights['input'].values()):
        print("错误: 未找到任何权重")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 fc_weights.h
    if fc_weights is not None:
        output_path = output_dir / 'fc_weights.h'
        with open(output_path, 'w') as f:
            f.write("// 自动从 tflite 提取的 FC 层权重和 bias\n")
            f.write("// 请勿手动修改\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        print(f"已生成: {output_path}")

    # 生成 lstm_weights.h
    output_path = output_dir / 'lstm_weights.h'
    with open(output_path, 'w') as f:
        f.write("// 自动从 tflite 提取的 LSTM 各门权重和 bias（已拼接）\n")
        f.write("// 顺序: i, f, g, o\n")
        f.write("// 请勿手动修改\n\n")

        # 导出拼接后的输入权重 [4, 20, 28] -> [4*20*28]
        export_concatenated_weights(lstm_weights['input'], f,
                          'lstm_input_weights', 'int8')
        # 导出拼接后的递归权重 [4, 20, 20] -> [4*20*20]
        export_concatenated_weights(lstm_weights['recurrent'], f,
                          'lstm_recurrent_weights', 'int8')
        # 导出拼接后的 bias [4, 20] -> [4*20]
        export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')

    print(f"已生成: {output_path}")

    # 打印统计信息
    print("\n=== 提取统计 ===")
    if fc_weights is not None:
        print(f"FC 权重数量: {fc_weights.size}, bias 数量: {fc_bias.size}")

    for gate in ['i', 'f', 'g', 'o']:
        w = lstm_weights['input'].get(gate)
        if w is not None:
            print(f"LSTM input_{gate}: {w.size} 个权重")
        r = lstm_weights['recurrent'].get(gate)
        if r is not None:
            print(f"LSTM recurrent_{gate}: {r.size} 个权重")
        b = lstm_weights['bias'].get(gate)
        if b is not None:
            print(f"LSTM bias_{gate}: {b.size} 个偏置")


if __name__ == "__main__":
    main()
