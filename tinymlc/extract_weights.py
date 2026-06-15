#!/usr/bin/env python3
"""
从 tflite 文件中提取 FC 和 LSTM 层的权重和 bias
"""

import sys
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.parser import parse_model


def extract_fc_weights(interpreter, op_info):
    """根据算子信息提取 FC 层的权重和 bias"""
    weights_idx = op_info.get("fc_weights_idx")
    bias_idx = op_info.get("fc_bias_idx")

    if weights_idx is None or bias_idx is None:
        print("错误: FC 权重/bias 索引未找到")
        return None, None

    weights = interpreter.get_tensor(weights_idx)
    bias = interpreter.get_tensor(bias_idx)
    return weights, bias

def extract_lstm_weights(interpreter, op_info):
    """提取 LSTM 各门的权重和 bias"""
    indices = op_info.get("lstm_weight_indices", {})
    input_indices = indices.get("input", [])
    recurrent_indices = indices.get("recurrent", [])
    bias_indices = indices.get("bias", [])

    gate_order = ['i', 'f', 'g', 'o']

    lstm_weights = {'input': {}, 'recurrent': {}, 'bias': {}}

    # 提取输入权重
    for idx, gate in zip(input_indices, gate_order[:len(input_indices)]):
        lstm_weights['input'][gate] = interpreter.get_tensor(idx)

    # 提取递归权重
    for idx, gate in zip(recurrent_indices,
                         gate_order[:len(recurrent_indices)]):
        lstm_weights['recurrent'][gate] = interpreter.get_tensor(idx)

    # 提取偏置（可能少于4个）
    for idx, gate in zip(bias_indices, gate_order[:len(bias_indices)]):
        lstm_weights['bias'][gate] = interpreter.get_tensor(idx)

    # 为缺失的门添加 None 占位
    for gate in gate_order:
        if gate not in lstm_weights['input']:
            lstm_weights['input'][gate] = None
        if gate not in lstm_weights['recurrent']:
            lstm_weights['recurrent'][gate] = None
        if gate not in lstm_weights['bias']:
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
    gate_order = ['i', 'f', 'g', 'o']
    arrays = []

    for gate in gate_order:
        b = bias_list.get(gate)
        if b is not None:
            arrays.append(b.flatten())
            print(f"  bias_{gate}: {b.size} 个元素")
        else:
            print(f"  警告: {gate} 门 bias 缺失")

    if not arrays:
        # 所有 bias 都缺失，生成占位符
        output_file.write(f"static const int32_t {array_name}[1] = {{0}};\n\n")
        return

    concatenated = np.concatenate(arrays)
    total_size = len(concatenated)

    output_file.write(
        f"static const int32_t {array_name}[{total_size}] = {{\n    ")
    for i, val in enumerate(concatenated):
        output_file.write(f"{int(val)}")
        if i < total_size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")

    print(f"  生成 {array_name}[{total_size}]")


def main():
    parser = argparse.ArgumentParser(
        description='从 tflite 模型提取 FC 和 LSTM 权重')
    parser.add_argument('model', help='TFLite 模型文件路径')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='输出目录 (默认: tinymlc_generated)')
    args = parser.parse_args()

    # 1. 加载模型并解析
    print(f"正在加载模型: {args.model}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    model_info = parse_model(interpreter)

    # 2. 查找算子信息
    fc_op_info = None
    lstm_op_info = None
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            fc_op_info = op
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_op_info = op

    # 3. 提取权重
    print("\n正在提取权重...")
    fc_weights = fc_bias = None
    if fc_op_info:
        fc_weights, fc_bias = extract_fc_weights(interpreter, fc_op_info)

    lstm_weights = None
    if lstm_op_info:
        lstm_weights = extract_lstm_weights(interpreter, lstm_op_info)

    # 4. 检查是否有任何权重被提取
    has_fc = fc_weights is not None
    has_lstm = lstm_weights is not None and any(
        v is not None for v in lstm_weights['input'].values())

    if not (has_fc or has_lstm):
        print("错误: 未找到任何权重")
        return 1

    # 5. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 生成 FC 权重文件
    if fc_weights is not None:
        output_path = output_dir / 'fc_weights.h'
        with open(output_path, 'w') as f:
            f.write("// 自动从 tflite 提取的 FC 层权重和 bias\n")
            f.write("// 请勿手动修改\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        print(f"已生成: {output_path}")

    # 7. 生成 LSTM 权重文件
    if lstm_weights and any(v is not None for v in lstm_weights['input'].values()):
        output_path = output_dir / 'lstm_weights.h'
        with open(output_path, 'w') as f:
            f.write("// 自动从 tflite 提取的 LSTM 各门权重和 bias（已拼接）\n")
            f.write("// 顺序: i, f, g, o\n")
            f.write("// 请勿手动修改\n\n")

            # 导出拼接后的输入权重
            export_concatenated_weights(lstm_weights['input'], f,
                                        'lstm_input_weights', 'int8')
            # 导出拼接后的递归权重
            export_concatenated_weights(lstm_weights['recurrent'], f,
                                        'lstm_recurrent_weights', 'int8')
            # 导出拼接后的 bias
            export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')

        print(f"已生成: {output_path}")

    # 8. 打印统计信息
    print("\n=== 提取统计 ===")
    if fc_weights is not None:
        print(
            f"FC 权重: {fc_weights.size} 个 int8, bias: {fc_bias.size} 个 int32")

    if lstm_weights:
        for gate in ['i', 'f', 'g', 'o']:
            w = lstm_weights['input'].get(gate)
            r = lstm_weights['recurrent'].get(gate)
            b = lstm_weights['bias'].get(gate)
            # 检查是否有任何非 None 的权重
            if w is not None or r is not None or b is not None:
                w_size = w.size if w is not None else 0
                r_size = r.size if r is not None else 0
                b_size = b.size if b is not None else 0
                print(
                    f"LSTM {gate} 门: input={w_size}, recurrent={r_size}, bias={b_size}")

    return 0


if __name__ == "__main__":
    main()
