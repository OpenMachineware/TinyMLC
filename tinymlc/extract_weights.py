#!/usr/bin/env python3
"""
从 tflite 文件中提取 FC 和 LSTM 层的权重和 bias
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.parser_litert import parse_model_tflite
from tinymlc.utils import fatal_error, warning, info


GATE_ORDER = ['i', 'f', 'g', 'o']
WEIGHTLESS_OPS = ["ADD", "SOFTMAX", "RESHAPE", "QUANTIZE"]
EXPORT_LINE_WIDTH = 16  # int8 数组每行显示 16 个
EXPORT_BIAS_LINE_WIDTH = 8  # int32 数组每行显示 8 个


def extract_fc_weights(interpreter, op_info):
    """提取 FULLY_CONNECTED 层的权重和 bias"""
    weights_idx = op_info.get("fc_weights_idx")
    bias_idx = op_info.get("fc_bias_idx")

    if weights_idx is None or bias_idx is None:
        fatal_error(
            f"FC 权重/bias 索引未找到: weights={weights_idx}, bias={bias_idx}",
            "检查模型转换时是否保留了张量名称"
        )

    try:
        # 尝试使用 tensor() 方法
        weights_tensor = interpreter.tensor(weights_idx)
        bias_tensor = interpreter.tensor(bias_idx)
        weights = weights_tensor()
        bias = bias_tensor()
    except ValueError as e:
        # 回退到 get_tensor
        try:
            weights = interpreter.get_tensor(weights_idx)
            bias = interpreter.get_tensor(bias_idx)
        except ValueError as e2:
            fatal_error(f"无法获取张量: {e2}", "请确保模型已正确加载")

    info(f"FC 权重: shape={weights.shape}, dtype={weights.dtype}")
    info(f"FC bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def extract_lstm_weights(interpreter, op_info):
    """提取 LSTM 各门的权重和 bias"""
    indices = op_info.get("lstm_weight_indices", {})
    input_indices = indices.get("input", [])
    recurrent_indices = indices.get("recurrent", [])
    bias_indices = indices.get("bias", [])

    if not input_indices or not recurrent_indices:
        fatal_error(
            "LSTM 权重索引不完整",
            "检查模型是否为标准 TFLite LSTM 格式"
        )

    gate_order = ['i', 'f', 'g', 'o']
    lstm_weights = {'input': {}, 'recurrent': {}, 'bias': {}}

    # 提取输入权重
    for gate, idx in zip(gate_order, input_indices):
        try:
            lstm_weights['input'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            fatal_error(f"LSTM input_{gate} 权重提取失败: {e}")

    # 提取递归权重
    for gate, idx in zip(gate_order, recurrent_indices):
        try:
            lstm_weights['recurrent'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            fatal_error(f"LSTM recurrent_{gate} 权重提取失败: {e}")

    # 提取偏置（可能少于4个）
    for gate, idx in zip(gate_order, bias_indices):
        try:
            lstm_weights['bias'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            warning(f"LSTM bias_{gate} 提取失败，将使用零数组", f"索引: {idx}")
            lstm_weights['bias'][gate] = None

    # 为缺失的门添加 None 占位
    for gate in gate_order:
        if gate not in lstm_weights['input']:
            lstm_weights['input'][gate] = None
        if gate not in lstm_weights['recurrent']:
            lstm_weights['recurrent'][gate] = None
        if gate not in lstm_weights['bias']:
            lstm_weights['bias'][gate] = None

    return lstm_weights


def export_concatenated_weights(weights_list, output_file, array_name,
                                dtype='int8'):
    """导出拼接后的权重数组（缺失的门跳过，用零数组补齐）"""
    arrays = []
    total_size = 0
    missing_gates = []

    for gate in GATE_ORDER:
        w = weights_list.get(gate)
        if w is None:
            missing_gates.append(gate)
            # 用零数组作为占位（形状从其他门推断）
            # 这里先收集缺失信息，后面统一处理
            continue
        arrays.append(w.flatten())
        total_size += w.size

    # 如果有缺失，打印警告并根据已有形状创建零数组
    if missing_gates:
        warning(f"警告: {missing_gates} 门权重缺失，使用零数组补齐")
        # 从第一个非 None 的权重获取形状
        for gate in GATE_ORDER:
            w = weights_list.get(gate)
            if w is not None:
                shape = w.shape
                for mg in missing_gates:
                    zero_array = np.zeros(shape, dtype=np.int8)
                    arrays.append(zero_array.flatten())
                    total_size += zero_array.size
                break

    if not arrays:
        fatal_error(
            f"所有 {array_name} 权重缺失",
            f"检查模型转换是否完整，或确认模型包含 {array_name} 权重"
        )

    # 拼接
    concatenated = np.concatenate(arrays)
    total_size = len(concatenated)

    c_type = 'int8_t' if dtype == 'int8' else 'int32_t'
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
        else:
            warning(f"{gate} 门 bias 缺失")

    if not arrays:
        # 所有 bias 都缺失，生成占位符
        warning(f"{array_name} 全部缺失，使用零数组占位")
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

    info(f"  生成 {array_name}[{total_size}]")


def extract_conv_weights(interpreter, op_info):
    """提取 CONV_2D 层的权重和 bias"""
    weights_idx = op_info.get("conv_weights_idx")
    bias_idx = op_info.get("conv_bias_idx")

    if weights_idx is None:
        fatal_error(
            "CONV_2D 权重索引未找到",
            "检查模型转换时是否保留了张量名称"
        )

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = interpreter.get_tensor(bias_idx) if bias_idx is not None else None
    except ValueError as e:
        fatal_error(f"无法获取 CONV_2D 张量: {e}", "请确保模型已正确加载")

    info(f"CONV_2D 权重: shape={weights.shape}, dtype={weights.dtype}")
    if bias is not None:
        info(f"CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


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


def extract_dw_weights(interpreter, op_info):
    """提取 DEPTHWISE_CONV_2D 层的权重和 bias"""
    weights_idx = op_info.get("dw_weights_idx")
    bias_idx = op_info.get("dw_bias_idx")

    if weights_idx is None:
        fatal_error(
            "DEPTHWISE_CONV_2D 权重索引未找到",
            "检查模型转换时是否保留了张量名称"
        )

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = interpreter.get_tensor(bias_idx) if bias_idx is not None else None
    except ValueError as e:
        fatal_error(f"无法获取 DEPTHWISE_CONV_2D 张量: {e}", "请确保模型已正确加载")

    info(f"DEPTHWISE_CONV_2D 权重: shape={weights.shape}, dtype={weights.dtype}")
    if bias is not None:
        info(f"DEPTHWISE_CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def main():
    parser = argparse.ArgumentParser(
        description='从 tflite 模型提取 FC 和 LSTM 权重')
    parser.add_argument('model', help='TFLite 模型文件路径')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='输出目录 (默认: tinymlc_generated)')
    args = parser.parse_args()

    # 1. 加载模型并解析
    info(f"正在加载模型: {args.model}")
    interpreter = LiteRTInterpreter(model_path=args.model)
    interpreter.allocate_tensors()

    model_info = parse_model_tflite(args.model)

    # 2. 查找算子信息
    fc_op_info = None
    lstm_op_info = None
    conv_op_info = None
    dw_op_info = None
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            fc_op_info = op
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_op_info = op
        elif op["op_name"] == "CONV_2D":
            conv_op_info = op
        elif op["op_name"] == "DEPTHWISE_CONV_2D":
            dw_op_info = op

    # 3. 提取权重
    info("\n正在提取权重...")
    fc_weights = fc_bias = None
    if fc_op_info:
        fc_weights, fc_bias = extract_fc_weights(interpreter, fc_op_info)
    lstm_weights = None
    if lstm_op_info:
        lstm_weights = extract_lstm_weights(interpreter, lstm_op_info)
    conv_weights = None
    conv_bias = None
    if conv_op_info:
        conv_weights, conv_bias = extract_conv_weights(interpreter,
                                                       conv_op_info)
    dw_weights = None
    dw_bias = None
    if dw_op_info:
        dw_weights, dw_bias = extract_dw_weights(interpreter, dw_op_info)

    # 4. 检查是否有任何权重被提取
    has_fc = fc_weights is not None
    has_lstm = lstm_weights is not None and any(v is not None for v in lstm_weights['input'].values())
    has_conv = conv_weights is not None
    has_dw = dw_weights is not None

    if not (has_fc or has_lstm or has_conv or has_dw):
        # 检查无权重算子
        has_weightless_op = False
        for op in model_info["ops"]:
            if op["op_name"] in WEIGHTLESS_OPS:
                has_weightless_op = True
                break

        if not has_weightless_op:
            fatal_error("未找到任何权重", "检查模型是否包含支持的算子")
        else:
            info("注意: 模型只包含无权重算子（ADD/Softmax/Reshape），继续...")

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
        info(f"已生成: {output_path}")

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

        info(f"已生成: {output_path}")

    # 8. 生成 CONV 权重文件 conv_weights.h
    if conv_weights is not None:
        output_path = output_dir / 'conv_weights.h'
        with open(output_path, 'w') as f:
            f.write("// 自动从 tflite 提取的 CONV_2D 权重和 bias\n")
            f.write("// 请勿手动修改\n\n")
            export_weights_to_c(conv_weights, "conv_weights", f)
            if conv_bias is not None:
                export_bias_to_c(conv_bias, "conv_bias", f)
        info(f"已生成: {output_path}")

    # 9. 生成 CONV 权重文件
    if dw_weights is not None:
        output_path = output_dir / 'dw_weights.h'
        with open(output_path, 'w') as f:
            f.write("// 自动从 tflite 提取的 DEPTHWISE_CONV_2D 权重和 bias\n")
            f.write("// 请勿手动修改\n\n")
            export_weights_to_c(dw_weights, "dw_weights", f)
            if dw_bias is not None:
                export_bias_to_c(dw_bias, "dw_bias", f)
        info(f"已生成: {output_path}")

    # 打印统计信息
    info("\n=== 提取统计 ===")
    if fc_weights is not None:
        info(
            f"FC 权重: {fc_weights.size} 个 int8, bias: {fc_bias.size} 个 int32")

    if lstm_weights:
        for gate in GATE_ORDER:
            w = lstm_weights['input'].get(gate)
            r = lstm_weights['recurrent'].get(gate)
            b = lstm_weights['bias'].get(gate)
            # 检查是否有任何非 None 的权重
            if w is not None or r is not None or b is not None:
                w_size = w.size if w is not None else 0
                r_size = r.size if r is not None else 0
                b_size = b.size if b is not None else 0
                info(
                    f"LSTM {gate} 门: input={w_size}, recurrent={r_size}, bias={b_size}")

    if conv_weights is not None:
        info(
            f"CONV 权重: {conv_weights.size} 个 int8, bias: {conv_weights.size} 个 int32")

    return 0


if __name__ == "__main__":
    main()
