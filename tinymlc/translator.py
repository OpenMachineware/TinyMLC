#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
将 TFLite 模型转换为 MCU 可执行的 C 代码
"""

import stat
import subprocess
import sys
import argparse
import math
import numpy as np
import shutil

from pathlib import Path
from jinja2 import Template
from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

from tinymlc.extract_weights import (extract_fc_weights, extract_lstm_weights,
                                     export_weights_to_c, export_bias_to_c,
                                     export_concatenated_weights,
                                     export_concatenated_bias,
                                     extract_conv_weights)
from tinymlc.generate_lut import generate_lut
from tinymlc.parser_litert import parse_model_tflite
from tinymlc.parser_onnx import parse_model_onnx
from tinymlc.utils import fatal_error, warning, info


SUPPORTED_OPS = ["FULLY_CONNECTED", "UNIDIRECTIONAL_SEQUENCE_LSTM", "ADD",
                 "SOFTMAX", "RESHAPE", "QUANTIZE", "SVDF", "CONV_2D"]
# 回退值，仅在无法从模型读取有效 scale 时使用
DEFAULT_SCALE = 0.01  # 经验值
DEFAULT_SHIFT = 8     # 经验值


def build_execution_order(ops, tensors):
    """根据张量依赖关系确定算子执行顺序"""

    # 转换所有索引为 Python int
    for op in ops:
        op["index"] = int(op["index"])
        if "input_indices" in op:
            op["input_indices"] = [int(i) for i in op["input_indices"]]
        if "output_indices" in op:
            op["output_indices"] = [int(i) for i in op["output_indices"]]

    # 1. 建立张量 → 生产算子的映射
    tensor_producer = {}
    for op in ops:
        for out_idx in op.get("output_indices", []):
            tensor_producer[int(out_idx)] = op

    # 2. 建立算子依赖关系
    op_deps = {}
    for op in ops:
        deps = set()
        op_idx = int(op["index"])
        for inp_idx in op.get("input_indices", []):
            inp_idx = int(inp_idx)
            if inp_idx in tensor_producer:
                producer = tensor_producer[inp_idx]
                prod_idx = int(producer["index"])
                if prod_idx != op_idx:
                    deps.add(prod_idx)
        op_deps[op_idx] = list(deps)

    # 3. 计算入度（当前算子依赖于多少个其他算子）
    in_degree = {}
    for op in ops:
        op_idx = int(op["index"])
        in_degree[op_idx] = len(op_deps.get(op_idx, []))

    # 4. 拓扑排序（Kahn 算法）
    from collections import deque
    queue = deque([op_idx for op_idx, deg in in_degree.items() if deg == 0])

    order = []
    while queue:
        op_idx = queue.popleft()
        op = next(o for o in ops if int(o["index"]) == op_idx)
        order.append(op)

        for next_op in ops:
            next_idx = int(next_op["index"])
            if op_idx in op_deps.get(next_idx, []):
                in_degree[next_idx] -= 1
                if in_degree[next_idx] == 0:
                    queue.append(next_idx)

    if len(order) != len(ops):
        fatal_error("模型存在循环依赖，无法确定执行顺序",
                    "请检查模型结构是否合理")

    return order


def generate_c_code(model_info, output_dir, target,
                    inference_func="tinymlc_inference",
                    with_test_main=False):
    ops = model_info.get("ops", [])
    tensors = model_info.get("tensors", {})

    execution_order = build_execution_order(ops, tensors)
    info("算子执行顺序:")
    for op in execution_order:
        info(f"  {op['index']}: {op['op_name']}")

    # 生成前检查所有算子
    for op in ops:
        if op["state"] != "translated" and op["state"] != "generated":
            fatal_error(
                f"算子 {op['op_name']} 状态为 {op.get('state')}，无法生成代码",
                f"Pass flags: {op.get('pass_flags', {})}")

    # 检查是否有支持的算子
    has_supported_op = False
    for op in model_info["ops"]:
        if op["op_name"] in SUPPORTED_OPS:
            has_supported_op = True
            break

    if not has_supported_op:
        fatal_error("模型不包含任何支持的算子",
                    f"支持的算子: {', '.join(SUPPORTED_OPS)}")

    """生成 C 代码和头文件"""
    template_dir = Path(__file__).parent / 'templates'

    # 计算输入输出大小
    input_size = 1
    for inp in model_info['input']:
        size = 1
        for dim in inp['shape']:
            size *= dim
        input_size *= size

    output_size = 1
    for out in model_info['output']:
        size = 1
        for dim in out['shape']:
            size *= dim
        output_size *= size

    # 检测模型包含的算子类型
    has_fc = False
    has_lstm = False
    has_conv = False
    has_dw = False

    lstm_params = None
    fc_scales = []
    for op in model_info.get("ops", []):
        op_name = op.get("op_name")
        if op_name == "FULLY_CONNECTED":
            has_fc = True
            fc_scale = op.get("fc_scale", 0.01)
            fc_scales.append(fc_scale)
        elif op_name == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            has_lstm = True
            lstm_params = op.get("lstm_params")
        elif op_name == "CONV_2D":
            has_conv = True
        elif op_name == "DEPTHWISE_CONV_2D":
            has_dw = True

    if has_lstm and lstm_params is None:
        fatal_error("模型包含 LSTM 算子但未提取到参数",
                    "请检查模型是否为标准 TFLite LSTM 格式")
    elif has_lstm and lstm_params is not None:
        # 有 LSTM，计算右移位数
        input_scales = lstm_params.get("input_scales",
                                       [DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE])
        recurrent_scales = lstm_params.get("recurrent_scales",
                                           [DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE])

        shifts = []
        for in_s, rec_s in zip(input_scales, recurrent_scales):
            gate_scale = in_s * rec_s
            if gate_scale > 0:
                shift = int(np.log2(1.0 / gate_scale))
            else:
                shift = DEFAULT_SHIFT
            # 限制 shift 范围，防止 LUT 索引越界
            shift = max(4, min(shift, 12))  # 经验范围，基于常见模型统计
            shifts.append(shift)

        lstm_params["shifts"] = shifts
        info(f"LSTM 右移位数: i={shifts[0]}, f={shifts[1]}, g={shifts[2]}, o={shifts[3]}")
    else:
        pass

    # FIXME 多层网络会出错
    # 如果有多个 FC，取第一个（或平均值）
    fc_scale = fc_scales[0] if fc_scales else None

    if fc_scale is not None:
        # FIXME output_scale 暂时用 1.0
        fc_multiplier, fc_shift = calculate_multiplier_shift_from_scale(
            fc_scale, 1.0)
    else:
        fc_multiplier, fc_shift = 213512, -30  # fallback

    # 构建 include 列表
    includes = []
    if has_fc:
        includes.append('#include "fc_weights.h"')
    if has_lstm:
        includes.append('#include "lstm_weights.h"')
    if has_conv:
        includes.append('#include "conv_weights.h"')
    if has_dw:
        includes.append('#include "dw_weights.h"')

    tensor_sizes = {}
    tensor_shapes = {}
    for tensor_idx, tensor_info in tensors.items():
        size = 1
        shape = tensor_info.get("shape", [])
        for dim in shape:
            size *= int(dim)
        tensor_sizes[int(tensor_idx)] = size
        tensor_shapes[int(tensor_idx)] = [int(dim) for dim in shape]

    # 提取所有 Reshape 算子的目标形状
    reshape_targets = []
    for op in execution_order:
        if op.get("op_name") == "RESHAPE":
            target_shape = op.get("reshape_target_shape", [])
            if target_shape:
                reshape_targets.append("{" + ", ".join(
                    str(int(s)) for s in target_shape) + "}")
            else:
                reshape_targets.append("{0}")

    fc_params = {}
    for op in execution_order:
        if op.get("op_name") == "FULLY_CONNECTED":
            # 获取输入张量大小
            input_idx = op["input_indices"][0]  # FC 的第一个输入是数据
            input_size = tensor_sizes.get(input_idx, 0)
            # 输出大小
            output_idx = op["output_indices"][0]
            output_size = tensor_sizes.get(output_idx, 0)
            fc_params[op["index"]] = {
                "input_size": input_size,
                "output_size": output_size,
            }

    for op in execution_order:
        if op.get("op_name") == "CONV_2D":
            # 从 model_info 中查找完整的 conv_params
            for orig_op in model_info.get("ops", []):
                if orig_op.get("index") == op["index"]:
                    op["conv_params"] = orig_op.get("conv_params", {})
                    break

    # 计算输入大小
    input_size_1 = 0
    input_size_2 = 0
    if len(model_info["input"]) == 1:
        for dim in model_info["input"][0]["shape"]:
            input_size_1 = input_size_1 * dim if input_size_1 else dim
    elif len(model_info["input"]) == 2:
        for dim in model_info["input"][0]["shape"]:
            input_size_1 = input_size_1 * dim if input_size_1 else dim
        for dim in model_info["input"][1]["shape"]:
            input_size_2 = input_size_2 * dim if input_size_2 else dim
    else:
        fatal_error(f"不支持 {len(model_info['input'])} 个输入的模型",
                    "当前支持 1 或 2 个输入")

    # 确保 lstm_params 有默认值，不为 None（即使没有 LSTM）
    if lstm_params is None:
        lstm_params = {
            "time_steps": 0,
            "batch_size": 0,
            "input_size": 0,
            "hidden_size": 0,
            "shifts": [8, 8, 8, 8],
            "input_scale": 0.00390625,
            "input_zp": 0,
        }
    context = {
        "input_size": input_size,
        "output_size": output_size,
        "inference_func": inference_func,
        "includes": "\n".join(includes),
        "has_fc": has_fc,
        "has_lstm": has_lstm,
        "has_conv": has_conv,
        "has_dw": has_dw,
        "target": target,
        "model_header": "model.h",  # 固定名称，用于 main_test.c 包含
        "lstm_time_steps": lstm_params["time_steps"],
        "lstm_batch_size": lstm_params["batch_size"],
        "lstm_input_size": lstm_params["input_size"],
        "lstm_hidden_size": lstm_params["hidden_size"],
        "lstm_input_scale": lstm_params.get("input_scale", 0.00390625),  # 1/256
        "lstm_input_zp": lstm_params.get("input_zp", 0),
        "lstm_shifts": lstm_params.get("shifts", [8, 8, 8, 8]),  # 默认 8
        "tensor_sizes": tensor_sizes,
        "tensor_shapes": tensor_shapes,
        "execution_order": execution_order,
        "last_output_tensor": execution_order[-1]["output_indices"][0],
        "reshape_targets": reshape_targets,
        "fc_params": fc_params,
        "inputs_count": len(model_info["input"]),
        "INPUT_SIZE_1": input_size_1,
        "INPUT_SIZE_2": input_size_2,
        "fc_multiplier": fc_multiplier,
        "fc_shift": fc_shift,
    }

    # 先生成文件，决定是否编译LSTM算子
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_features.txt", "w") as f:
        if has_lstm:
            f.write("HAS_LSTM\n")
        if has_fc:
            f.write("HAS_FC\n")

    # 生成 model.c
    with open(template_dir / 'model.c.tpl', 'r') as f:
        tmpl = Template(f.read())
    model_c = tmpl.render(**context)

    # 生成 model.h
    with open(template_dir / 'model.h.tpl', 'r') as f:
        tmpl = Template(f.read())
    model_h = tmpl.render(**context)

    result = {
        'model.c': model_c,
        'model.h': model_h,
    }

    # 可选：生成测试 main
    if with_test_main:
        with open(template_dir / 'main_test.c.tpl', 'r') as f:
            tmpl = Template(f.read())
        result['main_test.c'] = tmpl.render(**context)

    # ========== 生成代码后更新状态 ==========
    for op in ops:
        if op["state"] == "translated":
            op["state"] = "generated"
            op["pass_flags"]["codegen"] = "success"

    return result


def copy_files_to_build(output_dir: Path, target: str, mode: str, accel: str):
    """
    拷贝构建所需的所有文件到 tinymlc_generated/

    Args:
        output_dir: 输出目录 (tinymlc_generated)
        target: 目标架构 (riscv / arm)
        mode: 构建模式 (debug / release)
        accel: 加速库
    """
    # 确定源目录
    ops_root = Path(__file__).parent.parent / "ops"
    src_dir = ops_root / target

    if not src_dir.exists():
        fatal_error(f"架构目录不存在: {src_dir}", f"支持的架构: riscv, arm")

    # 1. 拷贝公共头文件
    include_src = ops_root / "include"
    if include_src.exists():
        shutil.copytree(include_src, output_dir / "include", dirs_exist_ok=True)

    # 2. 拷贝 C 算子 (ops/c/*.c) 到 output_dir/c/
    c_src = ops_root / "c"
    if c_src.exists():
        shutil.copytree(c_src, output_dir / "c", dirs_exist_ok=True)

    # 3. 拷贝目标架构的 .c 和 .S 文件
    for file in src_dir.glob("*.c"):
        shutil.copy2(file, output_dir / file.name)
    for file in src_dir.glob("*.S"):
        shutil.copy2(file, output_dir / file.name)
    for file in src_dir.glob("*.ld"):
        shutil.copy2(file, output_dir / file.name)

    # 4. 拷贝对应的 build 脚本
    if accel != 'none':
        build_script = src_dir / f"build_{target}_{accel.replace("-", "_")}_{mode}.sh"
    else:
        build_script = src_dir / f"build_{target}_{mode}.sh"

    if Path(build_script).exists():
        dest_build_script = output_dir / build_script.name
        shutil.copy2(build_script, dest_build_script)
        try:
            current_mode = dest_build_script.stat().st_mode
            dest_build_script.chmod(
                current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    else:
        fatal_error(f"构建脚本不存在: {build_script}",
                    suggestion=f"请检查加速器类型 {accel} 是否支持")

    # 5. 拷贝 LSTM 相关文件（如果有）
    lstm_src = ops_root / "lstm"
    if lstm_src.exists():
        shutil.copytree(lstm_src, output_dir / "lstm", dirs_exist_ok=True)


def calculate_multiplier_shift(input_scale, weight_scale, output_scale):
    """
    计算 CMSIS-NN 的 multiplier 和 shift
    使得 scale ≈ multiplier / 2^31
    """
    # 有效 scale = input_scale * weight_scale / output_scale
    effective_scale = (input_scale * weight_scale) / output_scale

    # 计算 shift 和 multiplier
    shift = 0
    multiplier = 0

    # 找到合适的 shift 使得 multiplier 在 Q31 范围内
    # 标准 CMSIS-NN 做法：multiplier = effective_scale * 2^31
    # 然后调整 shift
    for s in range(-31, 32):
        mult = effective_scale * (1 << 31)
        # 如果 mult 在 Q31 范围内，记录，32位有符号数的范围 INT32_MIN～INT32_MAX
        if -2147483648 < mult < 2147483647:
            shift = s
            multiplier = int(round(mult))
            break

    return multiplier, shift


def calculate_multiplier_shift_from_scale(weight_scale, output_scale):
    """从 weight_scale 和 output_scale 计算 multiplier 和 shift"""
    effective_scale = weight_scale / output_scale
    multiplier = int(round(effective_scale * (1 << 31)))
    shift = 0
    return multiplier, shift


def extract_all_weights_tflite(interpreter, model_info):
    """从 TFLite 模型中提取所有权重"""
    fc_op_info = None
    lstm_op_info = None
    conv_op_info = None

    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            fc_op_info = op
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_op_info = op
        elif op["op_name"] == "CONV_2D":
            conv_op_info = op

    fc_weights, fc_bias = extract_fc_weights(interpreter, fc_op_info) if fc_op_info else (None, None)
    lstm_weights = extract_lstm_weights(interpreter, lstm_op_info) if lstm_op_info else None
    conv_weights, conv_bias = extract_conv_weights(interpreter, conv_op_info) if conv_op_info else (None, None)

    return fc_weights, fc_bias, lstm_weights, conv_weights, conv_bias


def generate_weight_headers(output_dir, fc_weights, fc_bias, lstm_weights, conv_weights, conv_bias):
    """生成所有权重头文件"""
    if fc_weights is not None:
        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// 自动从模型提取的 FC 层权重和 bias\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        info(f"已生成: {output_dir}/fc_weights.h")

    if lstm_weights and lstm_weights['input']:
        with open(output_dir / 'lstm_weights.h', 'w') as f:
            f.write("// 自动从模型提取的 LSTM 各门权重和 bias\n")
            f.write("// 顺序: i, f, g, o\n\n")
            export_concatenated_weights(lstm_weights['input'], f, 'lstm_input_weights', 'int8')
            export_concatenated_weights(lstm_weights['recurrent'], f, 'lstm_recurrent_weights', 'int8')
            export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')
        info(f"已生成: {output_dir}/lstm_weights.h")

    if conv_weights is not None:
        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write("// 自动从模型提取的 CONV_2D 权重和 bias\n\n")
            export_weights_to_c(conv_weights, "conv_weights", f)
            if conv_bias is not None:
                export_bias_to_c(conv_bias, "conv_bias", f)
        info(f"已生成: {output_dir}/conv_weights.h")


def dump_model_info(model_info):
    """打印模型信息"""
    info("\n=== 模型信息 ===")
    info(f"输入张量: {len(model_info['input'])}")
    for inp in model_info["input"]:
        info(f"  - {inp.get('name', 'unnamed')}: shape={inp['shape']}, dtype={inp['dtype']}")
    info(f"输出张量: {len(model_info['output'])}")
    for out in model_info["output"]:
        info(f"  - {out.get('name', 'unnamed')}: shape={out['shape']}, dtype={out['dtype']}")
    info(f"\n算子数量: {len(model_info['ops'])}")
    for op in model_info["ops"]:
        info(f"\n  [{op['index']}] {op['op_name']}")
        info(f"      输入:")
        for inp in op.get("input_details", []):
            info(f"        - [{inp.get('index', '?')}] {inp.get('name', 'unknown')}: shape={inp.get('shape', [])}, size={inp.get('size', 0)}")
        info(f"      输出:")
        for out in op.get("output_details", []):
            info(f"        - [{out.get('index', '?')}] {out.get('name', 'unknown')}: shape={out.get('shape', [])}, size={out.get('size', 0)}")


def quantize_to_int8(tensor):
    """float32 量化为 int8"""
    min_val = tensor.min()
    max_val = tensor.max()
    # 对称量化
    scale = max(abs(min_val), abs(max_val)) / 127.0
    if scale == 0:
        scale = 1.0
    quantized = np.round(tensor / scale).astype(np.int8)
    return quantized, scale


def export_onnx_weights(output_dir, weights):
    """从 ONNX 权重字典导出 C 头文件"""
    # FC 权重
    fc_weight = weights.get("fc1.weight")
    fc_bias = weights.get("fc1.bias")
    if fc_weight is not None and fc_bias is not None:
        fc_weight_int8, fc_scale = quantize_to_int8(fc_weight)
        # bias 需要乘以 scale 再转 int32
        fc_bias_int32 = (fc_bias / fc_scale).astype(np.int32)

        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// 自动从 ONNX 模型提取的 FC 层权重和 bias（int8 量化）\n")
            export_weights_to_c(fc_weight_int8, "fc_weights", f)
            export_bias_to_c(fc_bias_int32, "fc_bias", f)
        info(f"FC 量化完成: scale={fc_scale}")

        # CONV_2D 权重
        conv_weight = weights.get("conv1.weight")
        conv_bias = weights.get("conv1.bias")
        if conv_weight is not None and conv_bias is not None:
            conv_weight_int8, conv_scale = quantize_to_int8(conv_weight)
            conv_bias_int32 = (conv_bias / conv_scale).astype(np.int32)

            with open(output_dir / 'conv_weights.h', 'w') as f:
                f.write(
                    "// 自动从 ONNX 模型提取的 CONV_2D 层权重和 bias（int8 量化）\n")
                export_weights_to_c(conv_weight_int8, "conv_weights", f)
                export_bias_to_c(conv_bias_int32, "conv_bias", f)
            info(f"CONV_2D 量化完成: scale={conv_scale}")

    # LSTM 权重（如果有）
    # 可根据权重名称前缀判断
    lstm_prefixes = ["lstm.", "lstm_"]
    lstm_weights = {}
    for name, tensor in weights.items():
        for prefix in lstm_prefixes:
            if name.startswith(prefix):
                lstm_weights[name] = tensor
                break
    if lstm_weights:
        # TODO: 导出 LSTM 权重
        pass


def main():
    parser = argparse.ArgumentParser(description="tinymlc - TinyML Compiler")
    parser.add_argument("model", help="TFLite 或 ONNX 模型文件路径")
    parser.add_argument("--entry-point", default="tinymlc_inference",
                        help="推理函数名称 (默认: tinymlc_inference)")
    parser.add_argument("--with-test-main", action="store_true",
                        help="生成测试用的 main 函数")
    parser.add_argument("--run", action="store_true",
                        help="生成后自动运行构建脚本")
    parser.add_argument("--arch", default="riscv",
                        help="目标芯片架构 (默认: riscv)")
    parser.add_argument("--accel", default="none",
                        help="加速库类型: none, cmsis-nn, nmsis-nn, nuclei-ai, ...")
    parser.add_argument("--acc-lib-inc",
                        default="third_party/CMSIS-NN-7.0.0/Include",
                        help="算子加速库头文件路径")
    parser.add_argument("--acc-lib-lib",
                        default="third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a",
                        help="算子加速库静态库路径")
    parser.add_argument("-o", "--output-dir", default="tinymlc_generated",
                        help="输出目录 (默认: tinymlc_generated)")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详细信息")

    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        fatal_error(f"模型文件不存在: {model_path}", "请检查文件路径")

    info(f"正在解析模型: {model_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. 根据模型格式选择解析路径
    # ==========================================
    if model_path.endswith(".tflite"):
        model_info = parse_model_tflite(model_path)
        interpreter = LiteRTInterpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # 提取权重
        fc_weights, fc_bias, lstm_weights, conv_weights, conv_bias = extract_all_weights_tflite(
            interpreter, model_info
        )
        generate_weight_headers(output_dir, fc_weights, fc_bias, lstm_weights,
                                conv_weights, conv_bias)
    elif model_path.endswith(".onnx"):
        model_info = parse_model_onnx(model_path)
        export_onnx_weights(output_dir, model_info.get("weights", {}))
    else:
        fatal_error("不支持的模型格式", "支持 .tflite 和 .onnx")

    # ==========================================
    # 2. 打印执行信息
    # ==========================================
    if args.verbose:
        dump_model_info(model_info)

    # ==========================================
    # 3. 生成共用模型 C 代码
    # ==========================================
    info("正在生成 C 代码...")
    target = args.arch
    mode = "debug" if args.with_test_main else "release"

    generated_files = generate_c_code(
        model_info, output_dir, target,
        inference_func=args.entry_point,
        with_test_main=args.with_test_main
    )

    for filename, content in generated_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        info(f"生成: {output_path}")

    # ==========================================
    # 4. 生成 LUT 和构建脚本
    # ==========================================
    generate_lut(output_dir)
    copy_files_to_build(output_dir, target, mode, args.accel)

    if args.accel != 'none':
        script_name = f"build_{target}_{args.accel.replace('-', '_')}_{mode}.sh"
    else:
        script_name = f"build_{target}_{mode}.sh"

    # ==========================================
    # 5. 自动运行（可选）
    # ==========================================
    if args.run:
        script_path = output_dir / script_name
        try:
            script_path.chmod(0o755)
        except OSError:
            pass
        info(f"执行: {script_path} {args.model}")
        result = subprocess.run([str(script_path.resolve()), args.model], cwd=output_dir)
        sys.exit(result.returncode)

    info(f"完成! 输出目录: {output_dir}")
    info("\n下一步:")
    info(f"  cd {output_dir}")
    info(f"  ./{script_name} {args.model}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
