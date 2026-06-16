#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
将 TFLite 模型转换为 MCU 可执行的 C 代码
"""

import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from jinja2 import Template

from tinymlc.extract_weights import (extract_fc_weights, extract_lstm_weights,
                                     export_weights_to_c, export_bias_to_c,
                                     export_concatenated_weights,
                                     export_concatenated_bias)
from tinymlc.generate_lut import generate_lut
from tinymlc.parser import parse_model
from tinymlc.utils import fatal_error, warning, info


def generate_c_code(model_info,
                    inference_func="tinymlc_inference",
                    with_test_main=False, output_dir="."):
    ops = model_info.get("ops", [])
    for op in ops:
        if op["state"] != "translated":
            fatal_error(
                f"算子 {op['op_name']} 状态为 {op.get('state')}，无法生成代码",
                f"Pass flags: {op.get('pass_flags', {})}")

    """生成 C 代码和头文件"""
    template_dir = Path(__file__).parent / 'templates'

    # 计算输入输出大小
    input_size = 1
    for dim in model_info['input'][0]['shape']:
        input_size *= dim

    output_size = 1
    for dim in model_info['output'][0]['shape']:
        output_size *= dim

    # 检测模型包含的算子类型
    has_fc = False
    has_lstm = False

    # 查找 LSTM 参数
    lstm_params = None
    for op in model_info.get("ops", []):
        op_name = op.get("op_name")
        if op_name == "FULLY_CONNECTED":
            has_fc = True
        elif op_name == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            has_lstm = True
            lstm_params = op.get("lstm_params")

    # 构建 include 列表
    includes = []
    if has_fc:
        includes.append('#include "fc_weights.h"')
    if has_lstm:
        includes.append('#include "lstm_weights.h"')

    if lstm_params is None:
        warning("未找到 LSTM 参数，使用默认值（可能出错）")
        lstm_params = {
            "time_steps": 28,
            "batch_size": 1,
            "input_size": 28,
            "hidden_size": 20,
            "shifts": [8, 8, 8, 8],  # 默认右移 8 位
        }
    else:
        # 计算每个 gate 的右移位数
        # 输入权重 scale (i,f,g,o) 和递归权重 scale (i,f,g,o)
        input_scales = lstm_params.get("input_scales", [0.01, 0.01, 0.01, 0.01])
        recurrent_scales = lstm_params.get("recurrent_scales",
                                           [0.01, 0.01, 0.01, 0.01])

        shifts = []
        for in_s, rec_s in zip(input_scales, recurrent_scales):
            gate_scale = in_s * rec_s
            # 计算右移位数：希望 gate >> shift 落在 [-128,127] 范围
            # shift = floor(log2(1 / gate_scale))
            if gate_scale > 0:
                shift = int(np.log2(1.0 / gate_scale))
            else:
                shift = 8
            # 限制范围 4-12，避免溢出
            shift = max(4, min(shift, 12))
            shifts.append(shift)

        lstm_params["shifts"] = shifts
        info(f"LSTM 右移位数: i={shifts[0]}, f={shifts[1]}, g={shifts[2]}, o={shifts[3]}")

    context = {
        "input_size": input_size,
        "output_size": output_size,
        "inference_func": inference_func,
        "includes": "\n".join(includes),
        "has_fc": has_fc,
        "has_lstm": has_lstm,
        "model_header": "model.h",  # 固定名称，用于 main_test.c 包含
        "lstm_time_steps": lstm_params["time_steps"],
        "lstm_batch_size": lstm_params["batch_size"],
        "lstm_input_size": lstm_params["input_size"],
        "lstm_hidden_size": lstm_params["hidden_size"],
        "lstm_input_scale": lstm_params.get("input_scale", 0.00390625),  # 1/256
        "lstm_input_zp": lstm_params.get("input_zp", 0),
        "lstm_shifts": lstm_params.get("shifts", [8, 8, 8, 8]),  # 默认 8
    }

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


def main():
    parser = argparse.ArgumentParser(description="tinymlc - TinyML Compiler")
    parser.add_argument("model", help="TFLite 模型文件路径")
    parser.add_argument("--entry-point", default="tinymlc_inference",
                        help="推理函数名称 (默认: tinymlc_inference)")
    parser.add_argument("--with-test-main", action="store_true",
                        help="生成测试用的 main 函数")
    parser.add_argument("-o", "--output-dir", default="tinymlc_generated",
                        help="输出目录 (默认: tinymlc_generated)")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详细信息")

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not Path(args.model).exists():
        fatal_error(f"模型文件不存在: {args.model}", "请检查文件路径")

    # 创建 interpreter 用于解析模型和提取权重
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    print(f"正在解析模型: {args.model}")
    model_info = parse_model(interpreter)

    if args.verbose:
        print("\n=== 模型信息 ===")
        print(f"输入张量: {len(model_info['input'])}")
        for inp in model_info["input"]:
            print(
                f"  - {inp.get('name', 'unnamed')}: shape={inp['shape']}, dtype={inp['dtype']}"
            )
        print(f"输出张量: {len(model_info['output'])}")
        for out in model_info["output"]:
            print(
                f"  - {out.get('name', 'unnamed')}: shape={out['shape']}, dtype={out['dtype']}"
            )

        print(f"\n算子数量: {len(model_info['ops'])}")
        for op in model_info["ops"]:
            print(f"\n  [{op['index']}] {op['op_name']}")
            print(f"      输入:")
            for inp in op["input_details"]:
                print(
                    f"        - [{inp['index']}] {inp['name']}: shape={inp['shape']}, size={inp['size']}"
                )
            print(f"      输出:")
            for out in op["output_details"]:
                print(
                    f"        - [{out['index']}] {out['name']}: shape={out['shape']}, size={out['size']}"
                )

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取并生成权重文件
    print("正在提取权重...")
    # 找到 FC 算子的 op_info
    fc_op_info = None
    lstm_op_info = None
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            fc_op_info = op
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_op_info = op

    # 提取权重
    if fc_op_info:
        fc_weights, fc_bias = extract_fc_weights(interpreter, fc_op_info)
        if fc_weights is None:
            fatal_error("FC 权重提取失败", "检查模型是否完整")
    else:
        fc_weights, fc_bias = None, None

    if lstm_op_info:
        lstm_weights = extract_lstm_weights(interpreter, lstm_op_info)
    else:
        lstm_weights = None

    # 检查是否有任何权重被提取
    has_fc = fc_weights is not None
    has_lstm = lstm_weights is not None and any(
        v is not None for v in lstm_weights['input'].values())
    if not (has_fc or has_lstm):
        fatal_error("未找到任何权重", "检查模型是否包含支持的算子")

    # 生成 fc_weights.h
    if fc_weights is not None:
        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// 自动从 tflite 提取的 FC 层权重和 bias\n")
            f.write("// 请勿手动修改\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        print(f"已生成: {output_dir}/fc_weights.h")

    # 生成 lstm_weights.h
    if lstm_weights and lstm_weights['input']:
        with open(output_dir / 'lstm_weights.h', 'w') as f:
            f.write("// 自动从 tflite 提取的 LSTM 各门权重和 bias\n")
            f.write("// 顺序: i, f, g, o\n\n")
            f.write("// 请勿手动修改\n\n")

            # 导入拼接函数（需要在文件顶部导入）
            export_concatenated_weights(lstm_weights['input'], f,
                                        'lstm_input_weights', 'int8')
            export_concatenated_weights(lstm_weights['recurrent'], f,
                                        'lstm_recurrent_weights', 'int8')
            export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')

        print(f"已生成: {output_dir}/lstm_weights.h")

    print("正在生成 C 代码...")
    generated_files = generate_c_code(
        model_info,
        inference_func=args.entry_point,
        with_test_main=args.with_test_main
    )

    # 写入所有文件
    for filename, content in generated_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"生成: {output_path}")

    # 生成 LUT
    generate_lut(output_dir)

    print(f"完成! 输出目录: {output_dir}")

    print("\n下一步:")
    print("  1. 查看生成的代码: ls", output_dir)
    print("  2. 编译: riscv64-elf-gcc -march=rv32imac -mabi=ilp32 -static -ffreestanding -nostdlib -c", output_dir)
    print("  3. 链接并烧录到 MCU")

    return 0


if __name__ == "__main__":
    sys.exit(main())
