#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
将 TFLite 模型转换为 MCU 可执行的 C 代码
"""

import sys
import argparse
from pathlib import Path

from tinymlc.extract_weights import (extract_fc_weights, extract_lstm_weights,
                                     export_weights_to_c, export_bias_to_c,
                                     export_concatenated_weights,
                                     export_concatenated_bias)
# 检查依赖
try:
    import tensorflow as tf
except ImportError:
    print("错误: 请先安装 tensorflow: uv add tensorflow")
    sys.exit(1)

try:
    from jinja2 import Template
except ImportError:
    print("错误: 请先安装 jinja2: uv add jinja2")
    sys.exit(1)


def parse_model(interpreter):
    """解析 TFLite 模型，返回算子列表和张量信息"""
    # 获取输入输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 获取所有张量详细信息
    tensor_details = interpreter.get_tensor_details()

    # 构建张量索引 -> 信息的映射
    tensor_map = {}
    for t in tensor_details:
        tensor_map[t["index"]] = {
            "name": t["name"],
            "shape": t["shape"],
            "dtype": str(t["dtype"]),
            "size": (
                t["shape"].num_elements() if hasattr(t["shape"], "num_elements") else 1
            ),
        }

    # 获取所有算子及其详细信息
    ops = []
    for op in interpreter._get_ops_details():
        op_info = {
            "index": op["index"],
            "op_name": op["op_name"],
            "inputs": op["inputs"],
            "outputs": op["outputs"],
            "input_details": [],
            "output_details": [],
        }

        # 获取输入张量的详细信息
        for inp_idx in op["inputs"]:
            if inp_idx != -1:  # -1 表示没有这个输入
                info = tensor_map.get(inp_idx, {})
                op_info["input_details"].append(
                    {
                        "index": inp_idx,
                        "name": info.get("name", "unknown"),
                        "shape": info.get("shape", []),
                        "size": info.get("size", 0),
                    }
                )

        # 获取输出张量的详细信息
        for out_idx in op["outputs"]:
            if out_idx != -1:
                info = tensor_map.get(out_idx, {})
                op_info["output_details"].append(
                    {
                        "index": out_idx,
                        "name": info.get("name", "unknown"),
                        "shape": info.get("shape", []),
                        "size": info.get("size", 0),
                    }
                )

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "tensors": tensor_map,
    }


def generate_c_code(model_info: dict,
                    inference_func: str = "tinymlc_inference",
                    with_test_main: bool = False,
                    output_dir: str = ".") -> dict:
    """生成 C 代码和头文件"""
    from jinja2 import Template
    from pathlib import Path

    template_dir = Path(__file__).parent / 'templates'

    # 计算输入输出大小
    input_size = 1
    for dim in model_info['input'][0]['shape']:
        input_size *= dim

    output_size = 1
    for dim in model_info['output'][0]['shape']:
        output_size *= dim

    context = {
        'input_size': input_size,
        'output_size': output_size,
        'inference_func': inference_func,
        'weights_header': 'fc_weights.h',  # 暂时固定，后续可配置
        'model_header': 'model.h',
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

    return result


def generate_lut(output_dir: Path):
    """生成 sigmoid 和 tanh 的 LUT 表"""
    import numpy as np
    from jinja2 import Template

    # 生成 LUT 数据
    x = np.linspace(-8, 8, 256, endpoint=False)

    # Sigmoid: 范围 [0,1]，量化到 int16 [0, 32767]
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_lut = np.round(sigmoid * 32767).astype(np.int16)

    # Tanh: 范围 [-1,1]，量化到 int16 [-32768, 32767]
    tanh = np.tanh(x)
    tanh_lut = np.round(tanh * 32767).astype(np.int16)

    # 渲染模板
    template_path = Path(__file__).parent / 'templates' / 'lut.h.tpl'
    with open(template_path, 'r') as f:
        template = Template(f.read())

    lut_h = template.render(
        sigmoid_lut=sigmoid_lut.tolist(),
        tanh_lut=tanh_lut.tolist()
    )

    # 写入文件
    with open(output_dir / 'lut.h', 'w') as f:
        f.write(lut_h)

    # 生成空的 lut.c（保持一致性）
    with open(output_dir / 'lut.c', 'w') as f:
        f.write('// LUT implementation is in lut.h\n')
        f.write('#include "lut.h"\n')

    print(f"已生成: {output_dir}/lut.h")
    print(f"已生成: {output_dir}/lut.c")


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
        print(f"错误: 模型文件不存在: {args.model}")
        return 1

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
    # 提取权重
    fc_weights, fc_bias = extract_fc_weights(interpreter)
    lstm_weights = extract_lstm_weights(interpreter)

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
