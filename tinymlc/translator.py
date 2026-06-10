#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
将 TFLite 模型转换为 MCU 可执行的 C 代码
"""

import sys
import argparse
from pathlib import Path

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


def parse_model(model_path: str):
    """解析 TFLite 模型，返回算子列表和张量信息"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

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


def generate_c_code(model_info: dict) -> str:
    """从模板生成 C 代码"""
    from jinja2 import Template
    from pathlib import Path

    template_path = Path(__file__).parent / 'templates' / 'model.c.tpl'
    with open(template_path, 'r') as f:
        template = Template(f.read())

    # 从 model_info 提取输入输出大小
    input_size = 1
    for dim in model_info['input'][0]['shape']:
        input_size *= dim

    output_size = 1
    for dim in model_info['output'][0]['shape']:
        output_size *= dim

    return template.render(
        input_size=input_size,
        output_size=output_size
    )



def main():
    parser = argparse.ArgumentParser(description="tinymlc - TinyML Compiler")
    parser.add_argument("model", help="TFLite 模型文件路径")
    parser.add_argument("-o", "--output", default="model.c", help="输出 C 文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详细信息")

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return 1

    print(f"正在解析模型: {args.model}")
    model_info = parse_model(args.model)

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

    print("正在生成 C 代码...")
    c_code = generate_c_code(model_info)

    with open(args.output, "w") as f:
        f.write(c_code)

    print(f"完成! 输出文件: {args.output}")
    print("\n下一步:")
    print("  1. 查看生成的代码: cat", args.output)
    print("  2. 编译: riscv64-elf-gcc -march=rv32imac -mabi=ilp32 -static -ffreestanding -nostdlib -c", args.output)
    print("  3. 链接并烧录到 MCU")

    return 0


if __name__ == "__main__":
    sys.exit(main())
