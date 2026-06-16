#!/usr/bin/env python3
"""
生成 sigmoid 和 tanh 的 int16 查找表（256 条目，用于 int8 量化 LSTM）
"""

import numpy as np
from pathlib import Path
from jinja2 import Template

from tinymlc.utils import fatal_error, warning, info


def generate_sigmoid_lut():
    """生成 sigmoid LUT，输出 int16 范围 [0, 32767]"""
    x = np.linspace(-8, 8, 256, endpoint=False)
    y = 1 / (1 + np.exp(-x))
    # 量化到 int16：0~1 映射到 0~32767
    y_scaled = np.round(y * 32767).astype(np.int16)
    return y_scaled


def generate_tanh_lut():
    """生成 tanh LUT，输出 int16 范围 [-32768, 32767]"""
    x = np.linspace(-8, 8, 256, endpoint=False)
    y = np.tanh(x)
    # 量化到 int16：-1~1 映射到 -32768~32767
    y_scaled = np.round(y * 32767).astype(np.int16)
    return y_scaled


def generate_lut(output_dir: Path):
    """生成 LUT 头文件和源文件"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        sigmoid_lut = generate_sigmoid_lut()
        tanh_lut = generate_tanh_lut()

        # 渲染 lut.h.tpl
        template_path = Path(__file__).parent / 'templates' / 'lut.h.tpl'
        with open(template_path, 'r') as f:
            template = Template(f.read())

        lut_h = template.render(
            sigmoid_lut=sigmoid_lut.tolist(),
            tanh_lut=tanh_lut.tolist()
        )

        with open(output_dir / 'lut.h', 'w') as f:
            f.write(lut_h)

        # 渲染 lut.c.tpl
        template_path = Path(__file__).parent / 'templates' / 'lut.c.tpl'
        with open(template_path, 'r') as f:
            template = Template(f.read())

        lut_c = template.render()

        with open(output_dir / 'lut.c', 'w') as f:
            f.write(lut_c)

        info(f"已生成: {output_dir}/lut.h")
        info(f"已生成: {output_dir}/lut.c")

    except Exception as e:
        fatal_error(f"LUT 生成失败: {e}", "检查 numpy 和文件系统权限")


def main():
    output_dir = Path('tinymlc_generated')
    generate_lut(output_dir)


if __name__ == "__main__":
    main()
