#!/usr/bin/env python3
"""
Generate sigmoid and tanh int16 lookup tables (256 entries,
for int8 quantized LSTM)
"""

import numpy as np
from pathlib import Path
from jinja2 import Template

from utils.dump import fatal_error, info


def generate_sigmoid_lut():
    """Generate sigmoid LUT, output int16 range [0, 32767]"""
    x = np.linspace(-8, 8, 256, endpoint=False)
    y = 1 / (1 + np.exp(-x))
    # Quantize to int16: 0~1 maps to 0~32767
    y_scaled = np.round(y * 32767).astype(np.int16)
    return y_scaled


def generate_tanh_lut():
    """Generate tanh LUT, output int16 range [-32768, 32767]"""
    x = np.linspace(-8, 8, 256, endpoint=False)
    y = np.tanh(x)
    # Quantize to int16: -1~1 maps to -32768~32767
    y_scaled = np.round(y * 32767).astype(np.int16)
    return y_scaled


def generate_lut(output_dir: Path):
    """Generate LUT header and source files"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        sigmoid_lut = generate_sigmoid_lut()
        tanh_lut = generate_tanh_lut()

        # Render lut.h.tpl
        template_path = Path(__file__).parent / 'templates' / 'lut.h.tpl'
        with open(template_path, 'r') as f:
            template = Template(f.read())

        lut_h = template.render(
            sigmoid_lut=sigmoid_lut.tolist(),
            tanh_lut=tanh_lut.tolist()
        )

        with open(output_dir / 'lut.h', 'w') as f:
            f.write(lut_h)

        # Render lut.c.tpl
        template_path = Path(__file__).parent / 'templates' / 'lut.c.tpl'
        with open(template_path, 'r') as f:
            template = Template(f.read())

        lut_c = template.render()

        with open(output_dir / 'lut.c', 'w') as f:
            f.write(lut_c)

        info(f"Generated: {output_dir}/lut.h")
        info(f"Generated: {output_dir}/lut.c")

    except Exception as e:
        fatal_error(
            f"LUT generation failed: {e}",
            "Check numpy and file system permissions")


def main():
    output_dir = Path('tinymlc_generated')
    generate_lut(output_dir)


if __name__ == "__main__":
    main()
