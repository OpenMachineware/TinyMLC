#!/usr/bin/env python3
"""CLI tool to extract weights from ONNX models

Usage:
    uv run python -m tinymlc.extract_onnx_weights model.onnx --output-dir output/

Or:
    python -m tinymlc.extract_onnx_weights model.onnx --output-dir output/
"""

import argparse
from pathlib import Path

from tinymlc.parser_onnx import (
    parse_model_onnx,
    extract_all_weights_onnx,
)
from tinymlc.translator import export_model_weights
from utils.dump import info, fatal_error


WEIGHTLESS_OPS = ["ADD", "SOFTMAX", "RESHAPE", "RELU", "SIGMOID", "TANH", "SUB", "MULTIPLY"]


def main():
    parser = argparse.ArgumentParser(
        description='Extract weights from ONNX model')
    parser.add_argument('model', help='ONNX model file path')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='Output directory (default: tinymlc_generated)')
    args = parser.parse_args()

    model_path = args.model

    # 1. Parse model
    info(f"Loading model: {model_path}")
    model_info = parse_model_onnx(model_path)

    # 2. Extract weights (model_path for consistency)
    extract_all_weights_onnx(model_path, model_info)

    # 3. Check if any weights were extracted
    has_weights = bool(model_info.get("weights"))
    if not has_weights:
        has_weightless_op = any(
            op["op_name"] in WEIGHTLESS_OPS for op in model_info["ops"]
        )
        if not has_weightless_op:
            fatal_error(
                "No weights found",
                "Check if model contains supported operators"
            )
        else:
            info("Note: Model only contains weightless operators, continuing...")

    # 4. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Export weights using unified function
    export_model_weights(output_dir, model_info)

    info(f"\nDone! Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    main()