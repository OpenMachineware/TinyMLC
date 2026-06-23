#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
Convert TFLite/ONNX models to C code executable on MCU
"""

import subprocess
import sys
import argparse
import numpy as np

from pathlib import Path

from tinymlc.converter.parser_litert import (
    parse_model_tflite,
    extract_all_weights_litert,
)
from tinymlc.converter.parser_onnx import (
    parse_model_onnx,
    extract_all_weights_onnx,
)
from tinymlc.codegen import generate_c_code, copy_files_to_build
from tinymlc.generate_lut import generate_lut
from tinymlc.converter.export_weights import export_model_weights
from utils.dump import fatal_error, warning, info, dump_model_info


def main():
    parser = argparse.ArgumentParser(
        description="tinymlc - TinyML Compiler")
    parser.add_argument(
        "model",
        help="TFLite or ONNX model file path")
    parser.add_argument(
        "--entry-point", default="tinymlc_inference",
        help="Inference function name (default: tinymlc_inference)")
    parser.add_argument(
        "--with-test-main", action="store_true",
        help="Generate test main function")
    parser.add_argument(
        "--run", action="store_true",
        help="Run build script after generation")
    parser.add_argument(
        "--arch", default="riscv",
        help="Target chip architecture (default: riscv)")
    parser.add_argument(
        "--accel", default="none",
        help="Acceleration library type: none, cmsis-nn, "
             "nmsis-nn, nuclei-ai, ...")
    parser.add_argument(
        "--acc-lib-inc",
        default="third_party/CMSIS-NN-7.0.0/Include",
        help="Operator acceleration library header path")
    parser.add_argument(
        "--acc-lib-lib",
        default="third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a",
        help="Operator acceleration library static library path")
    parser.add_argument(
        "-o", "--output-dir", default="tinymlc_generated",
        help="Output directory (default: tinymlc_generated)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed info")

    args = parser.parse_args()

    # Validate arch and accel combination
    arch = args.arch
    accel = args.accel

    if arch == "arm":
        if accel not in ("none", "cmsis-nn"):
            fatal_error(
                f"Invalid --accel '{accel}' for --arch arm",
                "Supported accel for ARM: none, cmsis-nn")
    elif arch == "riscv":
        if accel not in ("none", "nmsis-nn", "nuclei-ai"):
            fatal_error(
                f"Invalid --accel '{accel}' for --arch riscv",
                "Supported accel for RISC-V: none, nmsis-nn, nuclei-ai")
    else:
        fatal_error(
            f"Invalid --arch '{arch}'",
            "Supported architectures: arm, riscv")

    model_path = args.model
    if not Path(model_path).exists():
        fatal_error(f"Model file not found: {model_path}", "Please check file path")

    info(f"Parsing model: {model_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. Unified parsing and weight extraction
    # ==========================================
    if model_path.endswith(".tflite"):
        model_info = parse_model_tflite(model_path)
        # Extract weights (interpreter created internally)
        extract_all_weights_litert(model_path, model_info)
        # Export weights using unified function
        quant_scales = export_model_weights(output_dir, model_info)
        model_info["quant_scales"] = quant_scales
    elif model_path.endswith(".onnx"):
        model_info = parse_model_onnx(model_path)
        # Extract weights (model_path for consistency)
        extract_all_weights_onnx(model_path, model_info)
        # Export weights using unified function
        quant_scales = export_model_weights(output_dir, model_info)
        model_info["quant_scales"] = quant_scales
    else:
        fatal_error("Unsupported model format", "Supported: .tflite and .onnx")

    # ==========================================
    # 2. Print execution info
    # ==========================================
    if args.verbose:
        dump_model_info(model_info)

    # ==========================================
    # 3. Generate common model C code
    # ==========================================
    info("Generating C code...")
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
        info(f"Generated: {output_path}")

    # ==========================================
    # 4. Generate LUT and build script
    # ==========================================
    generate_lut(output_dir)
    copy_files_to_build(output_dir, target, mode, args.accel)

    if args.accel != 'none':
        script_name = (
            f"build_{target}_{args.accel.replace('-', '_')}_{mode}.sh")
    else:
        script_name = f"build_{target}_{mode}.sh"

    # ==========================================
    # 5. Auto run (optional)
    # ==========================================
    if args.run:
        script_path = output_dir / script_name
        try:
            script_path.chmod(0o755)
        except OSError:
            pass
        info(f"Executing: {script_path} {args.model}")
        result = subprocess.run(
            [str(script_path.resolve()), args.model],
            cwd=output_dir)
        sys.exit(result.returncode)

    info(f"Done! Output directory: {output_dir}")
    info("\nNext steps:")
    info(f"  cd {output_dir}")
    info(f"  ./{script_name} {args.model}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
