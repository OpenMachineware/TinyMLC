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

from tinymlc.model_converter.parser_litert import (
    parse_model_tflite,
    extract_all_weights_litert,
)
from tinymlc.model_converter.parser_onnx import (
    parse_model_onnx,
    extract_all_weights_onnx,
)
from tinymlc.generate_c_code import generate_c_code, copy_files_to_build
from tinymlc.generate_lut import generate_lut
from utils.dump import fatal_error, warning, info


def generate_weight_headers(output_dir, fc_weights, fc_bias,
                             lstm_weights, conv_weights, conv_bias,
                             dw_weights, dw_bias):
    """Generate all weight header files"""
    if fc_weights is not None:
        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// FC layer weights and bias extracted from model\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        info(f"Generated: {output_dir}/fc_weights.h")

    if lstm_weights and lstm_weights['input']:
        with open(output_dir / 'lstm_weights.h', 'w') as f:
            f.write("// LSTM gate weights and bias extracted from model\n")
            f.write("// Order: i, f, g, o\n\n")
            export_concatenated_weights(
                lstm_weights['input'], f,
                'lstm_input_weights', 'int8')
            export_concatenated_weights(
                lstm_weights['recurrent'], f,
                'lstm_recurrent_weights', 'int8')
            export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')
        info(f"Generated: {output_dir}/lstm_weights.h")

    if conv_weights is not None:
        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write("// CONV_2D weights and bias extracted from model\n\n")
            export_weights_to_c(conv_weights, "conv_weights", f)
            if conv_bias is not None:
                export_bias_to_c(conv_bias, "conv_bias", f)
        info(f"Generated: {output_dir}/conv_weights.h")

    if dw_weights is not None:
        with open(output_dir / 'dw_weights.h', 'w') as f:
            f.write("// Depthwise Conv2D weights and bias extracted from model\n\n")
            export_weights_to_c(dw_weights, "dw_weights", f)
            if dw_bias is not None:
                export_bias_to_c(dw_bias, "dw_bias", f)
        info(f"Generated: {output_dir}/dw_weights.h")


def export_model_weights(output_dir, model_info):
    """Unified weight export function for both ONNX and TFLite models.

    Weights are identified by source-specific keys:
    - TFLite: "fc_tflite.weight", "fc_tflite.bias", "lstm_tflite.weight_ih", etc.
    - ONNX: "fc_onnx.weight", "fc_onnx.bias", "conv_onnx.weight", etc.

    Returns quant_scales dict.
    """
    weights = model_info.get("weights", {})
    quant_scales = {}

    # Input scale fixed at 1/256 (symmetric quantization, zero_point=0)
    input_scale = 0.00390625

    # FC weights - try both TFLite and ONNX sources
    fc_weight = weights.get("fc_tflite.weight") or weights.get("fc_onnx.weight")
    fc_bias = weights.get("fc_tflite.bias") or weights.get("fc_onnx.bias")
    if fc_weight is not None and fc_bias is not None:
        fc_scale = 0.01  # default
        if fc_weight.dtype == np.int8:
            # Already quantized (TFLite or QDQ ONNX format)
            fc_weight_int8 = fc_weight
            fc_bias_int32 = fc_bias.astype(np.int32) if fc_bias.dtype != np.int32 else fc_bias
        else:
            # FP32, needs quantization
            fc_weight_int8, fc_scale = quantize_to_int8(fc_weight)
            fc_bias_int32 = (fc_bias / (input_scale * fc_scale)).astype(np.int32)

        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// FC layer weights and bias extracted from model\n\n")
            export_weights_to_c(fc_weight_int8, "fc_weights", f)
            export_bias_to_c(fc_bias_int32, "fc_bias", f)
        info(f"Generated: {output_dir}/fc_weights.h")
        quant_scales["fc_scale"] = fc_scale

    # LSTM weights - TFLite only (ONNX uses decomposed operators)
    lstm_weight_ih = (
        weights.get("lstm_tflite.weight_ih") or
        weights.get("lstm_onnx.weight_ih"))
    lstm_weight_hh = (
        weights.get("lstm_tflite.weight_hh") or
        weights.get("lstm_onnx.weight_hh"))
    lstm_bias = (
        weights.get("lstm_tflite.bias") or
        weights.get("lstm_onnx.bias"))
    if lstm_weight_ih is not None and lstm_weight_hh is not None:
        with open(output_dir / 'lstm_weights.h', 'w') as f:
            f.write("// LSTM gate weights and bias extracted from model\n")
            f.write("// Order: i, f, g, o\n\n")
            export_weights_to_c(lstm_weight_ih, "lstm_input_weights", f)
            export_weights_to_c(lstm_weight_hh, "lstm_recurrent_weights", f)
            if lstm_bias is not None:
                export_bias_to_c(lstm_bias, "lstm_bias", f)
        info(f"Generated: {output_dir}/lstm_weights.h")

    # Conv weights - try both TFLite and ONNX sources
    conv_weight = weights.get("conv_tflite.weight") or weights.get("conv_onnx.weight")
    conv_bias = weights.get("conv_tflite.bias") or weights.get("conv_onnx.bias")
    if conv_weight is not None:
        conv_weight_int8 = conv_weight if conv_weight.dtype == np.int8 else quantize_to_int8(conv_weight)[0]
        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write("// CONV_2D weights and bias extracted from model\n\n")
            export_weights_to_c(conv_weight_int8, "conv_weights", f)
            if conv_bias is not None:
                conv_bias_int32 = conv_bias.astype(np.int32) if conv_bias.dtype != np.int32 else conv_bias
                export_bias_to_c(conv_bias_int32, "conv_bias", f)
        info(f"Generated: {output_dir}/conv_weights.h")

    # Depthwise Conv weights - try both TFLite and ONNX sources
    dw_weight = weights.get("dw_tflite.weight") or weights.get("dw_onnx.weight")
    dw_bias = weights.get("dw_tflite.bias") or weights.get("dw_onnx.bias")
    if dw_weight is not None:
        dw_weight_int8 = dw_weight if dw_weight.dtype == np.int8 else quantize_to_int8(dw_weight)[0]
        with open(output_dir / 'dw_weights.h', 'w') as f:
            f.write("// Depthwise Conv2D weights and bias extracted from model\n\n")
            export_weights_to_c(dw_weight_int8, "dw_weights", f)
            if dw_bias is not None:
                dw_bias_int32 = dw_bias.astype(np.int32) if dw_bias.dtype != np.int32 else dw_bias
                export_bias_to_c(dw_bias_int32, "dw_bias", f)
        info(f"Generated: {output_dir}/dw_weights.h")

    return quant_scales


def dump_model_info(model_info):
    """Print model info"""
    info("\n=== Model Info ===")
    info(f"Input tensors: {len(model_info['input'])}")
    for inp in model_info["input"]:
        info(f"  - {inp.get('name', 'unnamed')}: "
             f"shape={inp['shape']}, dtype={inp['dtype']}")
    info(f"Output tensors: {len(model_info['output'])}")
    for out in model_info["output"]:
        info(f"  - {out.get('name', 'unnamed')}: "
             f"shape={out['shape']}, dtype={out['dtype']}")
    info(f"\nOperator count: {len(model_info['ops'])}")
    for op in model_info["ops"]:
        info(f"\n  [{op['index']}] {op['op_name']}")
        info(f"      Inputs:")
        for inp in op.get("input_details", []):
            info(f"        - [{inp.get('index', '?')}] "
                 f"{inp.get('name', 'unknown')}: "
                 f"shape={inp.get('shape', [])}, "
                 f"size={inp.get('size', 0)}")
        info(f"      Outputs:")
        for out in op.get("output_details", []):
            info(f"        - [{out.get('index', '?')}] "
                 f"{out.get('name', 'unknown')}: "
                 f"shape={out.get('shape', [])}, "
                 f"size={out.get('size', 0)}")


def quantize_to_int8(tensor):
    """
    Quantize float32 weights to int8 (symmetric quantization)

    Symmetric quantization:
    - Value range: -127 ~ 127 (avoid -128 for symmetry)
    - scale = max(|min|, |max|) / 127
    - quantized = round(tensor / scale)

    Args:
        tensor: float32 weight tensor

    Returns:
        quantized: int8 quantized weights
        scale: quantization scale
    """
    min_val = tensor.min()
    max_val = tensor.max()
    # Symmetric quantization: scale = max(|min|, |max|) / 127
    scale = max(abs(min_val), abs(max_val)) / 127.0
    if scale == 0:
        scale = 1.0
    quantized = np.round(tensor / scale).astype(np.int8)
    return quantized, scale


def export_weights_to_c(weights, name, output_file):
    """Export int8 weights to C header file"""
    if weights is None:
        output_file.write(f"// {name} not found, using placeholder\n")
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
    """Export int32 bias to C header file"""
    if bias is None:
        output_file.write(f"// {name} not found, using placeholder\n")
        output_file.write(f"static const int32_t {name}[1] = {{0}};\n\n")
        return

    flat = bias.flatten()
    output_file.write(f"static const int32_t {name}[{flat.size}] = {{\n    ")
    for i, val in enumerate(flat):
        output_file.write(f"{int(val)}")
        if i < flat.size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def export_concatenated_weights(weights_dict, output_file, array_name,
                                dtype='int8'):
    """Export concatenated weight array from gate dictionary.

    Args:
        weights_dict: dict with keys ['i', 'f', 'g', 'o'] containing weight arrays
        output_file: file handle to write to
        array_name: C array name
        dtype: 'int8' or 'int32'
    """
    gate_order = ['i', 'f', 'g', 'o']
    arrays = []
    total_size = 0
    missing_gates = []

    for gate in gate_order:
        w = weights_dict.get(gate)
        if w is None:
            missing_gates.append(gate)
            continue
        arrays.append(w.flatten())
        total_size += w.size

    if missing_gates:
        warning(f"{missing_gates} gate weights missing, padding with zeros")
        # Get shape from first non-None weight
        for gate in gate_order:
            w = weights_dict.get(gate)
            if w is not None:
                shape = w.shape
                for mg in missing_gates:
                    zero_array = np.zeros(shape, dtype=np.int8)
                    arrays.append(zero_array.flatten())
                    total_size += zero_array.size
                break

    if not arrays:
        output_file.write(f"// {array_name} not found, using placeholder\n")
        output_file.write(f"static const int8_t {array_name}[1] = {{0}};\n\n")
        return

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


def export_concatenated_bias(bias_dict, output_file, array_name):
    """Export concatenated bias array from gate dictionary.

    Args:
        bias_dict: dict with keys ['i', 'f', 'g', 'o'] containing bias arrays
        output_file: file handle to write to
        array_name: C array name
    """
    gate_order = ['i', 'f', 'g', 'o']
    arrays = []

    for gate in gate_order:
        b = bias_dict.get(gate)
        if b is not None:
            arrays.append(b.flatten())

    if not arrays:
        output_file.write(f"// {array_name} not found, using placeholder\n")
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


def export_onnx_weights(output_dir, weights):
    """Export C header files from ONNX weights dict"""
    scales = {}

    # Input scale fixed at 1/256 (symmetric quantization, zero_point=0)
    input_scale = 0.00390625

    # FC weights - supports FP32 and QDQ formats
    fc_weight = (
        weights.get("fc1.weight") or
        weights.get("fc.weight") or
        weights.get("fc.weight_quantized"))
    fc_bias = (
        weights.get("fc1.bias") or
        weights.get("fc.bias") or
        weights.get("fc.bias_quantized"))
    fc_weight_scale = weights.get("fc.weight_scale")

    if fc_weight is not None and fc_bias is not None:
        # Check if QDQ format (int8 weights)
        if fc_weight.dtype == np.int8:
            fc_weight_int8 = fc_weight
            # Read scale from model
            if fc_weight_scale is not None:
                fc_scale = float(fc_weight_scale.flat[0])
            else:
                fc_scale = 0.01
                info(f"FC using default weight scale: {fc_scale}")

            # bias is already quantized (int32), use directly
            if fc_bias.dtype == np.int32:
                fc_bias_int32 = fc_bias
            else:
                fc_bias_int32 = (
                fc_bias / (input_scale * fc_scale)).astype(np.int32)
        else:
            # FP32 format, needs quantization
            fc_weight_int8, fc_scale = quantize_to_int8(fc_weight)
            fc_bias_int32 = (
                fc_bias / (input_scale * fc_scale)).astype(np.int32)

        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write(
                "// FC layer weights and bias extracted from "
                "ONNX model (int8 quantized)\n")
            export_weights_to_c(fc_weight_int8, "fc_weights", f)
            export_bias_to_c(fc_bias_int32, "fc_bias", f)
        info(f"FC quantization complete: scale={fc_scale}")
        scales["fc_scale"] = fc_scale

    # CONV_2D weights - supports FP32 and QDQ formats
    conv_weight = (
        weights.get("conv1.weight") or
        weights.get("conv.weight") or
        weights.get("conv.weight_quantized"))
    conv_bias = (
        weights.get("conv1.bias") or
        weights.get("conv.bias") or
        weights.get("conv.bias_quantized"))
    conv_weight_scale = weights.get("conv.weight_scale")

    if conv_weight is not None and conv_bias is not None:
        # Check if QDQ format (int8 weights)
        if conv_weight.dtype == np.int8:
            conv_weight_int8 = conv_weight
            # Read scale from model
            if conv_weight_scale is not None:
                conv_scale = float(conv_weight_scale.flat[0])
            else:
                conv_scale = 0.01
                info(f"CONV_2D using default weight scale: {conv_scale}")

            # bias is already quantized (int32), use directly
            if conv_bias.dtype == np.int32:
                conv_bias_int32 = conv_bias
            else:
                conv_bias_int32 = (
                    conv_bias / (input_scale * conv_scale)).astype(np.int32)
        else:
            # FP32 format, needs quantization
            conv_weight_int8, conv_scale = quantize_to_int8(conv_weight)
            conv_bias_int32 = (
                conv_bias / (input_scale * conv_scale)).astype(np.int32)

        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write(
                "// CONV_2D layer weights and bias extracted from "
                "ONNX model (int8 quantized)\n")
            export_weights_to_c(conv_weight_int8, "conv_weights", f)
            export_bias_to_c(conv_bias_int32, "conv_bias", f)
        info(f"CONV_2D quantization complete: scale={conv_scale}")
        scales["conv_scale"] = conv_scale

    # SVDF weights
    svdf_weight = weights.get("weights") if "weights" in weights else None
    svdf_bias = weights.get("bias") if "bias" in weights else None

    if svdf_weight is not None and svdf_bias is not None:
        if svdf_weight.dtype == np.int8:
            svdf_weight_int8 = svdf_weight
            svdf_scale = 0.01
        else:
            svdf_weight_int8, svdf_scale = quantize_to_int8(svdf_weight)

        svdf_bias_int32 = (svdf_bias / (input_scale * svdf_scale)).astype(np.int32)

        with open(output_dir / 'svdf_weights.h', 'w') as f:
            f.write(
                "// SVDF layer weights and bias extracted from "
                "ONNX model (int8 quantized)\n")
            export_weights_to_c(svdf_weight_int8, "svdf_weights", f)
            export_bias_to_c(svdf_bias_int32, "svdf_bias", f)
        info(f"SVDF quantization complete: scale={svdf_scale}")
        scales["svdf_scale"] = svdf_scale

    # LSTM weights (if any)
    lstm_prefixes = ["lstm.", "lstm_"]
    lstm_weights = {}
    for name, tensor in weights.items():
        for prefix in lstm_prefixes:
            if name.startswith(prefix):
                lstm_weights[name] = tensor
                break
    if lstm_weights:
        # TODO: export LSTM weights
        pass

    return scales


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
