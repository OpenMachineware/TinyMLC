#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
Convert TFLite/ONNX models to C code executable on MCU
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
                 "SOFTMAX", "RESHAPE", "QUANTIZE", "SVDF", "CONV_2D",
                 "MULTIPLY", "SIGMOID", "CONCAT", "SUB", "TANH"]
# Fallback values, used when valid scale cannot be read from model
DEFAULT_SCALE = 0.01  # empirical value
DEFAULT_SHIFT = 8     # empirical value


def build_execution_order(ops, tensors):
    """Determine operator execution order based on tensor dependencies"""

    # Convert all indices to Python int
    for op in ops:
        op["index"] = int(op["index"])
        if "input_indices" in op:
            op["input_indices"] = [int(i) for i in op["input_indices"]]
        if "output_indices" in op:
            op["output_indices"] = [int(i) for i in op["output_indices"]]

    # 1. Build tensor -> producer operator mapping
    tensor_producer = {}
    for op in ops:
        for out_idx in op.get("output_indices", []):
            tensor_producer[int(out_idx)] = op

    # 2. Build operator dependency relationships
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

    # 3. Calculate in-degree (how many operators current op depends on)
    in_degree = {}
    for op in ops:
        op_idx = int(op["index"])
        in_degree[op_idx] = len(op_deps.get(op_idx, []))

    # 4. Topological sort (Kahn's algorithm)
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
        fatal_error(
            "Model has cyclic dependencies, cannot determine execution order",
            "Please check if model structure is valid")

    return order


def generate_c_code(model_info, output_dir, target,
                    inference_func="tinymlc_inference",
                    with_test_main=False):
    ops = model_info.get("ops", [])
    tensors = model_info.get("tensors", {})

    execution_order = build_execution_order(ops, tensors)
    info("Operator execution order:")
    for op in execution_order:
        info(f"  {op['index']}: {op['op_name']}")

    # Check all operators before code generation
    for op in ops:
        if op["state"] != "translated" and op["state"] != "generated":
            fatal_error(
                f"Operator {op['op_name']} state is "
                f"{op.get('state')}, cannot generate code",
                f"Pass flags: {op.get('pass_flags', {})}")

    # Check if there are supported operators
    has_supported_op = False
    for op in model_info["ops"]:
        if op["op_name"] in SUPPORTED_OPS:
            has_supported_op = True
            break

    if not has_supported_op:
        fatal_error("Model does not contain any supported operators",
                    f"Supported operators: {', '.join(SUPPORTED_OPS)}")

    """Generate C code and header files"""
    template_dir = Path(__file__).parent / 'templates'

    # Calculate input/output sizes
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

    # Detect operator types in model
    has_fc = False
    has_lstm = False
    has_conv = False
    has_dw = False
    has_svdf = False

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
        elif op_name == "SVDF":
            has_svdf = True
        elif op_name == "CONV_2D":
            has_conv = True
        elif op_name == "DEPTHWISE_CONV_2D":
            has_dw = True

    if has_lstm and lstm_params is None:
        fatal_error("Model contains LSTM operator but no parameters extracted",
                    "Please check if model is standard TFLite LSTM format")
    elif has_lstm and lstm_params is not None:
        # Calculate right shift for LSTM
        input_scales = lstm_params.get(
            "input_scales",
            [DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE])
        recurrent_scales = lstm_params.get(
            "recurrent_scales",
            [DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE])

        shifts = []
        for in_s, rec_s in zip(input_scales, recurrent_scales):
            gate_scale = in_s * rec_s
            if gate_scale > 0:
                shift = int(np.log2(1.0 / gate_scale))
            else:
                shift = DEFAULT_SHIFT
            # Limit shift range to prevent LUT index overflow
            shift = max(4, min(shift, 12))  # empirical range based on common models
            shifts.append(shift)

        lstm_params["shifts"] = shifts
        info(
            f"LSTM right shifts: i={shifts[0]}, f={shifts[1]}, "
            f"g={shifts[2]}, o={shifts[3]}")
    else:
        pass

    # Process FC quantization parameters
    fc_scale = None
    fc_output_scale = None
    fc_multiplier = None
    fc_shift = None

    # Read FC parameters from model info
    for op in model_info.get("ops", []):
        if op.get("op_name") == "FULLY_CONNECTED":
            # Read from operator attributes
            fc_scale = op.get("fc_scale")
            fc_output_scale = op.get("fc_output_scale")

            # If fc_scale not in operator, try quant_scales (ONNX model)
            if fc_scale is None:
                quant_scales = model_info.get("quant_scales", {})
                fc_scale = quant_scales.get("fc_scale")

            # If fc_scale not in operator, try input tensor
            if fc_scale is None:
                input_indices = op.get("input_indices", [])
                if len(input_indices) > 1:
                    weight_idx = input_indices[1]  # weight tensor
                    if weight_idx in tensors:
                        fc_scale = tensors[weight_idx].get("scale")

            # If fc_output_scale not in operator, try output tensor
            if fc_output_scale is None:
                output_indices = op.get("output_indices", [])
                if output_indices:
                    output_idx = output_indices[0]
                    if output_idx in tensors:
                        fc_output_scale = tensors[output_idx].get("scale")

            # Get input tensor scale
            fc_input_scale = 0.00390625
            input_indices = op.get("input_indices", [])
            if len(input_indices) > 0:
                data_input_idx = input_indices[0]
                if data_input_idx in tensors:
                    fc_input_scale = tensors[data_input_idx].get(
                        "scale", 0.00390625)

            # If still not found, use default values
            if fc_scale is None:
                fc_scale = 0.01
                info(f"FC using default weight scale: {fc_scale}")
            if fc_output_scale is None:
                fc_output_scale = 0.00390625
                info(f"FC using default output scale: {fc_output_scale}")

            # Calculate multiplier and shift
            fc_multiplier, fc_shift = calculate_multiplier_shift_from_scale(
                fc_input_scale, fc_scale, fc_output_scale)

            info(
                f"FC quantization params: scale={fc_scale}, "
                f"output_scale={fc_output_scale}, "
                f"multiplier={fc_multiplier}, shift={fc_shift}")
            break  # Only process first FC

    # If FC params not found, use fallback
    if fc_multiplier is None or fc_shift is None:
        fc_multiplier, fc_shift = 213512, -30
        info("Using fallback FC quantization params")

    # Process CONV_2D quantization parameters
    conv_scale = None
    conv_output_scale = None
    conv_multiplier = None
    conv_shift = None

    for op in model_info.get("ops", []):
        if op.get("op_name") == "CONV_2D":
            # Read from operator attributes
            conv_scale = op.get("conv_scale")
            conv_output_scale = op.get("conv_output_scale")

            # If conv_scale not in operator, try quant_scales (ONNX model)
            if conv_scale is None:
                quant_scales = model_info.get("quant_scales", {})
                conv_scale = quant_scales.get("conv_scale")

            # If conv_scale not in operator, try input tensor
            if conv_scale is None:
                input_indices = op.get("input_indices", [])
                if len(input_indices) > 1:
                    weight_idx = input_indices[1]
                    if weight_idx in tensors:
                        conv_scale = tensors[weight_idx].get("scale")

            # If conv_output_scale not in operator, try output tensor
            if conv_output_scale is None:
                output_indices = op.get("output_indices", [])
                if output_indices:
                    output_idx = output_indices[0]
                    if output_idx in tensors:
                        conv_output_scale = tensors[output_idx].get("scale")

            # Get input tensor scale
            conv_input_scale = 0.00390625
            input_indices = op.get("input_indices", [])
            if len(input_indices) > 0:
                data_input_idx = input_indices[0]
                if data_input_idx in tensors:
                    conv_input_scale = tensors[data_input_idx].get(
                        "scale", 0.00390625)

            # Use default values
            if conv_scale is None:
                conv_scale = 0.01
                info(f"CONV_2D using default weight scale: {conv_scale}")
            if conv_output_scale is None:
                conv_output_scale = 0.00390625
                info(f"CONV_2D using default output scale: {conv_output_scale}")

            # Calculate multiplier and shift
            conv_multiplier, conv_shift = calculate_multiplier_shift_from_scale(
                conv_input_scale, conv_scale, conv_output_scale
            )

            info(
                f"CONV_2D quantization params: scale={conv_scale}, "
                f"output_scale={conv_output_scale}, "
                f"multiplier={conv_multiplier}, shift={conv_shift}")
            break

    # If CONV_2D params not found, use fallback
    if conv_multiplier is None or conv_shift is None:
        conv_multiplier, conv_shift = 0, 0

    # Build include list
    includes = []
    if has_fc:
        includes.append('#include "fc_weights.h"')
    if has_lstm:
        includes.append('#include "lstm_weights.h"')
    if has_conv:
        includes.append('#include "conv_weights.h"')
    if has_dw:
        includes.append('#include "dw_weights.h"')
    if has_svdf:
        includes.append('#include "svdf_weights.h"')

    tensor_sizes = {}
    tensor_shapes = {}
    for tensor_idx, tensor_info in tensors.items():
        size = 1
        shape = tensor_info.get("shape", [])
        for dim in shape:
            size *= int(dim)
        tensor_sizes[int(tensor_idx)] = size
        tensor_shapes[int(tensor_idx)] = [int(dim) for dim in shape]

    # Extract target shapes for all Reshape operators
    reshape_targets = []
    for op in execution_order:
        if op.get("op_name") == "RESHAPE":
            target_shape = op.get("reshape_target_shape", [])
            if target_shape:
                reshape_targets.append(
                    "{" + ", ".join(str(int(s)) for s in target_shape) + "}"
                )
            else:
                reshape_targets.append("{0}")

    fc_params = {}
    for op in execution_order:
        if op.get("op_name") == "FULLY_CONNECTED":
            # Get input tensor size
            input_idx = op["input_indices"][0]  # FC's first input is data
            input_size_fc = tensor_sizes.get(input_idx, 0)
            # Output size
            output_idx = op["output_indices"][0]
            output_size_fc = tensor_sizes.get(output_idx, 0)
            fc_params[op["index"]] = {
                "input_size": input_size_fc,
                "output_size": output_size_fc,
                # Add quantization params
                "multiplier": fc_multiplier,
                "shift": fc_shift,
                "scale": fc_scale,
                "output_scale": fc_output_scale,
            }

    for op in execution_order:
        if op.get("op_name") == "CONV_2D":
            for orig_op in model_info.get("ops", []):
                if orig_op.get("index") == op["index"]:
                    op["conv_params"] = orig_op.get("conv_params", {})
                    break
        elif op.get("op_name") == "SVDF":
            for orig_op in model_info.get("ops", []):
                if orig_op.get("index") == op["index"]:
                    op["svdf_params"] = orig_op.get("svdf_params", {})
                    break

    # Calculate input size
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
        fatal_error(f"Unsupported {len(model_info['input'])} inputs in model",
                    "Currently supports 1 or 2 inputs")

    # Ensure lstm_params has default value, not None (even without LSTM)
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

    # Collect input tensor indices (for input mapping in template)
    input_tensor_indices = []
    for inp in model_info["input"]:
        for tensor_idx, tensor_info in tensors.items():
            if tensor_info.get("name") == inp.get("name"):
                input_tensor_indices.append(int(tensor_idx))
                break
        else:
            input_tensor_indices.append(0)

    # Collect all intermediate tensors to define (avoid duplicate definitions)
    tensors_to_define = []
    defined_set = set(input_tensor_indices)
    
    for op in execution_order:
        for out_idx in op["output_indices"]:
            out_idx = int(out_idx)
            if out_idx in tensor_sizes and out_idx not in defined_set:
                tensors_to_define.append({
                    "index": out_idx,
                    "size": tensor_sizes[out_idx],
                    "type": "int8_t"
                })
                defined_set.add(out_idx)
        
        if "data_input_idx" in op and op["data_input_idx"] is not None:
            data_idx = int(op["data_input_idx"])
            if data_idx not in op["output_indices"]:
                if (data_idx in tensor_sizes and
                        data_idx not in defined_set and
                        data_idx not in input_tensor_indices):
                    tensors_to_define.append({
                        "index": data_idx,
                        "size": tensor_sizes[data_idx],
                        "type": "int8_t"
                    })
                    defined_set.add(data_idx)
        
        if op["op_name"] == "SVDF":
            if "svdf_weights_idx" in op and op["svdf_weights_idx"] is not None:
                idx = int(op["svdf_weights_idx"])
                if idx not in defined_set and idx in tensor_sizes:
                    tensors_to_define.append({
                        "index": idx,
                        "size": tensor_sizes[idx],
                        "type": "int8_t"
                    })
                    defined_set.add(idx)
            if "svdf_bias_idx" in op and op["svdf_bias_idx"] is not None:
                idx = int(op["svdf_bias_idx"])
                if idx not in defined_set and idx in tensor_sizes:
                    tensors_to_define.append({
                        "index": idx,
                        "size": tensor_sizes[idx],
                        "type": "int32_t"
                    })
                    defined_set.add(idx)
        elif op["op_name"] == "ADD":
            for idx_name in ["add_input1_idx", "add_input2_idx"]:
                if idx_name in op and op[idx_name] is not None:
                    idx = int(op[idx_name])
                    if idx not in defined_set and idx in tensor_sizes:
                        tensors_to_define.append({
                            "index": idx,
                            "size": tensor_sizes[idx],
                            "type": "int8_t"
                        })
                        defined_set.add(idx)

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
        "model_header": "model.h",  # fixed name for main_test.c include
        "lstm_time_steps": lstm_params["time_steps"],
        "lstm_batch_size": lstm_params["batch_size"],
        "lstm_input_size": lstm_params["input_size"],
        "lstm_hidden_size": lstm_params["hidden_size"],
        "lstm_input_scale": lstm_params.get("input_scale", 0.00390625),  # 1/256
        "lstm_input_zp": lstm_params.get("input_zp", 0),
        "lstm_shifts": lstm_params.get("shifts", [8, 8, 8, 8]),  # default 8
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
        "conv_multiplier": conv_multiplier,
        "conv_shift": conv_shift,
        "input_tensor_indices": input_tensor_indices,
        "tensors_to_define": tensors_to_define,
    }

    # Generate files first, decide whether to compile LSTM operator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_features.txt", "w") as f:
        if has_lstm:
            f.write("HAS_LSTM\n")
        if has_fc:
            f.write("HAS_FC\n")

    # Generate model.c
    with open(template_dir / 'model.c.tpl', 'r') as f:
        tmpl = Template(f.read())
    model_c = tmpl.render(**context)

    # Generate model.h
    with open(template_dir / 'model.h.tpl', 'r') as f:
        tmpl = Template(f.read())
    model_h = tmpl.render(**context)

    result = {
        'model.c': model_c,
        'model.h': model_h,
    }

    # Optional: copy architecture-specific test main
    if with_test_main:
        # Copy main_test.c from architecture-specific directory
        src_dir = Path(__file__).parent.parent / "ops" / target
        main_test_src = src_dir / "main_test.c"
        if main_test_src.exists():
            with open(main_test_src, 'r') as f:
                result['main_test.c'] = f.read()
        else:
            fatal_error(
                f"Architecture-specific main_test.c not found: "
                f"{main_test_src}",
                f"Supported architectures: riscv, arm")

    # Update state after code generation
    for op in ops:
        if op["state"] == "translated":
            op["state"] = "generated"
            op["pass_flags"]["codegen"] = "success"

    return result


def copy_files_to_build(output_dir: Path, target: str, mode: str, accel: str):
    """
    Copy all files needed for build to tinymlc_generated/

    Args:
        output_dir: output directory (tinymlc_generated)
        target: target architecture (riscv / arm)
        mode: build mode (debug / release)
        accel: acceleration library
    """
    # Determine source directory
    ops_root = Path(__file__).parent.parent / "ops"
    src_dir = ops_root / target

    if not src_dir.exists():
        fatal_error(
            f"Architecture directory not found: {src_dir}",
            f"Supported architectures: riscv, arm")

    # 1. Copy common header files
    include_src = ops_root / "include"
    if include_src.exists():
        shutil.copytree(include_src, output_dir / "include", dirs_exist_ok=True)

    # 2. Copy C operators (ops/c/*.c) to output_dir/c/
    c_src = ops_root / "c"
    if c_src.exists():
        shutil.copytree(c_src, output_dir / "c", dirs_exist_ok=True)

    # 3. Copy target architecture .c and .S files
    for file in src_dir.glob("*.c"):
        shutil.copy2(file, output_dir / file.name)
    for file in src_dir.glob("*.S"):
        shutil.copy2(file, output_dir / file.name)
    for file in src_dir.glob("*.ld"):
        shutil.copy2(file, output_dir / file.name)

    # 4. Copy corresponding build script
    if accel != 'none':
        accel_underscore = accel.replace("-", "_")
        build_script = src_dir / f"build_{target}_{accel_underscore}_{mode}.sh"
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
        fatal_error(
            f"Build script not found: {build_script}",
            suggestion=f"Please check if accelerator type {accel} "
                      "is supported")

    # 5. Copy LSTM related files (if any)
    lstm_src = ops_root / "lstm"
    if lstm_src.exists():
        shutil.copytree(lstm_src, output_dir / "lstm", dirs_exist_ok=True)


def calculate_multiplier_shift(input_scale, weight_scale, output_scale):
    """
    Calculate multiplier and shift for int8 quantization

    Quantization formula: output = round((acc * multiplier) >> (31 + shift))
    where acc = sum(input * weight) + bias

    Q31 fixed-point format:
    - 1 << 31 = 2147483648, represents max value in Q31 format
    - multiplier stored in 32-bit signed int, range -2147483648 ~ 2147483647
    - shift adjusts effective_scale * 2^31 to valid range

    Args:
        input_scale: input tensor quantization scale
        weight_scale: weight quantization scale
        output_scale: output tensor quantization scale

    Returns:
        multiplier: Q31 fixed-point scale factor
        shift: right shift adjustment bits
    """
    effective_scale = (input_scale * weight_scale) / output_scale

    if effective_scale == 0:
        return 0, 0

    # Q31 format: multiplier = effective_scale * 2^31
    mult = effective_scale * (1 << 31)
    shift = 0

    # multiplier exceeds int32 range: decrease shift (increase actual scale)
    while mult > 2147483647:
        shift -= 1
        mult /= 2

    # multiplier too small for precision: increase shift (decrease actual scale)
    while mult < 0.5:
        shift += 1
        mult *= 2

    multiplier = int(round(mult))
    multiplier = max(0, min(multiplier, 2147483647))

    return multiplier, shift


def calculate_multiplier_shift_from_scale(input_scale, weight_scale,
                                     output_scale):
    """Calculate multiplier and shift from scales"""
    return calculate_multiplier_shift(input_scale, weight_scale, output_scale)


def extract_all_weights_tflite(interpreter, model_info):
    """Extract all weights from TFLite model"""
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

    fc_weights, fc_bias = (
        extract_fc_weights(interpreter, fc_op_info)
        if fc_op_info else (None, None))
    lstm_weights = (
        extract_lstm_weights(interpreter, lstm_op_info)
        if lstm_op_info else None)
    conv_weights, conv_bias = (
        extract_conv_weights(interpreter, conv_op_info)
        if conv_op_info else (None, None))

    return fc_weights, fc_bias, lstm_weights, conv_weights, conv_bias


def generate_weight_headers(output_dir, fc_weights, fc_bias,
                             lstm_weights, conv_weights, conv_bias):
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
    # 1. Choose parsing path based on model format
    # ==========================================
    if model_path.endswith(".tflite"):
        model_info = parse_model_tflite(model_path)
        interpreter = LiteRTInterpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Extract weights
        fc_weights, fc_bias, lstm_weights, conv_weights, conv_bias = (
            extract_all_weights_tflite(interpreter, model_info))
        generate_weight_headers(
            output_dir, fc_weights, fc_bias, lstm_weights,
            conv_weights, conv_bias)
    elif model_path.endswith(".onnx"):
        model_info = parse_model_onnx(model_path)
        scales = export_onnx_weights(output_dir, model_info.get("weights", {}))
        # Save quantization scales to model_info for generate_c_code
        model_info["quant_scales"] = scales
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
