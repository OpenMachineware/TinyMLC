# -*- coding: utf-8 -*-
# TinyMLC - Tiny Machine Learning Compiler
#
# Copyright (c) 2026 Jia Liu & TinyMLC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of TinyMLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import stat
import numpy as np
import shutil

from pathlib import Path
from jinja2 import Template
from typing import Dict, Any, List, Optional

from TinyMLC.ops import SUPPORTED_OPS
from utils.dump import fatal_error, info


# Fallback values, used when valid scale cannot be read from model
DEFAULT_SCALE = 0.01  # empirical value
DEFAULT_SHIFT = 8     # empirical value


def build_execution_order(ops, tensors):
    """Determine operator execution order based on tensor dependencies"""

    # Convert all indices to Python int
    for op in ops:
        # Skip if index is not yet assigned
        if "index" not in op or op["index"] is None:
            continue
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


def validate_ops(model_info: Dict[str, Any]) -> None:
    """Validate all operators and check for supported operators."""
    ops = model_info.get("ops", [])

    for op in ops:
        state = op.get("state")
        if state not in ("translated", "generated"):
            fatal_error(
                f"Operator {op['op_name']} state is {state}, "
                "cannot generate code",
                f"Pass flags: {op.get('pass_flags', {})}")

    has_supported = any(op.get("op_name") in SUPPORTED_OPS for op in ops)
    if not has_supported:
        fatal_error(
            "Model does not contain any supported operators",
            f"Supported operators: {', '.join(SUPPORTED_OPS)}")


def analyze_ops(
        model_info: Dict[str, Any],
        execution_order: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze operator types, LSTM params, FC/Conv quantization params.

    Returns:
        {
            "has_fc": bool,
            "has_conv": bool,
            "has_dw": bool,
            "has_svdf": bool,
            "lstm_params": dict,
            "fc_scale": float,
            "fc_output_scale": float,
            "fc_multiplier": int,
            "fc_shift": int,
            "conv_multiplier": int,
            "conv_shift": int,
        }
    """
    tensors = model_info.get("tensors", {})

    # ---- Detect operator types ----
    has_fc = False
    has_conv = False
    has_dw = False
    has_svdf = False
    lstm_params = None

    for op in model_info.get("ops", []):
        op_name = op.get("op_name")
        if op_name == "FULLY_CONNECTED":
            has_fc = True
        elif op_name == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_params = op.get("lstm_params")
        elif op_name == "SVDF":
            has_svdf = True
        elif op_name == "CONV_2D":
            has_conv = True
        elif op_name == "DEPTHWISE_CONV_2D":
            has_dw = True

    # ---- LSTM params ----
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
    else:
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
            shift = max(4, min(shift, 12))
            shifts.append(shift)

        lstm_params["shifts"] = shifts
        info(
            f"LSTM right shifts: i={shifts[0]}, f={shifts[1]}, "
            f"g={shifts[2]}, o={shifts[3]}")

    # ---- FC quantization ----
    fc_scale = None
    fc_output_scale = None
    fc_multiplier = None
    fc_shift = None

    for op in model_info.get("ops", []):
        if op.get("op_name") != "FULLY_CONNECTED":
            continue

        fc_scale = op.get("fc_scale")
        fc_output_scale = op.get("fc_output_scale")

        if fc_scale is None:
            quant_scales = model_info.get("quant_scales", {})
            fc_scale = quant_scales.get("fc_scale")

        if fc_scale is None:
            input_indices = op.get("input_indices", [])
            if len(input_indices) > 1:
                weight_idx = input_indices[1]
                if weight_idx in tensors:
                    fc_scale = tensors[weight_idx].get("scale")

        if fc_output_scale is None:
            output_indices = op.get("output_indices", [])
            if output_indices:
                output_idx = output_indices[0]
                if output_idx in tensors:
                    fc_output_scale = tensors[output_idx].get("scale")

        fc_input_scale = 0.00390625
        input_indices = op.get("input_indices", [])
        if input_indices:
            data_idx = input_indices[0]
            if data_idx in tensors:
                fc_input_scale = tensors[data_idx].get("scale", 0.00390625)

        if fc_scale is None:
            fc_scale = 0.01
            info(f"FC using default weight scale: {fc_scale}")

        if fc_output_scale is None:
            fc_output_scale = 0.00390625
            info(f"FC using default output scale: {fc_output_scale}")

        fc_multiplier, fc_shift = calculate_multiplier_shift_from_scale(
            fc_input_scale, fc_scale, fc_output_scale)

        info(
            f"FC quantization params: scale={fc_scale}, "
            f"output_scale={fc_output_scale}, "
            f"multiplier={fc_multiplier}, shift={fc_shift}")
        break

    if fc_multiplier is None:
        fc_multiplier, fc_shift = 213512, -30
        info("Using fallback FC quantization params")

    # ---- CONV quantization ----
    conv_multiplier = None
    conv_shift = None

    for op in model_info.get("ops", []):
        if op.get("op_name") == "CONV_2D":
            conv_scale = op.get("conv_scale")
            conv_output_scale = op.get("conv_output_scale")

            if conv_scale is None:
                quant_scales = model_info.get("quant_scales", {})
                conv_scale = quant_scales.get("conv_scale")

            if conv_scale is None:
                input_indices = op.get("input_indices", [])
                if len(input_indices) > 1:
                    weight_idx = input_indices[1]
                    if weight_idx in tensors:
                        conv_scale = tensors[weight_idx].get("scale")

            if conv_output_scale is None:
                output_indices = op.get("output_indices", [])
                if output_indices:
                    output_idx = output_indices[0]
                    if output_idx in tensors:
                        conv_output_scale = tensors[output_idx].get("scale")

            conv_input_scale = 0.00390625
            input_indices = op.get("input_indices", [])
            if input_indices:
                data_idx = input_indices[0]
                if data_idx in tensors:
                    conv_input_scale = tensors[data_idx].get("scale", 0.00390625)

            if conv_scale is None:
                conv_scale = 0.01
                info(f"CONV_2D using default weight scale: {conv_scale}")

            if conv_output_scale is None:
                conv_output_scale = 0.00390625
                info(f"CONV_2D using default output scale: {conv_output_scale}")

            conv_multiplier, conv_shift = calculate_multiplier_shift_from_scale(
                conv_input_scale, conv_scale, conv_output_scale)

            info(
                f"CONV_2D quantization params: scale={conv_scale}, "
                f"output_scale={conv_output_scale}, "
                f"multiplier={conv_multiplier}, shift={conv_shift}")
            break

    if conv_multiplier is None and has_dw:
        for op in model_info.get("ops", []):
            if op.get("op_name") == "DEPTHWISE_CONV_2D":
                dw_scale = op.get("dw_scale", 0.01)
                dw_output_scale = op.get("dw_output_scale", 0.00390625)
                conv_input_scale = 0.00390625
                conv_multiplier, conv_shift = (
                    calculate_multiplier_shift_from_scale(
                        conv_input_scale, dw_scale, dw_output_scale))
                info(
                    f"DEPTHWISE_CONV_2D quantization params: "
                    f"scale={dw_scale}, output_scale={dw_output_scale}, "
                    f"multiplier={conv_multiplier}, shift={conv_shift}")
                break

    if conv_multiplier is None:
        conv_multiplier, conv_shift = 0, 0

    return {
        "has_fc": has_fc,
        "has_conv": has_conv,
        "has_dw": has_dw,
        "has_svdf": has_svdf,
        "lstm_params": lstm_params,
        "fc_scale": fc_scale,
        "fc_output_scale": fc_output_scale,
        "fc_multiplier": fc_multiplier,
        "fc_shift": fc_shift,
        "conv_multiplier": conv_multiplier,
        "conv_shift": conv_shift,
    }


def build_context(
        model_info: Dict[str, Any],
        execution_order: List[Dict[str, Any]],
        op_analysis: Dict[str, Any],
        # stats: Dict[str, Any],  # Keep this code for convert might need it
) -> Dict[str, Any]:
    """
    Build the template context from model_info.

    This collects all data needed for the Jinja2 templates:
        - tensor sizes/shapes
        - reshape targets
        - FC parameters
        - tensor definitions
        - etc.
    """
    tensors = model_info.get("tensors", {})
    target = model_info.get("target", "riscv")
    inference_func = model_info.get("inference_func", "tinymlc_inference")

    has_fc = op_analysis["has_fc"]
    has_conv = op_analysis["has_conv"]
    has_dw = op_analysis["has_dw"]
    has_svdf = op_analysis["has_svdf"]
    lstm_params = op_analysis["lstm_params"]
    fc_multiplier = op_analysis["fc_multiplier"]
    fc_shift = op_analysis["fc_shift"]
    fc_scale = op_analysis["fc_scale"]
    fc_output_scale = op_analysis["fc_output_scale"]
    conv_multiplier = op_analysis["conv_multiplier"]
    conv_shift = op_analysis["conv_shift"]

    # ---- Tensor sizes and shapes ----
    tensor_sizes = {}
    tensor_shapes = {}
    for idx, spec in tensors.items():
        shape = spec.get("shape", [])
        size = 1
        for d in shape:
            size *= int(d)
        tensor_sizes[int(idx)] = size
        tensor_shapes[int(idx)] = [int(d) for d in shape]

    # ---- Build includes ----
    includes = []
    if has_fc:
        includes.append('#include "fc_weights.h"')
    if lstm_params["time_steps"] > 0:
        includes.append('#include "lstm_weights.h"')
    if has_conv:
        includes.append('#include "conv_weights.h"')
    if has_dw:
        includes.append('#include "dw_weights.h"')
    if has_svdf:
        includes.append('#include "svdf_weights.h"')

    # ---- Reshape targets ----
    reshape_targets = []
    for op in execution_order:
        if op.get("op_name") == "RESHAPE":
            target_shape = op.get("reshape_target_shape", [])
            if target_shape:
                reshape_targets.append(
                    "{" + ", ".join(str(int(s)) for s in target_shape) + "}")
            else:
                reshape_targets.append("{0}")

    # ---- FC params ----
    fc_params = {}
    for op in execution_order:
        if op.get("op_name") == "FULLY_CONNECTED":
            input_idx = op["input_indices"][0]
            output_idx = op["output_indices"][0]
            fc_params[op["index"]] = {
                "input_size": tensor_sizes.get(input_idx, 0),
                "output_size": tensor_sizes.get(output_idx, 0),
                "multiplier": fc_multiplier,
                "shift": fc_shift,
                "scale": fc_scale,
                "output_scale": fc_output_scale,
            }

    # ---- Copy conv_params from original ops ----
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

    # ---- Input sizes ----
    input_size_1 = 1
    input_size_2 = 1
    if len(model_info.get("input", [])) >= 1:
        for dim in model_info["input"][0]["shape"]:
            input_size_1 *= int(dim)
    if len(model_info.get("input", [])) >= 2:
        for dim in model_info["input"][1]["shape"]:
            input_size_2 *= int(dim)

    # ---- Input/output sizes ----
    input_size = 1
    if model_info.get("input"):
        for dim in model_info["input"][0]["shape"]:
            input_size *= int(dim)

    output_size = 1
    if model_info.get("output"):
        for dim in model_info["output"][0]["shape"]:
            output_size *= int(dim)

    # ---- Input tensor indices ----
    input_tensor_indices = []
    for inp in model_info.get("input", []):
        found = False
        for idx, spec in tensors.items():
            if spec.get("name") == inp.get("name"):
                input_tensor_indices.append(int(idx))
                found = True
                break
        if not found:
            input_tensor_indices.append(0)

    # ---- Tensors to define ----
    tensors_to_define = []
    defined_set = set(input_tensor_indices)

    for op in execution_order:
        for out_idx in op.get("output_indices", []):
            out_idx = int(out_idx)
            if out_idx in tensor_sizes and out_idx not in defined_set:
                tensors_to_define.append({
                    "index": out_idx,
                    "size": tensor_sizes[out_idx],
                    "type": "int8_t"
                })
                defined_set.add(out_idx)

        data_idx = op.get("data_input_idx")
        if data_idx is not None:
            data_idx = int(data_idx)
            if data_idx not in op.get("output_indices", []):
                if (data_idx in tensor_sizes and
                        data_idx not in defined_set and
                        data_idx not in input_tensor_indices):
                    tensors_to_define.append({
                        "index": data_idx,
                        "size": tensor_sizes[data_idx],
                        "type": "int8_t"
                    })
                    defined_set.add(data_idx)

        if op.get("op_name") == "SVDF":
            for key in ["svdf_weights_idx", "svdf_bias_idx"]:
                idx = op.get(key)
                if idx is not None:
                    idx = int(idx)
                    if idx not in defined_set and idx in tensor_sizes:
                        dtype = "int32_t" if key == "svdf_bias_idx" else "int8_t"
                        tensors_to_define.append({
                            "index": idx,
                            "size": tensor_sizes[idx],
                            "type": dtype
                        })
                        defined_set.add(idx)

        elif op.get("op_name") == "ADD":
            for key in ["add_input1_idx", "add_input2_idx"]:
                idx = op.get(key)
                if idx is not None:
                    idx = int(idx)
                    if idx not in defined_set and idx in tensor_sizes:
                        tensors_to_define.append({
                            "index": idx,
                            "size": tensor_sizes[idx],
                            "type": "int8_t"
                        })
                        defined_set.add(idx)

    # ---- Pool params defaults ----
    for op in execution_order:
        if op.get("op_name") in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            pool_params = op.get("pool_params", {})
            for key, default in [("pool_size_h", 2), ("pool_size_w", 2),
                                 ("stride_h", 2), ("stride_w", 2)]:
                if key not in pool_params or pool_params[key] is None:
                    pool_params[key] = default
            op["pool_params"] = pool_params

    # "stats": stats, # Keep this code for convert might need it
    return {
        "input_size": input_size,
        "output_size": output_size,
        "inference_func": inference_func,
        "includes": "\n".join(includes),
        "has_fc": has_fc,
        "has_lstm": lstm_params["time_steps"] > 0,
        "has_conv": has_conv,
        "has_dw": has_dw,
        "has_svdf": has_svdf,
        "target": target,
        "model_header": "model.h",
        "lstm_time_steps": lstm_params["time_steps"],
        "lstm_batch_size": lstm_params["batch_size"],
        "lstm_input_size": lstm_params["input_size"],
        "lstm_hidden_size": lstm_params["hidden_size"],
        "lstm_input_scale": lstm_params.get("input_scale", 0.00390625),
        "lstm_input_zp": lstm_params.get("input_zp", 0),
        "lstm_shifts": lstm_params.get("shifts", [8, 8, 8, 8]),
        "tensor_sizes": tensor_sizes,
        "tensor_shapes": tensor_shapes,
        "execution_order": execution_order,
        "last_output_tensor": execution_order[-1]["output_indices"][0],
        "reshape_targets": reshape_targets,
        "fc_params": fc_params,
        "inputs_count": len(model_info.get("input", [])),
        "INPUT_SIZE_1": input_size_1,
        "INPUT_SIZE_2": input_size_2,
        "fc_multiplier": fc_multiplier,
        "fc_shift": fc_shift,
        "conv_multiplier": conv_multiplier,
        "conv_shift": conv_shift,
        "input_tensor_indices": input_tensor_indices,
        "tensors_to_define": tensors_to_define,
    }


def render_code(
        context: Dict[str, Any],
        output_dir: Path,
        target: str,
        inference_func: str,
        with_test_main: bool,
        accel_lib_inc: Optional[str] = None,
        accel_lib_lib: Optional[str] = None,
) -> Dict[str, str]:
    """Render all templates and write files."""
    template_dir = Path(__file__).parent / "templates"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write feature flags ----
    with open(output_dir / "model_features.txt", "w") as f:
        if context.get("has_lstm"):
            f.write("HAS_LSTM\n")
        if context.get("has_fc"):
            f.write("HAS_FC\n")

    # ---- Render model.c ----
    with open(template_dir / "model.c.tpl", "r") as f:
        tmpl = Template(f.read())
    model_c = tmpl.render(**context)

    # ---- Render model.h ----
    with open(template_dir / "model.h.tpl", "r") as f:
        tmpl = Template(f.read())
    model_h = tmpl.render(**context)

    result = {
        "model.c": model_c,
        "model.h": model_h,
    }

    # ---- main_test.c ----
    if with_test_main:
        main_test_tpl = template_dir / "main_test.c.tpl"
        if main_test_tpl.exists():
            with open(main_test_tpl, "r") as f:
                tmpl = Template(f.read())
            result["main_test.c"] = tmpl.render(**context)
        else:
            # Fallback: copy from architecture-specific directory
            src_dir = Path(__file__).parent.parent / "ops" / target
            main_test_src = src_dir / "main_test.c"
            if main_test_src.exists():
                with open(main_test_src, "r") as f:
                    result["main_test.c"] = f.read()
            else:
                fatal_error(
                    f"main_test.c template not found: {main_test_tpl}",
                    f"Supported architectures: riscv, arm")

    # ---- Write stats to model_info for UI ----
    # Keep this code for convert might need it
    # stats = context.get("stats", {})
    # if stats:
    #     # The stats are already in model_info["quant_scales"] via compute_stats
    #     pass

    return result


def generate_c_code(
        model_info: Dict[str, Any],
        output_dir: str,
        target: str,
        inference_func: str = "tinymlc_inference",
        with_test_main: bool = False,
        accel_lib_inc: Optional[str] = None,
        accel_lib_lib: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate C code from model_info.

    This is the main entry point for code generation.
    """
    output_dir = Path(output_dir)

    # ---- Validate ----
    validate_ops(model_info)

    # ---- Build execution order ----
    execution_order = build_execution_order(
        model_info.get("ops", []),
        model_info.get("tensors", {})
    )

    # Log execution order
    info("Operator execution order:")
    for op in execution_order:
        info(f"  {op['index']}: {op['op_name']}")

    # ---- Analyze ops ----
    op_analysis = analyze_ops(model_info, execution_order)

    # ---- Compute stats ----
    # Keep this code for convert might need it
    # stats = {
    #     "macs": calculate_macs(model_info),
    #     "params": calculate_params(model_info),
    #     "peak_ram": calculate_peak_ram(model_info),
    #     "flash": calculate_flash(model_info),
    # }

    # ---- Store stats in model_info for UI ----
    # Keep this code for convert might need it
    # if "quant_scales" not in model_info:
    #     model_info["quant_scales"] = {}
    # model_info["quant_scales"]["macs"] = stats["macs"]
    # model_info["quant_scales"]["params"] = stats["params"]
    # model_info["quant_scales"]["peak_ram"] = stats["peak_ram"]
    # model_info["quant_scales"]["flash"] = stats["flash"]

    # ---- Build context ----
    context = build_context(
        model_info,
        execution_order,
        op_analysis,
        # stats,  # Keep this code for convert might need it
    )

    # ---- Render ----
    result = render_code(
        context,
        output_dir,
        target,
        inference_func,
        with_test_main,
        accel_lib_inc,
        accel_lib_lib,
    )

    # ---- Update state ----
    for op in model_info.get("ops", []):
        if op.get("state") == "translated":
            op["state"] = "generated"
            op["pass_flags"]["codegen"] = "success"

    return result


def copy_files_to_build(output_dir: Path, target: str, mode: str, accel: str,
                       accel_lib_inc=None, accel_lib_lib=None):
    """
    Copy all files needed for build to tinymlc_generated/

    Args:
        output_dir: output directory (tinymlc_generated)
        target: target architecture (riscv / arm / host)
        mode: build mode (debug / release)
        accel: acceleration library
    """
    # Determine source directory
    ops_root = Path(__file__).parent.parent / "ops"
    src_dir = ops_root / target

    if not src_dir.exists():
        fatal_error(
            f"Architecture directory not found: {src_dir}",
            f"Supported architectures: riscv, arm, host")

    # 1. Copy common header files
    include_src = ops_root / "include"
    if include_src.exists():
        shutil.copytree(include_src, output_dir / "include", dirs_exist_ok=True)

    # 2. Copy C operators (ops/c/*.c) to output_dir/c/
    c_src = ops_root / "c"
    if c_src.exists():
        shutil.copytree(c_src, output_dir / "c", dirs_exist_ok=True)

    # 3. Copy accelerator-specific operators (override ops/c/*.c)
    if accel == "cmsis-nn":
        accel_src = ops_root / target / "cmsis_nn"
        if accel_src.exists():
            for file in accel_src.glob("*.c"):
                shutil.copy2(file, output_dir / "c" / file.name)
    elif accel == "nmsis-nn":
        accel_src = ops_root / target / "nmsis_nn"
        if accel_src.exists():
            for file in accel_src.glob("*.c"):
                shutil.copy2(file, output_dir / "c" / file.name)

    # 4. Copy target architecture files
    # Host only needs .c files (no .S, .ld)
    if target == "host":
        # Create host directory in output
        host_src = ops_root / "host"
        if host_src.exists():
            shutil.copytree(host_src, output_dir / "host", dirs_exist_ok=True)
    else:
        # ARM/RISC-V need .c, .S, .ld files
        for file in src_dir.glob("*.c"):
            shutil.copy2(file, output_dir / file.name)
        for file in src_dir.glob("*.S"):
            shutil.copy2(file, output_dir / file.name)
        for file in src_dir.glob("*.ld"):
            shutil.copy2(file, output_dir / file.name)

    # 5. Copy corresponding build script
    if target == "host":
        # Host only has debug build script
        build_script = src_dir / "build_host_debug.sh"
    elif accel != 'none':
        accel_underscore = accel.replace("-", "_")
        build_script = src_dir / f"build_{target}_{accel_underscore}_{mode}.sh"
    else:
        build_script = src_dir / f"build_{target}_{mode}.sh"

    dest_build_script = output_dir / build_script.name

    # Check if .sh or .tpl exists
    tpl_script = src_dir / f"{build_script.name}.tpl"
    if build_script.exists():
        # Use .sh as source
        source_script = build_script
        use_template = tpl_script.exists() and accel_lib_inc and accel_lib_lib
    elif tpl_script.exists():
        # Use .tpl as source (no .sh file)
        source_script = tpl_script
        use_template = True
        dest_build_script = output_dir / build_script.name  # Output still .sh
    else:
        fatal_error(
            f"Build script not found: {build_script}",
            suggestion=f"Please check if accelerator type {accel} "
                      "is supported")

    if use_template:
        # Render template with accel library paths
        with open(source_script, 'r') as f:
            tmpl = Template(f.read())
        rendered = tmpl.render(
            accel_lib_inc=accel_lib_inc,
            accel_lib_lib=accel_lib_lib
        )
        with open(dest_build_script, 'w') as f:
            f.write(rendered)
    else:
        # Just copy the script
        shutil.copy2(source_script, dest_build_script)

    try:
        current_mode = dest_build_script.stat().st_mode
        dest_build_script.chmod(
            current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass
