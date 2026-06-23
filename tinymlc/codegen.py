import stat
import numpy as np
import shutil

from pathlib import Path
from jinja2 import Template

from utils.dump import fatal_error, warning, info


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

    # Calculate input/output sizes (size of first input/output tensor)
    # Note: For multi-input models, INPUT_SIZE is the size of one input tensor,
    # not the product of all inputs
    input_size = 1
    if model_info['input']:
        for dim in model_info['input'][0]['shape']:
            input_size *= int(dim)

    output_size = 1
    if model_info['output']:
        for dim in model_info['output'][0]['shape']:
            output_size *= int(dim)

    # Detect operator types in model
    has_fc = False
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
            lstm_params = op.get("lstm_params")
        elif op_name == "SVDF":
            has_svdf = True
        elif op_name == "CONV_2D":
            has_conv = True
        elif op_name == "DEPTHWISE_CONV_2D":
            has_dw = True

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
            shift = max(4, min(shift,
                               12))  # empirical range based on common models
            shifts.append(shift)

        lstm_params["shifts"] = shifts
        info(
            f"LSTM right shifts: i={shifts[0]}, f={shifts[1]}, "
            f"g={shifts[2]}, o={shifts[3]}")

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

    # If CONV_2D params not found but has DEPTHWISE_CONV_2D, calculate from DW params
    if (conv_multiplier is None or conv_shift is None) and has_dw:
        for op in model_info.get("ops", []):
            if op.get("op_name") == "DEPTHWISE_CONV_2D":
                dw_scale = op.get("dw_scale", 0.01)
                dw_output_scale = op.get("dw_output_scale", 0.00390625)
                conv_input_scale = 0.00390625
                conv_multiplier, conv_shift = calculate_multiplier_shift_from_scale(
                    conv_input_scale, dw_scale, dw_output_scale
                )
                info(
                    f"DEPTHWISE_CONV_2D quantization params: scale={dw_scale}, "
                    f"output_scale={dw_output_scale}, "
                    f"multiplier={conv_multiplier}, shift={conv_shift}")
                break

    # If CONV_2D params not found, use fallback
    if conv_multiplier is None or conv_shift is None:
        conv_multiplier, conv_shift = 0, 0

    # Build include list
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

    # Calculate individual input sizes for multi-input models
    input_size_1 = 1
    input_size_2 = 1
    if len(model_info["input"]) >= 1:
        for dim in model_info["input"][0]["shape"]:
            input_size_1 *= int(dim)
    if len(model_info["input"]) >= 2:
        for dim in model_info["input"][1]["shape"]:
            input_size_2 *= int(dim)

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
        "has_lstm": lstm_params["time_steps"] > 0,
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
        if lstm_params["time_steps"] > 0:
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

    # Optional: generate architecture-specific test main from template
    if with_test_main:
        main_test_tpl = template_dir / "main_test.c.tpl"
        if main_test_tpl.exists():
            with open(main_test_tpl, 'r') as f:
                tmpl = Template(f.read())
            result['main_test.c'] = tmpl.render(**context)
        else:
            # Fallback: copy from architecture-specific directory
            src_dir = Path(__file__).parent.parent / "ops" / target
            main_test_src = src_dir / "main_test.c"
            if main_test_src.exists():
                with open(main_test_src, 'r') as f:
                    result['main_test.c'] = f.read()
            else:
                fatal_error(
                    f"main_test.c template not found: {main_test_tpl}",
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
