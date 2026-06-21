#!/usr/bin/env python3
"""
Extract FC and LSTM layer weights and biases from TFLite files
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.parser_litert import parse_model_tflite
from tinymlc.utils import fatal_error, warning, info


GATE_ORDER = ['i', 'f', 'g', 'o']
WEIGHTLESS_OPS = ["ADD", "SOFTMAX", "RESHAPE", "QUANTIZE"]
EXPORT_LINE_WIDTH = 16  # int8 array: 16 values per line
EXPORT_BIAS_LINE_WIDTH = 8  # int32 array: 8 values per line


def extract_fc_weights(interpreter, op_info):
    """Extract FULLY_CONNECTED layer weights and bias"""
    weights_idx = op_info.get("fc_weights_idx")
    bias_idx = op_info.get("fc_bias_idx")

    if weights_idx is None or bias_idx is None:
        fatal_error(
            "FC weights/bias indices not found: "
            f"weights={weights_idx}, bias={bias_idx}",
            "Check if tensor names were preserved during model conversion"
        )

    try:
        # Try using tensor() method
        weights_tensor = interpreter.tensor(weights_idx)
        bias_tensor = interpreter.tensor(bias_idx)
        weights = weights_tensor()
        bias = bias_tensor()
    except ValueError as e:
        # Fallback to get_tensor
        try:
            weights = interpreter.get_tensor(weights_idx)
            bias = interpreter.get_tensor(bias_idx)
        except ValueError as e2:
            fatal_error(
                f"Cannot get tensor: {e2}",
                "Ensure model is loaded correctly"
            )

    info(f"FC weights: shape={weights.shape}, dtype={weights.dtype}")
    info(f"FC bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def extract_lstm_weights(interpreter, op_info):
    """Extract LSTM gate weights and biases"""
    indices = op_info.get("lstm_weight_indices", {})
    input_indices = indices.get("input", [])
    recurrent_indices = indices.get("recurrent", [])
    bias_indices = indices.get("bias", [])

    if not input_indices or not recurrent_indices:
        fatal_error(
            "LSTM weight indices incomplete",
            "Check if model follows standard TFLite LSTM format"
        )

    gate_order = ['i', 'f', 'g', 'o']
    lstm_weights = {'input': {}, 'recurrent': {}, 'bias': {}}

    # Extract input weights
    for gate, idx in zip(gate_order, input_indices):
        try:
            lstm_weights['input'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            fatal_error(f"LSTM input_{gate} weight extraction failed: {e}")

    # Extract recurrent weights
    for gate, idx in zip(gate_order, recurrent_indices):
        try:
            lstm_weights['recurrent'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            fatal_error(f"LSTM recurrent_{gate} weight extraction failed: {e}")

    # Extract biases (may have fewer than 4)
    for gate, idx in zip(gate_order, bias_indices):
        try:
            lstm_weights['bias'][gate] = interpreter.get_tensor(idx)
        except ValueError as e:
            warning(
                f"LSTM bias_{gate} extraction failed, will use zero array",
                f"Index: {idx}"
            )
            lstm_weights['bias'][gate] = None

    # Add None placeholders for missing gates
    for gate in gate_order:
        if gate not in lstm_weights['input']:
            lstm_weights['input'][gate] = None
        if gate not in lstm_weights['recurrent']:
            lstm_weights['recurrent'][gate] = None
        if gate not in lstm_weights['bias']:
            lstm_weights['bias'][gate] = None

    return lstm_weights


def export_concatenated_weights(weights_list, output_file, array_name,
                                dtype='int8'):
    """Export concatenated weight array.

    Missing gates are skipped and padded with zero arrays.
    """
    arrays = []
    total_size = 0
    missing_gates = []

    for gate in GATE_ORDER:
        w = weights_list.get(gate)
        if w is None:
            missing_gates.append(gate)
            # Use zero array as placeholder (shape inferred from other gates)
            # Collect missing info first, handle later
            continue
        arrays.append(w.flatten())
        total_size += w.size

    # If missing gates exist, print warning and create
    # zero arrays based on existing shapes
    if missing_gates:
        warning(
            f"Warning: {missing_gates} gate weights missing, "
            "padding with zero arrays"
        )
        # Get shape from first non-None weight
        for gate in GATE_ORDER:
            w = weights_list.get(gate)
            if w is not None:
                shape = w.shape
                for mg in missing_gates:
                    zero_array = np.zeros(shape, dtype=np.int8)
                    arrays.append(zero_array.flatten())
                    total_size += zero_array.size
                break

    if not arrays:
        fatal_error(
            f"All {array_name} weights missing",
            f"Check model conversion completeness, or confirm "
            f"model contains {array_name} weights"
        )

    # Concatenate
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


def export_concatenated_bias(bias_list, output_file, array_name):
    """Export concatenated bias array"""
    gate_order = ['i', 'f', 'g', 'o']
    arrays = []

    for gate in gate_order:
        b = bias_list.get(gate)
        if b is not None:
            arrays.append(b.flatten())
        else:
            warning(f"{gate} gate bias missing")

    if not arrays:
        # All biases missing, generate placeholder
        warning(f"{array_name} all missing, using zero array placeholder")
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

    info(f"  Generated {array_name}[{total_size}]")


def extract_conv_weights(interpreter, op_info):
    """Extract CONV_2D layer weights and bias"""
    weights_idx = op_info.get("conv_weights_idx")
    bias_idx = op_info.get("conv_bias_idx")

    if weights_idx is None:
        fatal_error(
            "CONV_2D weight index not found",
            "Check if tensor names were preserved during model conversion"
        )

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = (
            interpreter.get_tensor(bias_idx)
            if bias_idx is not None else None
        )
    except ValueError as e:
        fatal_error(
            f"Cannot get CONV_2D tensor: {e}",
            "Ensure model is loaded correctly"
        )

    info(f"CONV_2D weights: shape={weights.shape}, dtype={weights.dtype}")
    if bias is not None:
        info(f"CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def export_weights_to_c(weights, name, output_file):
    """Export int8 weights"""
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
    """Export int32 bias"""
    if bias is None:
        output_file.write(f"// {name} not found, using placeholder\n")
        output_file.write(f"static const int32_t {name}[1] = {{0}};\n\n")
        return

    output_file.write(f"static const int32_t {name}[{bias.size}] = {{\n    ")
    for i, val in enumerate(bias):
        output_file.write(f"{int(val)}")
        if i < bias.size - 1:
            output_file.write(", ")
        if (i + 1) % 8 == 0:
            output_file.write("\n    ")
    output_file.write("\n};\n\n")


def extract_dw_weights(interpreter, op_info):
    """Extract DEPTHWISE_CONV_2D layer weights and bias"""
    weights_idx = op_info.get("dw_weights_idx")
    bias_idx = op_info.get("dw_bias_idx")

    if weights_idx is None:
        fatal_error(
            "DEPTHWISE_CONV_2D weight index not found",
            "Check if tensor names were preserved during model conversion"
        )

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = (
            interpreter.get_tensor(bias_idx)
            if bias_idx is not None else None
        )
    except ValueError as e:
        fatal_error(
            f"Cannot get DEPTHWISE_CONV_2D tensor: {e}",
            "Ensure model is loaded correctly"
        )

    info(
        f"DEPTHWISE_CONV_2D weights: "
        f"shape={weights.shape}, dtype={weights.dtype}"
    )
    if bias is not None:
        info(f"DEPTHWISE_CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def main():
    parser = argparse.ArgumentParser(
        description='Extract FC and LSTM weights from TFLite model')
    parser.add_argument('model', help='TFLite model file path')
    parser.add_argument('--output-dir', default='tinymlc_generated',
                        help='Output directory (default: tinymlc_generated)')
    args = parser.parse_args()

    # 1. Load and parse model
    info(f"Loading model: {args.model}")
    interpreter = LiteRTInterpreter(model_path=args.model)
    interpreter.allocate_tensors()

    model_info = parse_model_tflite(args.model)

    # 2. Find operator info
    fc_op_info = None
    lstm_op_info = None
    conv_op_info = None
    dw_op_info = None
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            fc_op_info = op
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            lstm_op_info = op
        elif op["op_name"] == "CONV_2D":
            conv_op_info = op
        elif op["op_name"] == "DEPTHWISE_CONV_2D":
            dw_op_info = op

    # 3. Extract weights
    info("\nExtracting weights...")
    fc_weights = fc_bias = None
    if fc_op_info:
        fc_weights, fc_bias = extract_fc_weights(interpreter, fc_op_info)
    lstm_weights = None
    if lstm_op_info:
        lstm_weights = extract_lstm_weights(interpreter, lstm_op_info)
    conv_weights = None
    conv_bias = None
    if conv_op_info:
        conv_weights, conv_bias = extract_conv_weights(interpreter,
                                                       conv_op_info)
    dw_weights = None
    dw_bias = None
    if dw_op_info:
        dw_weights, dw_bias = extract_dw_weights(interpreter, dw_op_info)

    # 4. Check if any weights were extracted
    has_fc = fc_weights is not None
    has_lstm = (
        lstm_weights is not None
        and any(v is not None for v in lstm_weights['input'].values())
    )
    has_conv = conv_weights is not None
    has_dw = dw_weights is not None

    if not (has_fc or has_lstm or has_conv or has_dw):
        # Check for weightless operators
        has_weightless_op = False
        for op in model_info["ops"]:
            if op["op_name"] in WEIGHTLESS_OPS:
                has_weightless_op = True
                break

        if not has_weightless_op:
            fatal_error(
                "No weights found",
                "Check if model contains supported operators"
            )
        else:
            info(
                "Note: Model only contains weightless operators "
                "(ADD/Softmax/Reshape), continuing..."
            )

    # 5. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Generate FC weights file
    if fc_weights is not None:
        output_path = output_dir / 'fc_weights.h'
        with open(output_path, 'w') as f:
            f.write("// Auto-extracted FC layer weights and bias from TFLite\n")
            f.write("// Do not modify manually\n\n")
            export_weights_to_c(fc_weights, "fc_weights", f)
            export_bias_to_c(fc_bias, "fc_bias", f)
        info(f"Generated: {output_path}")

    # 7. Generate LSTM weights file
    if (
        lstm_weights
        and any(v is not None for v in lstm_weights['input'].values())
    ):
        output_path = output_dir / 'lstm_weights.h'
        with open(output_path, 'w') as f:
            f.write(
                "// Auto-extracted LSTM gate weights and bias "
                "from TFLite (concatenated)\n"
            )
            f.write("// Order: i, f, g, o\n")
            f.write("// Do not modify manually\n\n")

            # Export concatenated input weights
            export_concatenated_weights(lstm_weights['input'], f,
                                        'lstm_input_weights', 'int8')
            # Export concatenated recurrent weights
            export_concatenated_weights(lstm_weights['recurrent'], f,
                                        'lstm_recurrent_weights', 'int8')
            # Export concatenated bias
            export_concatenated_bias(lstm_weights['bias'], f, 'lstm_bias')

        info(f"Generated: {output_path}")

    # 8. Generate CONV weights file conv_weights.h
    if conv_weights is not None:
        output_path = output_dir / 'conv_weights.h'
        with open(output_path, 'w') as f:
            f.write("// Auto-extracted CONV_2D weights and bias from TFLite\n")
            f.write("// Do not modify manually\n\n")
            export_weights_to_c(conv_weights, "conv_weights", f)
            if conv_bias is not None:
                export_bias_to_c(conv_bias, "conv_bias", f)
        info(f"Generated: {output_path}")

    # 9. Generate DEPTHWISE_CONV weights file
    if dw_weights is not None:
        output_path = output_dir / 'dw_weights.h'
        with open(output_path, 'w') as f:
            f.write(
                "// Auto-extracted DEPTHWISE_CONV_2D weights "
                "and bias from TFLite\n"
            )
            f.write("// Do not modify manually\n\n")
            export_weights_to_c(dw_weights, "dw_weights", f)
            if dw_bias is not None:
                export_bias_to_c(dw_bias, "dw_bias", f)
        info(f"Generated: {output_path}")

    # Print statistics
    info("\n=== Extraction Statistics ===")
    if fc_weights is not None:
        info(
            f"FC weights: {fc_weights.size} int8, bias: {fc_bias.size} int32")

    if lstm_weights:
        for gate in GATE_ORDER:
            w = lstm_weights['input'].get(gate)
            r = lstm_weights['recurrent'].get(gate)
            b = lstm_weights['bias'].get(gate)
            # Check if any non-None weights exist
            if w is not None or r is not None or b is not None:
                w_size = w.size if w is not None else 0
                r_size = r.size if r is not None else 0
                b_size = b.size if b is not None else 0
                info(
                    f"LSTM {gate} gate: input={w_size}, "
                    f"recurrent={r_size}, bias={b_size}"
                )

    if conv_weights is not None:
        info(
            f"CONV weights: {conv_weights.size} int8, "
            f"bias: {conv_bias.size} int32"
        )

    return 0


if __name__ == "__main__":
    main()
