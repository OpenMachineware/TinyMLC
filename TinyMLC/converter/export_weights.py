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


import numpy as np

from utils.dump import warning, info


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
        weights_dict: dict with keys ['i', 'f', 'g', 'o']
            containing weight arrays
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


def export_model_weights(output_dir, model_info):
    """Unified weight export function for ONNX, TFLite and ANG models.

    Weights are identified by source-specific keys:
    - TFLite: "fc_tflite.weight", "fc_tflite.bias",
      "lstm_tflite.weight_ih", etc.
    - ONNX: "fc_onnx.weight", "fc_onnx.bias", "conv_onnx.weight", etc.
    - ANG: tensor index as key in weights dict

    Returns quant_scales dict.
    """
    weights = model_info.get("weights", {})
    ops = model_info.get("ops", [])
    quant_scales = {}

    # Input scale fixed at 1/256 (symmetric quantization, zero_point=0)
    input_scale = 0.00390625

    # For ANG models, weights are stored by tensor index
    # We need to extract them from ops
    def get_weight_by_idx(weights, idx):
        """Get weight from weights dict by tensor index"""
        if idx is None:
            return None
        # ANG format: weights[str(idx)] = numpy array or list
        if str(idx) in weights:
            w = weights[str(idx)]
            # Convert list to numpy array
            if isinstance(w, list):
                return np.array(w)
            return w
        # Also try int key
        if idx in weights:
            w = weights[idx]
            if isinstance(w, list):
                return np.array(w)
            return w
        return None

    # Collect FC weights from ops
    fc_weights = []
    for op in ops:
        if op.get("op_name") == "FULLY_CONNECTED":
            input_indices = op.get("input_indices", [])
            if len(input_indices) >= 3:
                weight_idx = input_indices[1]
                bias_idx = input_indices[2]
                fc_w = get_weight_by_idx(weights, weight_idx)
                fc_b = get_weight_by_idx(weights, bias_idx)
                if fc_w is not None:
                    fc_weights.append((fc_w, fc_b))

    # Export FC weights
    if fc_weights:
        all_fc_weights = []
        all_fc_bias = []
        for fc_w, fc_b in fc_weights:
            all_fc_weights.append(fc_w.flatten())
            if fc_b is not None:
                all_fc_bias.append(fc_b.flatten())
            else:
                all_fc_bias.append(np.zeros(fc_w.shape[-1], dtype=np.int32))

        fc_weights_concat = (np.concatenate(all_fc_weights)
                            if len(all_fc_weights) > 1 else all_fc_weights[0])
        fc_bias_concat = (np.concatenate(all_fc_bias)
                         if len(all_fc_bias) > 1 else all_fc_bias[0])

        fc_scale = 0.01
        fc_weight_int8 = (fc_weights_concat
                         if fc_weights_concat.dtype == np.int8
                         else quantize_to_int8(fc_weights_concat)[0])
        fc_bias_int32 = (fc_bias_concat.astype(np.int32)
                        if fc_bias_concat.dtype != np.int32
                        else fc_bias_concat)

        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// FC layer weights and bias extracted from ANG model\n\n")
            export_weights_to_c(fc_weight_int8, "fc_weights", f)
            export_bias_to_c(fc_bias_int32, "fc_bias", f)
        info(f"Generated: {output_dir}/fc_weights.h")
        quant_scales["fc_scale"] = fc_scale

    # Collect CONV weights from ops
    conv_weights_list = []
    conv_bias_list = []
    for op in ops:
        if op.get("op_name") == "CONV_2D":
            input_indices = op.get("input_indices", [])
            if len(input_indices) >= 2:
                weight_idx = input_indices[1]
                bias_idx = input_indices[2] if len(input_indices) >= 3 else None
                conv_w = get_weight_by_idx(weights, weight_idx)
                conv_b = (get_weight_by_idx(weights, bias_idx)
                          if bias_idx else None)
                if conv_w is not None:
                    conv_w = np.array(conv_w)
                    conv_weights_list.append(conv_w)
                    if conv_b is not None:
                        conv_b = np.array(conv_b)
                    conv_bias_list.append(
                    conv_b if conv_b is not None
                    else np.zeros(conv_w.shape[-1], dtype=np.int32))

    # Export CONV weights
    if conv_weights_list:
        conv_weights_concat = np.concatenate(
            [np.array(w).flatten() for w in conv_weights_list])
        conv_bias_concat = np.concatenate(
            [np.array(b).flatten() for b in conv_bias_list])

        conv_weight_int8 = (conv_weights_concat
                           if conv_weights_concat.dtype == np.int8
                           else quantize_to_int8(conv_weights_concat)[0])
        conv_bias_int32 = (conv_bias_concat.astype(np.int32)
                          if conv_bias_concat.dtype != np.int32
                          else conv_bias_concat)

        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write("// CONV_2D weights and bias extracted from ANG model\n\n")
            export_weights_to_c(conv_weight_int8, "conv_weights", f)
            export_bias_to_c(conv_bias_int32, "conv_bias", f)
        info(f"Generated: {output_dir}/conv_weights.h")

    # Try TFLite/ONNX format as fallback
    fc_weight = weights.get("fc_tflite.weight") or weights.get("fc_onnx.weight")
    fc_bias = weights.get("fc_tflite.bias") or weights.get("fc_onnx.bias")
    if fc_weight is not None and fc_bias is not None:
        fc_scale = 0.01
        if fc_weight.dtype == np.int8:
            fc_weight_int8 = fc_weight
            fc_bias_int32 = (fc_bias.astype(np.int32)
                            if fc_bias.dtype != np.int32 else fc_bias)
        else:
            fc_weight_int8, fc_scale = quantize_to_int8(fc_weight)
            fc_bias_int32 = (fc_bias / (input_scale * fc_scale)
                            ).astype(np.int32)

        with open(output_dir / 'fc_weights.h', 'w') as f:
            f.write("// FC layer weights and bias extracted from model\n\n")
            export_weights_to_c(fc_weight_int8, "fc_weights", f)
            export_bias_to_c(fc_bias_int32, "fc_bias", f)
        info(f"Generated: {output_dir}/fc_weights.h")
        quant_scales["fc_scale"] = fc_scale

    # LSTM weights
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
    conv_weight = (weights.get("conv_tflite.weight") or
                   weights.get("conv_onnx.weight"))
    conv_bias = (weights.get("conv_tflite.bias") or
                 weights.get("conv_onnx.bias"))
    if conv_weight is not None:
        conv_weight_int8 = (conv_weight
                           if conv_weight.dtype == np.int8
                           else quantize_to_int8(conv_weight)[0])
        with open(output_dir / 'conv_weights.h', 'w') as f:
            f.write("// CONV_2D weights and bias extracted from model\n\n")
            export_weights_to_c(conv_weight_int8, "conv_weights", f)
            if conv_bias is not None:
                conv_bias_int32 = (conv_bias.astype(np.int32)
                                  if conv_bias.dtype != np.int32 else conv_bias)
                export_bias_to_c(conv_bias_int32, "conv_bias", f)
        info(f"Generated: {output_dir}/conv_weights.h")

    # Depthwise Conv weights
    dw_weight = (weights.get("dw_tflite.weight") or
                 weights.get("dw_onnx.weight"))
    dw_bias = (weights.get("dw_tflite.bias") or
               weights.get("dw_onnx.bias"))
    if dw_weight is not None:
        dw_weight_int8 = (dw_weight
                         if dw_weight.dtype == np.int8
                         else quantize_to_int8(dw_weight)[0])
        with open(output_dir / 'dw_weights.h', 'w') as f:
            f.write("// Depthwise Conv2D weights and bias "
                    "extracted from model\n\n")
            export_weights_to_c(dw_weight_int8, "dw_weights", f)
            if dw_bias is not None:
                dw_bias_int32 = (dw_bias.astype(np.int32)
                                if dw_bias.dtype != np.int32 else dw_bias)
                export_bias_to_c(dw_bias_int32, "dw_bias", f)
        info(f"Generated: {output_dir}/dw_weights.h")

    return quant_scales
