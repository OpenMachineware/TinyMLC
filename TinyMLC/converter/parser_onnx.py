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

#!/usr/bin/env python3
"""ONNX model parser"""

import onnx
from onnx import numpy_helper
from utils.dump import warning

# ONNX operator to TinyMLC IR mapping
OP_MAP = {
    "Conv": "CONV_2D",
    "Gemm": "FULLY_CONNECTED",
    "MatMul": "FULLY_CONNECTED",
    "Relu": "RELU",
    "Softmax": "SOFTMAX",
    "Add": "ADD",
    "Sub": "SUB",
    "Mul": "MULTIPLY",
    "MaxPool": "MAX_POOL_2D",
    "AveragePool": "AVERAGE_POOL_2D",
    "Reshape": "RESHAPE",
    "Transpose": "TRANSPOSE",
    "Pad": "PAD",
    "Mean": "MEAN",
    "ReduceMean": "MEAN",
    "LSTM": "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "SVDF": "SVDF",
    "Concat": "CONCAT",
    "Sigmoid": "SIGMOID",
    "Tanh": "TANH",
}


def get_tensor_shape(graph, name, initializer_map=None):
    """Get tensor shape from graph"""
    # Check inputs
    for inp in graph.input:
        if inp.name == name:
            return [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    # Check outputs
    for out in graph.output:
        if out.name == name:
            return [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    # Check intermediate tensors
    for val in graph.value_info:
        if val.name == name:
            return [dim.dim_value for dim in val.type.tensor_type.shape.dim]
    # Check initializer (quantized weights in QDQ models)
    if initializer_map is not None and name in initializer_map:
        return list(initializer_map[name].shape)
    return []


def parse_model_onnx(model_path: str):
    """Parse ONNX model, return model_info"""
    # 0. Load model
    model = onnx.load(model_path)
    graph = model.graph

    # 1. Build global tensor index mapping
    tensor_index_map = {}
    next_idx = 0

    # Collect all tensor info
    tensors = {}

    # Assign indices to all weights
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        tensor_index_map[init.name] = next_idx
        tensors[next_idx] = {
            "name": init.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": tensor.size,
            "scale": 1.0,
            "zero_point": 0,
        }
        next_idx += 1

    # Assign indices to all inputs
    for inp in graph.input:
        if inp.name not in tensor_index_map:
            tensor_index_map[inp.name] = next_idx
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            tensors[next_idx] = {
                "name": inp.name,
                "shape": shape,
                "dtype": "float32",
                "size": 1 if not shape else 1,
                "scale": 1.0,
                "zero_point": 0,
            }
            # Calculate size
            size = 1
            for dim in shape:
                size *= dim
            tensors[next_idx]["size"] = size
            next_idx += 1

    # Assign indices to all outputs
    for out in graph.output:
        if out.name not in tensor_index_map:
            tensor_index_map[out.name] = next_idx
            shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            tensors[next_idx] = {
                "name": out.name,
                "shape": shape,
                "dtype": "float32",
                "size": 1,
                "scale": 1.0,
                "zero_point": 0,
            }
            size = 1
            for dim in shape:
                size *= dim
            tensors[next_idx]["size"] = size
            next_idx += 1

    # Assign indices to all intermediate tensors
    for val in graph.value_info:
        if val.name not in tensor_index_map:
            tensor_index_map[val.name] = next_idx
            shape = [dim.dim_value for dim in val.type.tensor_type.shape.dim]
            tensors[next_idx] = {
                "name": val.name,
                "shape": shape,
                "dtype": "float32",
                "size": 1,
                "scale": 1.0,
                "zero_point": 0,
            }
            size = 1
            for dim in shape:
                size *= dim
            tensors[next_idx]["size"] = size
            next_idx += 1

    # Build initializer name to array mapping (for QDQ nodes)
    initializer_map = {}
    for init in graph.initializer:
        initializer_map[init.name] = numpy_helper.to_array(init)

    # Assign indices to QDQ node outputs (these tensors are not in value_info)
    for node in graph.node:
        for out_name in node.output:
            if out_name not in tensor_index_map:
                # Get shape: if quantized weight, read from
                # initializer; otherwise use input shape
                shape = []
                if out_name in initializer_map:
                    shape = list(initializer_map[out_name].shape)
                elif node.input:
                    inp_name = node.input[0]
                    if inp_name in tensor_index_map:
                        shape = tensors[tensor_index_map[inp_name]].get(
                            "shape", []
                        )
                    else:
                        shape = get_tensor_shape(
                            graph, inp_name, initializer_map
                        )

                size = 1
                for dim in shape:
                    size *= dim

                tensor_index_map[out_name] = next_idx
                tensors[next_idx] = {
                    "name": out_name,
                    "shape": shape,
                    "dtype": (
                        "int8"
                        if node.op_type == "QuantizeLinear"
                        else "float32"
                    ),
                    "size": size,
                    "scale": 1.0,
                    "zero_point": 0,
                }
                next_idx += 1

    # Parse QDQ quantization parameters (QuantizeLinear/DequantizeLinear)

    # QDQ mapping: quantized node output -> original input
    # Used to replace computation operator inputs from
    # QuantizeLinear/DequantizeLinear outputs to original tensors
    qdq_map = {}

    # Traverse all nodes, extract quantization params and build mapping
    for node in graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            input_name = node.input[0]
            output_name = node.output[0]
            scale_name = node.input[1]

            # Build mapping: quantized node output -> original input
            qdq_map[output_name] = input_name
            # Recursive mapping: if input is also
            # quantized node output, continue mapping
            while input_name in qdq_map:
                input_name = qdq_map[input_name]
            qdq_map[output_name] = input_name

            if scale_name in initializer_map:
                scale_arr = initializer_map[scale_name]
                scale_val = float(scale_arr.flat[0])
                if input_name in tensor_index_map:
                    idx = tensor_index_map[input_name]
                    tensors[idx]["scale"] = scale_val
                if output_name in tensor_index_map:
                    idx = tensor_index_map[output_name]
                    tensors[idx]["scale"] = scale_val

            if len(node.input) >= 3 and node.input[2] in initializer_map:
                zp_arr = initializer_map[node.input[2]]
                zp_val = int(zp_arr.flat[0])
                if input_name in tensor_index_map:
                    idx = tensor_index_map[input_name]
                    tensors[idx]["zero_point"] = zp_val
                if output_name in tensor_index_map:
                    idx = tensor_index_map[output_name]
                    tensors[idx]["zero_point"] = zp_val

    # 2. Get inputs and outputs
    input_details = []
    for inp in graph.input:
        # Skip weights (initializers)
        if inp.name in [init.name for init in graph.initializer]:
            continue
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        input_details.append({
            "name": inp.name,
            "shape": shape,
            "dtype": "float32",  # ONNX default float
        })

    output_details = []
    for out in graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        output_details.append({
            "name": out.name,
            "shape": shape,
            "dtype": "float32",
        })

    # 3. Parse operators
    ops = []
    # Pseudo operators in QDQ models, no code generation needed
    skip_ops = {"QuantizeLinear", "DequantizeLinear", "Constant"}

    for node in graph.node:
        # Skip pseudo operators
        if node.op_type in skip_ops:
            continue

        # Use QDQ mapping to replace inputs: replace
        # DequantizeLinear output with original input
        mapped_inputs = []
        for inp_name in node.input:
            # If input is DequantizeLinear output, replace with original input
            if inp_name in qdq_map:
                mapped_inputs.append(qdq_map[inp_name])
            else:
                mapped_inputs.append(inp_name)

        op_info = {
            "index": len(ops),
            "op_name": node.op_type,
            "inputs": mapped_inputs,
            "outputs": list(node.output),
            "input_indices": [
                tensor_index_map.get(name, -1)
                for name in mapped_inputs
            ],
            "output_indices": [
                tensor_index_map.get(name, -1)
                for name in node.output
            ],
            "state": "created",
            "pass_flags": {},
            "input_details": [],
            "output_details": [],
        }

        # Fill input_details
        for inp_name in mapped_inputs:
            shape = []
            size = 1
            if inp_name in tensor_index_map:
                idx = tensor_index_map[inp_name]
                shape = tensors[idx].get("shape", [])
                size = tensors[idx].get("size", 1)
            else:
                shape = get_tensor_shape(graph, inp_name, initializer_map)
                for dim in shape:
                    size *= dim
            scale = 1.0
            zero_point = 0
            if inp_name in tensor_index_map:
                idx = tensor_index_map[inp_name]
                scale = tensors[idx].get("scale", 1.0)
                zero_point = tensors[idx].get("zero_point", 0)
            op_info["input_details"].append({
                "index": len(op_info["input_details"]),
                "name": inp_name,
                "shape": shape,
                "size": size,
                "scale": scale,
                "zero_point": zero_point,
            })
        # Fill output_details
        for out_name in node.output:
            shape = []
            size = 1
            if out_name in tensor_index_map:
                idx = tensor_index_map[out_name]
                shape = tensors[idx].get("shape", [])
                size = tensors[idx].get("size", 1)
            else:
                shape = get_tensor_shape(graph, out_name, initializer_map)
                for dim in shape:
                    size *= dim
            scale = 1.0
            zero_point = 0
            if out_name in tensor_index_map:
                idx = tensor_index_map[out_name]
                scale = tensors[idx].get("scale", 1.0)
                zero_point = tensors[idx].get("zero_point", 0)
            op_info["output_details"].append({
                "index": len(op_info["output_details"]),
                "name": out_name,
                "shape": shape,
                "size": size,
                "scale": scale,
                "zero_point": zero_point,
            })

        # Map to TinyMLC operator name
        if node.op_type in OP_MAP:
            op_info["op_name"] = OP_MAP[node.op_type]
            op_info["state"] = "translated"
            op_info["pass_flags"]["onnx_parse"] = "success"
        else:
            warning(f"Unknown ONNX operator: {node.op_type}")
            op_info["state"] = "created"
            op_info["pass_flags"]["unknown"] = "needs_implementation"

        # Handle special operators
        # Gemm is FULLY_CONNECTED
        if node.op_type == "Gemm":
            op_info["data_input_idx"] = op_info["input_indices"][0]
            op_info["fc_weights_idx"] = op_info["input_indices"][1]
            op_info["fc_bias_idx"] = (
                op_info["input_indices"][2]
                if len(node.input) >= 3 else None
            )
            op_info["weights_name"] = node.input[1]
            op_info["bias_name"] = (
                node.input[2]
                if len(node.input) >= 3 else None
            )

            # fc_scale will be calculated during weight extraction
            op_info["fc_scale"] = 0.01  # default, will be updated
            op_info["fc_output_scale"] = 1.0  # default

        if node.op_type == "Conv":
            op_info["data_input_idx"] = op_info["input_indices"][0]
            op_info["conv_weights_idx"] = op_info["input_indices"][1]
            op_info["conv_bias_idx"] = (
                op_info["input_indices"][2]
                if len(node.input) >= 3 else None
            )

            weights_name = node.input[1]

            # Infer kernel_shape from weight shape
            kernel_h, kernel_w = 1, 1
            if weights_name in initializer_map:
                weight_tensor = initializer_map[weights_name]
                if len(weight_tensor.shape) >= 4:
                    kernel_h = weight_tensor.shape[2]
                    kernel_w = weight_tensor.shape[3]

            # Get input shape
            input_shape = get_tensor_shape(graph, node.input[0])
            output_shape = get_tensor_shape(graph, node.output[0])

            input_h = input_shape[2] if len(input_shape) >= 4 else 1
            input_w = input_shape[3] if len(input_shape) >= 4 else 1
            input_c = input_shape[1] if len(input_shape) >= 4 else 1

            output_h = output_shape[2] if len(output_shape) >= 4 else 1
            output_w = output_shape[3] if len(output_shape) >= 4 else 1
            output_c = output_shape[1] if len(output_shape) >= 4 else 1

            # Extract stride and padding from attributes
            strides = [1, 1]
            pads = [0, 0, 0, 0]
            for attr in node.attribute:
                if attr.name == "strides":
                    strides = list(attr.ints)
                elif attr.name == "pads":
                    pads = list(attr.ints)

            op_info["conv_params"] = {
                "input_h": input_h,
                "input_w": input_w,
                "input_c": input_c,
                "output_h": output_h,
                "output_w": output_w,
                "output_c": output_c,
                "kernel_h": kernel_h,
                "kernel_w": kernel_w,
                "stride_h": strides[0] if len(strides) >= 2 else strides[0],
                "stride_w": strides[1] if len(strides) >= 2 else strides[0],
                "padding_h": pads[0] if len(pads) >= 2 else 0,
                "padding_w": pads[1] if len(pads) >= 2 else 0,
            }

        if node.op_type == "Softmax":
            # Get softmax axis size from output shape
            output_shape = get_tensor_shape(
                graph, node.output[0], initializer_map)
            if output_shape:
                # Default axis is -1 (last dimension)
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = attr.i
                # Convert negative axis to positive
                if axis < 0:
                    axis = len(output_shape) + axis
                op_info["softmax_size"] = (output_shape[axis]
                                          if axis < len(output_shape)
                                          else output_shape[-1])
            else:
                # Fallback: get from output_details
                if op_info["output_details"]:
                    op_info["softmax_size"] = (
                        op_info["output_details"][0].get("size", 10))

        if node.op_type == "Reshape":
            # Target shape in inputs[1]
            target_shape_name = node.input[1]
            target_shape = initializer_map.get(target_shape_name)
            if target_shape is not None:
                op_info["reshape_target_shape"] = target_shape.tolist()

        # Extract conv params (from attributes)
        if node.op_type in ["MaxPool", "AveragePool"]:
            strides = [1, 1]
            pads = [0, 0, 0, 0]
            kernel_shape = []
            for attr in node.attribute:
                if attr.name == "strides":
                    strides = list(attr.ints)
                elif attr.name == "pads":
                    pads = list(attr.ints)
                elif attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints)

            input_shape = get_tensor_shape(
                graph, node.input[0], initializer_map
            )
            output_shape = get_tensor_shape(
                graph, node.output[0], initializer_map
            )

            op_info["pool_params"] = {
                "input_h": input_shape[2] if len(input_shape) >= 4 else 1,
                "input_w": input_shape[3] if len(input_shape) >= 4 else 1,
                "input_c": input_shape[1] if len(input_shape) >= 4 else 1,
                "output_h": output_shape[2] if len(output_shape) >= 4 else 1,
                "output_w": output_shape[3] if len(output_shape) >= 4 else 1,
                "output_c": output_shape[1] if len(output_shape) >= 4 else 1,
                "pool_size_h": (
                    kernel_shape[0] if len(kernel_shape) >= 2
                    else kernel_shape[0]
                ),
                "pool_size_w": (
                    kernel_shape[1] if len(kernel_shape) >= 2
                    else kernel_shape[0]
                ),
                "stride_h": strides[0] if len(strides) >= 2 else strides[0],
                "stride_w": strides[1] if len(strides) >= 2 else strides[0],
            }

            op_info["data_input_idx"] = op_info["input_indices"][0]

        if node.op_type == "SVDF":
            inputs = op_info["input_indices"]
            if len(inputs) >= 3:
                op_info["data_input_idx"] = inputs[0]
                op_info["svdf_weights_idx"] = inputs[1]
                op_info["svdf_bias_idx"] = inputs[2]

            rank = 1
            activation_function = "Tanh"
            for attr in node.attribute:
                if attr.name == "rank":
                    rank = attr.i
                elif attr.name == "activation_function":
                    activation_function = attr.s.decode('utf-8')

            input_shape = get_tensor_shape(
                graph, node.input[0], initializer_map
            )
            output_shape = get_tensor_shape(
                graph, node.output[0], initializer_map
            )

            time_steps = input_shape[1] if len(input_shape) >= 3 else 1
            input_size = input_shape[2] if len(input_shape) >= 3 else 1
            units = output_shape[2] // rank if len(output_shape) >= 3 else 1

            op_info["svdf_params"] = {
                "rank": rank,
                "activation_function": activation_function,
                "time_steps": time_steps,
                "input_size": input_size,
                "units": units,
            }

        if node.op_type in ["Mean", "ReduceMean"]:
            input_shape = get_tensor_shape(
                graph, node.input[0], initializer_map
            )
            output_shape = get_tensor_shape(
                graph, node.output[0], initializer_map
            )

            op_info["data_input_idx"] = op_info["input_indices"][0]
            op_info["mean_params"] = {
                "input_dims": len(input_shape),
                "input_shape": input_shape,
                "output_shape": output_shape,
            }

        ops.append(op_info)

    # Store raw weights (without standardized keys)
    raw_weights = {}
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        raw_weights[init.name] = tensor

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "weights": raw_weights,  # Raw weights, will be processed
                                  # by extract_all_weights_onnx
        "tensors": tensors,
        "initializer_map": initializer_map,  # Keep for weight extraction
    }


def extract_all_weights_onnx(model_path, model_info):
    """Extract all weights from ONNX model and store in model_info['weights']

    Uses source-specific keys: fc_onnx.weight, conv_onnx.weight, etc.

    Args:
        model_path: ONNX model file path (for consistency with LiteRT interface)
        model_info: Model info dict from parse_model_onnx
    """
    weights = model_info.get("weights", {})
    initializer_map = model_info.get("initializer_map", {})
    ops = model_info.get("ops", [])

    # Extract weights with standardized keys based on operator type
    for op in ops:
        op_name = op.get("op_name")

        if op_name == "FULLY_CONNECTED":
            # Gemm/MatMul operator
            input_indices = op.get("input_indices", [])
            if len(input_indices) >= 2:
                weights_idx = input_indices[1]
                weights_name = op.get("weights_name")
                if weights_name and weights_name in weights:
                    weights["fc_onnx.weight"] = weights[weights_name]
                if len(input_indices) >= 3:
                    bias_name = op.get("bias_name")
                    if bias_name and bias_name in weights:
                        weights["fc_onnx.bias"] = weights[bias_name]

        elif op_name == "CONV_2D":
            # Conv operator
            input_indices = op.get("input_indices", [])
            if len(input_indices) >= 2:
                weights_name = None
                # Find weights name from original node input
                tensors = model_info.get("tensors", {})
                for idx, tensor_info in tensors.items():
                    if idx == input_indices[1]:
                        weights_name = tensor_info.get("name")
                        break
                if weights_name and weights_name in weights:
                    weights["conv_onnx.weight"] = weights[weights_name]
                if len(input_indices) >= 3:
                    bias_idx = input_indices[2]
                    bias_name = None
                    for idx, tensor_info in tensors.items():
                        if idx == bias_idx:
                            bias_name = tensor_info.get("name")
                            break
                    if bias_name and bias_name in weights:
                        weights["conv_onnx.bias"] = weights[bias_name]

        elif op_name == "SVDF":
            # SVDF operator
            input_indices = op.get("input_indices", [])
            if len(input_indices) >= 3:
                weights_idx = input_indices[1]
                bias_idx = input_indices[2]
                # Find weights name
                tensors = model_info.get("tensors", {})
                for idx, tensor_info in tensors.items():
                    if idx == weights_idx:
                        weights["svdf_onnx.weight"] = weights.get(
                            tensor_info.get("name"))
                    if idx == bias_idx:
                        weights["svdf_onnx.bias"] = weights.get(
                            tensor_info.get("name"))

    # Update model_info weights
    model_info["weights"] = weights
    return weights
