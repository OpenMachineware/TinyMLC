#!/usr/bin/env python3
"""LiteRT-based TFLite model parser

Note: Uses private API _get_ops_details() as LiteRT has no
public operator list API.
LSTM input indices [1:5], [5:9], [9:13] follow TFLite
UNIDIRECTIONAL_SEQUENCE_LSTM spec.
"""

from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
import numpy as np

from utils.dump import fatal_error, info, warning


# Weight extraction functions
def extract_fc_weights(interpreter, op_info):
    """Extract FULLY_CONNECTED layer weights and bias using LiteRT API"""
    weights_idx = op_info.get("fc_weights_idx")
    bias_idx = op_info.get("fc_bias_idx")

    if weights_idx is None:
        return None, None

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = (interpreter.get_tensor(bias_idx)
                if bias_idx is not None else None)
    except ValueError as e:
        fatal_error(f"Cannot get FC tensor: {e}",
                    "Ensure model is loaded correctly")

    info(f"FC weights: shape={weights.shape}, dtype={weights.dtype}")
    if bias is not None:
        info(f"FC bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def extract_lstm_weights(interpreter, op_info):
    """Extract LSTM gate weights and biases using LiteRT API

    TFLite LSTM input indices follow standard spec:
    - inputs[1:5]: input gate weights (i, f, g, o)
    - inputs[5:9]: recurrent weights (i, f, g, o)
    - inputs[9:13]: biases (i, f, g, o)
    """
    indices = op_info.get("lstm_weight_indices", {})
    input_indices = indices.get("input", [])
    recurrent_indices = indices.get("recurrent", [])
    bias_indices = indices.get("bias", [])

    if not input_indices or not recurrent_indices:
        return None

    gate_order = ['i', 'f', 'g', 'o']
    lstm_weights = {'input': {}, 'recurrent': {}, 'bias': {}}

    for gate, idx in zip(gate_order, input_indices):
        try:
            lstm_weights['input'][gate] = interpreter.get_tensor(idx)
        except ValueError:
            lstm_weights['input'][gate] = None

    for gate, idx in zip(gate_order, recurrent_indices):
        try:
            lstm_weights['recurrent'][gate] = interpreter.get_tensor(idx)
        except ValueError:
            lstm_weights['recurrent'][gate] = None

    for gate, idx in zip(gate_order, bias_indices):
        try:
            lstm_weights['bias'][gate] = interpreter.get_tensor(idx)
        except ValueError:
            lstm_weights['bias'][gate] = None

    return lstm_weights


def extract_conv_weights(interpreter, op_info):
    """Extract CONV_2D layer weights and bias using LiteRT API"""
    weights_idx = op_info.get("conv_weights_idx")
    bias_idx = op_info.get("conv_bias_idx")

    if weights_idx is None:
        return None, None

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = (interpreter.get_tensor(bias_idx)
                if bias_idx is not None else None)
    except ValueError as e:
        fatal_error(f"Cannot get CONV_2D tensor: {e}",
                    "Ensure model is loaded correctly")

    info(f"CONV_2D weights: shape={weights.shape}, dtype={weights.dtype}")
    if bias is not None:
        info(f"CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def extract_dw_weights(interpreter, op_info):
    """Extract DEPTHWISE_CONV_2D layer weights and bias using LiteRT API"""
    weights_idx = op_info.get("dw_weights_idx")
    bias_idx = op_info.get("dw_bias_idx")

    if weights_idx is None:
        return None, None

    try:
        weights = interpreter.get_tensor(weights_idx)
        bias = (interpreter.get_tensor(bias_idx)
                if bias_idx is not None else None)
    except ValueError as e:
        fatal_error(f"Cannot get DEPTHWISE_CONV_2D tensor: {e}",
                    "Ensure model is loaded correctly")

    info(f"DEPTHWISE_CONV_2D weights: shape={weights.shape}, "
         f"dtype={weights.dtype}")
    if bias is not None:
        info(f"DEPTHWISE_CONV_2D bias: shape={bias.shape}, dtype={bias.dtype}")
    return weights, bias


def extract_all_weights_litert(model_path, model_info):
    """Extract all weights from LiteRT model and store in model_info['weights']

    Uses source-specific keys: fc_tflite.weight, lstm_tflite.weight_ih, etc.

    Args:
        model_path: TFLite model file path
        model_info: Model info dict from parse_model_tflite
    """
    # Create interpreter internally
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

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

    fc_weights, fc_bias = (extract_fc_weights(interpreter, fc_op_info)
                          if fc_op_info else (None, None))
    lstm_weights = (extract_lstm_weights(interpreter, lstm_op_info)
                   if lstm_op_info else None)
    conv_weights, conv_bias = (extract_conv_weights(interpreter, conv_op_info)
                              if conv_op_info else (None, None))
    dw_weights, dw_bias = (extract_dw_weights(interpreter, dw_op_info)
                          if dw_op_info else (None, None))

    model_info["weights"] = {}

    if fc_weights is not None and fc_bias is not None:
        model_info["weights"]["fc_tflite.weight"] = fc_weights
        model_info["weights"]["fc_tflite.bias"] = fc_bias

    if lstm_weights and lstm_weights['input']:
        gates = ['i', 'f', 'g', 'o']
        if all(lstm_weights['input'].get(g) is not None for g in gates):
            input_concat = np.concatenate(
                [lstm_weights['input'][g].flatten() for g in gates])
            model_info["weights"]["lstm_tflite.weight_ih"] = input_concat
        if all(lstm_weights['recurrent'].get(g) is not None for g in gates):
            recurrent_concat = np.concatenate(
                [lstm_weights['recurrent'][g].flatten() for g in gates])
            model_info["weights"]["lstm_tflite.weight_hh"] = recurrent_concat
        if all(lstm_weights['bias'].get(g) is not None for g in gates):
            bias_concat = np.concatenate(
                [lstm_weights['bias'][g].flatten() for g in gates])
            model_info["weights"]["lstm_tflite.bias"] = bias_concat

    if conv_weights is not None:
        model_info["weights"]["conv_tflite.weight"] = conv_weights
        if conv_bias is not None:
            model_info["weights"]["conv_tflite.bias"] = conv_bias

    if dw_weights is not None:
        model_info["weights"]["dw_tflite.weight"] = dw_weights
        if dw_bias is not None:
            model_info["weights"]["dw_tflite.bias"] = dw_bias

    return fc_weights, fc_bias, lstm_weights, conv_weights, \
           conv_bias, dw_weights, dw_bias


def parse_model_tflite(model_path: str):
    """Parse TFLite model using LiteRT"""

    # 1. Load model (using LiteRT Interpreter)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. Get input/output tensors (same as old API)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Get all tensor info, try to read constant tensor data
    tensor_details = interpreter.get_tensor_details()
    tensors = {}  # Unified key: tensors (was tensors)

    for tensor in tensor_details:
        shape = tensor["shape"]
        tensor_info = {
            "name": tensor["name"],
            "shape": list(tensor["shape"]),
            "dtype": str(tensor["dtype"]),
            "size": int(np.prod(shape)) if shape is not None and len(
                shape) > 0 else 1,
            "scale": (
                tensor["quantization"][0]
                if tensor["quantization"][0] is not None else 1.0
            ),
            "zero_point": (
                tensor["quantization"][1]
                if tensor["quantization"][1] is not None else 0
            ),
        }

        # Try to read constant tensor data (for Reshape target shape etc)
        try:
            data = interpreter.get_tensor(tensor["index"])
            tensor_info["data"] = data
        except:
            pass

        tensors[tensor["index"]] = tensor_info

    # 4. Get operator list
    ops = []
    for op in interpreter._get_ops_details():
        # Skip DELEGATE operator
        if op["op_name"] == "DELEGATE":
            continue

        op_info = {
            "index": op["index"],
            "op_name": op["op_name"],
            "inputs": [inp for inp in op["inputs"] if inp != -1],
            "outputs": [out for out in op["outputs"] if out != -1],
            "input_indices": [inp for inp in op["inputs"] if inp != -1],
            "output_indices": [out for out in op["outputs"] if out != -1],
            "state": "created",
            "pass_flags": {},
            "input_details": [],
            "output_details": [],
        }

        # Set state based on operator type
        if op["op_name"] == "ADD":
            inputs = op_info["input_indices"]
            # Input count based on operator spec:
            # inputs[0] inputs[1] are valid
            if len(inputs) >= 2:
                op_info["add_input1_idx"] = inputs[0]
                op_info["add_input2_idx"] = inputs[1]
            op_info["state"] = "translated"
            op_info["pass_flags"]["add_check"] = "success"
        elif op["op_name"] == "FULLY_CONNECTED":
            inputs = op_info["input_indices"]
            # Input count based on operator spec:
            # inputs[0] inputs[1] inputs[2] are valid
            if len(inputs) >= 3:
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = inputs[1]
                op_info["fc_bias_idx"] = inputs[2]
            # Input count based on operator spec:
            # inputs[0] inputs[1] are valid
            elif len(inputs) >= 2:
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = inputs[1]
                op_info["fc_bias_idx"] = None
            else:
                # Input count is based on operator spec, inputs[0] is valid
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = None
                op_info["fc_bias_idx"] = None
            op_info["state"] = "translated"
            op_info["pass_flags"]["fc_check"] = "success"
        elif op["op_name"] == "SOFTMAX":
            if len(op_info["input_indices"]) < 1:
                fatal_error("SOFTMAX missing input", "Check model format")
            if len(op_info["output_indices"]) < 1:
                fatal_error("SOFTMAX missing output", "Check model format")
            op_info["state"] = "translated"
            op_info["pass_flags"]["softmax_check"] = "success"
        elif op["op_name"] == "RESHAPE":
            inputs = op_info["input_indices"]
            if len(inputs) < 2:
                fatal_error(
                    "RESHAPE missing target shape parameter",
                    "Check model format"
                )

            shape_idx = inputs[1]
            shape_tensor = tensors.get(shape_idx, {})

            # Prefer reading actual value from data
            if "data" in shape_tensor:
                target_shape = [int(s) for s in shape_tensor["data"].flatten()]
            else:
                target_shape = shape_tensor.get("shape", [])

            if not target_shape:
                fatal_error(
                    "RESHAPE cannot extract target shape",
                    "Check model format"
                )

            # Handle dynamic dimension -1
            if -1 in target_shape:
                # Get input tensor size
                input_idx = inputs[0]
                input_tensor = tensors.get(input_idx, {})
                input_size = input_tensor.get("size", 1)

                # Calculate actual value for -1
                other_dims = 1
                for s in target_shape:
                    if s != -1:
                        other_dims *= s
                if other_dims > 0:
                    dynamic_size = input_size // other_dims
                    target_shape = [dynamic_size if s == -1 else s for s in
                                    target_shape]

            op_info["reshape_target_shape"] = [int(s) for s in target_shape]
            op_info["state"] = "translated"
            op_info["pass_flags"]["reshape_check"] = "success"
        elif op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            inputs = op_info["input_indices"]
            if len(inputs) >= 13:
                # TFLite/LiteRT LSTM input order:
                # [0] input data
                # [1-4] input gate weights
                # [5-8] recurrent weights
                # [9-12] biases
                # TFLite/LiteRT UNIDIRECTIONAL_SEQUENCE_LSTM
                # operator input order:
                # [0] input data
                # [1-4] input gate weights (i, f, g, o)
                # [5-8] recurrent weights (i, f, g, o)
                # [9-12] biases (i, f, g, o)
                # [13+] other parameters
                op_info["lstm_weight_indices"] = {
                    "input": inputs[1:5],
                    "recurrent": inputs[5:9],
                    "bias": inputs[9:13],
                }

                # Extract hidden_size from output shape
                output_shape = tensors.get(op_info["output_indices"][0],
                                              {}).get("shape", [])
                if len(output_shape) >= 3:
                    hidden_size = output_shape[2]
                else:
                    fatal_error(
                        "Cannot extract hidden_size from LSTM output shape",
                        "Check model format"
                    )

                # Extract time_steps, batch_size, input_size from input shape
                input_shape = tensors.get(inputs[0], {}).get("shape", [])
                if len(input_shape) >= 3:
                    op_info["lstm_params"] = {
                        "time_steps": input_shape[1],
                        # [batch, time_steps, input_size]
                        "batch_size": input_shape[0],
                        "input_size": input_shape[2],
                        "hidden_size": hidden_size,
                    }
                else:
                    fatal_error(
                        "Cannot extract parameters from LSTM input shape",
                        "Check model format"
                    )

                op_info["state"] = "translated"
                op_info["pass_flags"]["lstm_check"] = "success"
            else:
                fatal_error("LSTM input incomplete", "Check model format")
        elif op["op_name"] == "SVDF":
            # SVDF needs to record weight indices
            inputs = op_info["input_indices"]
            # Input count based on operator spec:
            # inputs[0] inputs[1] inputs[2] are valid
            if len(inputs) >= 3:
                op_info["data_input_idx"] = inputs[0]
                op_info["svdf_weights_idx"] = inputs[1]
                op_info["svdf_bias_idx"] = inputs[2]
            op_info["state"] = "translated"
            op_info["pass_flags"]["svdf_check"] = "success"
        elif op["op_name"] == "CONV_2D":
            inputs = op_info["input_indices"]
            if len(inputs) >= 3:
                op_info["data_input_idx"] = inputs[0]
                op_info["conv_weights_idx"] = inputs[1]
                op_info["conv_bias_idx"] = inputs[2]
            else:
                fatal_error("CONV_2D input incomplete", "Check model format")

            # Extract convolution parameters
            # Calculate stride, padding, etc. from input/output shapes
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            # Weight shape: [out_channels, kernel_h, kernel_w, in_channels]
            weights_tensor = tensors.get(inputs[1], {})
            weights_shape = weights_tensor.get("shape", [])

            # Default parameters
            stride_h = 1
            stride_w = 1
            padding = "VALID"

            # Infer stride and padding from input/output shapes
            if len(input_shape) >= 4 and len(output_shape) >= 4 and len(
                    weights_shape) >= 4:
                input_h = input_shape[1]
                input_w = input_shape[2]
                output_h = output_shape[1]
                output_w = output_shape[2]
                kernel_h = weights_shape[1]
                kernel_w = weights_shape[2]

                # Calculate stride (assuming stride_h == stride_w)
                if input_h > output_h:
                    stride_h = (input_h - kernel_h) // (
                                output_h - 1) if output_h > 1 else 1
                    stride_w = (input_w - kernel_w) // (
                                output_w - 1) if output_w > 1 else 1

                # Determine padding
                # If output size = ceil(input / stride), usually SAME padding
                # Otherwise VALID
                expected_h = (input_h + stride_h - 1) // stride_h
                if output_h == expected_h:
                    padding = "SAME"
                else:
                    padding = "VALID"

            op_info["conv_params"] = {
                "input_h": input_shape[1] if len(input_shape) >= 4 else 0,
                "input_w": input_shape[2] if len(input_shape) >= 4 else 0,
                "input_c": input_shape[3] if len(input_shape) >= 4 else 0,
                "output_h": output_shape[1] if len(output_shape) >= 4 else 0,
                "output_w": output_shape[2] if len(output_shape) >= 4 else 0,
                "output_c": output_shape[3] if len(output_shape) >= 4 else 0,
                "kernel_h": weights_shape[1] if len(weights_shape) >= 4 else 0,
                "kernel_w": weights_shape[2] if len(weights_shape) >= 4 else 0,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "padding": padding,
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["conv_check"] = "success"
        elif op["op_name"] == "MAX_POOL_2D":
            inputs = op_info["input_indices"]
            if len(inputs) < 1:
                fatal_error("MAX_POOL_2D missing input", "Check model format")

            op_info["data_input_idx"] = inputs[0]

            # Extract parameters from input/output shapes
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            # Default parameters
            pool_size_h = 2
            pool_size_w = 2
            stride_h = 2
            stride_w = 2
            padding = "VALID"

            if len(input_shape) >= 4 and len(output_shape) >= 4:
                input_h = input_shape[1]
                input_w = input_shape[2]
                output_h = output_shape[1]
                output_w = output_shape[2]

                # Infer stride from input/output
                if input_h > output_h:
                    stride_h = input_h // output_h if output_h > 0 else 1
                    stride_w = input_w // output_w if output_w > 0 else 1

            op_info["pool_params"] = {
                "input_h": input_shape[1] if len(input_shape) >= 4 else 0,
                "input_w": input_shape[2] if len(input_shape) >= 4 else 0,
                "input_c": input_shape[3] if len(input_shape) >= 4 else 0,
                "output_h": output_shape[1] if len(output_shape) >= 4 else 0,
                "output_w": output_shape[2] if len(output_shape) >= 4 else 0,
                "output_c": output_shape[3] if len(output_shape) >= 4 else 0,
                "pool_size_h": pool_size_h,
                "pool_size_w": pool_size_w,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "padding": padding,
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["pool_check"] = "success"
        elif op["op_name"] == "DEPTHWISE_CONV_2D":
            inputs = op_info["input_indices"]
            if len(inputs) < 3:
                fatal_error(
                    "DEPTHWISE_CONV_2D input incomplete",
                    "Check model format"
                )

            op_info["data_input_idx"] = inputs[0]
            op_info["dw_weights_idx"] = inputs[1]
            op_info["dw_bias_idx"] = inputs[2]

            # Extract parameters
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])
            weights_tensor = tensors.get(inputs[1], {})
            weights_shape = weights_tensor.get("shape", [])

            op_info["dw_params"] = {
                "input_h": input_shape[1] if len(input_shape) >= 4 else 0,
                "input_w": input_shape[2] if len(input_shape) >= 4 else 0,
                "input_c": input_shape[3] if len(input_shape) >= 4 else 0,
                "output_h": output_shape[1] if len(output_shape) >= 4 else 0,
                "output_w": output_shape[2] if len(output_shape) >= 4 else 0,
                "output_c": output_shape[3] if len(output_shape) >= 4 else 0,
                "kernel_h": weights_shape[1] if len(weights_shape) >= 4 else 0,
                "kernel_w": weights_shape[2] if len(weights_shape) >= 4 else 0,
                "depth_multiplier": weights_shape[3] // input_shape[3] if len(
                    input_shape) >= 4 and len(weights_shape) >= 4 else 1,
                "stride_h": 1,
                "stride_w": 1,
                "padding_h": 0,
                "padding_w": 0,
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["dw_check"] = "success"
        elif op["op_name"] == "RELU":
            # ReLU only needs input/output, no extra parameters
            if len(op_info["input_indices"]) < 1:
                fatal_error("RELU missing input", "Check model format")
            if len(op_info["output_indices"]) < 1:
                fatal_error("RELU missing output", "Check model format")
            op_info["state"] = "translated"
            op_info["pass_flags"]["relu_check"] = "success"
        elif op["op_name"] == "AVERAGE_POOL_2D":
            inputs = op_info["input_indices"]
            if len(inputs) < 1:
                fatal_error(
                    "AVERAGE_POOL_2D missing input",
                    "Check model format"
                )

            op_info["data_input_idx"] = inputs[0]

            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            pool_h = 2
            pool_w = 2
            stride_h = 2
            stride_w = 2

            if len(input_shape) >= 4 and len(output_shape) >= 4:
                input_h = input_shape[1]
                input_w = input_shape[2]
                output_h = output_shape[1]
                output_w = output_shape[2]
                if input_h > output_h and output_h > 0:
                    stride_h = input_h // output_h
                    stride_w = input_w // output_w

            op_info["pool_params"] = {
                "input_h": input_shape[1] if len(input_shape) >= 4 else 0,
                "input_w": input_shape[2] if len(input_shape) >= 4 else 0,
                "input_c": input_shape[3] if len(input_shape) >= 4 else 0,
                "output_h": output_shape[1] if len(output_shape) >= 4 else 0,
                "output_w": output_shape[2] if len(output_shape) >= 4 else 0,
                "output_c": output_shape[3] if len(output_shape) >= 4 else 0,
                "pool_h": pool_h,
                "pool_w": pool_w,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "padding": "VALID",
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["avg_pool_check"] = "success"
        elif op["op_name"] == "TRANSPOSE":
            inputs = op_info["input_indices"]
            if len(inputs) < 2:
                fatal_error(
                    "TRANSPOSE missing perm parameter",
                    "Check model format"
                )

            # inputs[0] = input data, inputs[1] = perm (transpose order)
            op_info["data_input_idx"] = inputs[0]
            op_info["transpose_perm_idx"] = inputs[1]

            # Infer transpose parameters from input/output shapes
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]
            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})
            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            op_info["transpose_params"] = {
                "input_dims": len(input_shape),
                "output_dims": len(output_shape),
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["transpose_check"] = "success"
        elif op["op_name"] == "QUANTIZE":
            op_info["state"] = "translated"
            op_info["pass_flags"]["quantize_check"] = "success"
        elif op["op_name"] == "PAD":
            inputs = op_info["input_indices"]
            if len(inputs) < 2:
                fatal_error(
                    "PAD missing padding parameter",
                    "Check model format"
                )

            op_info["data_input_idx"] = inputs[0]
            op_info["pad_paddings_idx"] = inputs[1]
            op_info["state"] = "translated"
            op_info["pass_flags"]["pad_check"] = "success"
        elif op["op_name"] == "MEAN":
            inputs = op_info["input_indices"]
            if len(inputs) < 1:
                fatal_error("MEAN missing input", "Check model format")

            op_info["data_input_idx"] = inputs[0]
            # Record axis parameter if present
            if len(inputs) >= 2:
                op_info["mean_axis_idx"] = inputs[1]

            # Extract parameters from input/output shapes
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensors.get(input_idx, {})
            output_tensor = tensors.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            op_info["mean_params"] = {
                "input_dims": len(input_shape),
                "output_dims": len(output_shape),
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["mean_check"] = "success"
        elif op["op_name"] == "DELEGATE":
            continue  # skip
        else:
            # Unknown operator, keep created state
            warning(
                "Unknown operator encountered, "
                "please submit an issue or patch set"
            )
            op_info["state"] = "created"
            op_info["pass_flags"]["unknown"] = "needs_implementation"

        # Add input/output details
        for inp_idx in op_info["input_indices"]:
            tensor_info = tensors.get(inp_idx, {})
            op_info["input_details"].append({
                "index": inp_idx,
                "name": tensor_info.get("name", "unknown"),
                "shape": tensor_info.get("shape", []),
                "size": tensor_info.get("size", 0),
                "scale": tensor_info.get("scale", 1.0),
                "zero_point": tensor_info.get("zero_point", 0),
            })

        for out_idx in op_info["output_indices"]:
            tensor_info = tensors.get(out_idx, {})
            op_info["output_details"].append({
                "index": out_idx,
                "name": tensor_info.get("name", "unknown"),
                "shape": tensor_info.get("shape", []),
                "size": tensor_info.get("size", 0),
                "scale": tensor_info.get("scale", 1.0),
                "zero_point": tensor_info.get("zero_point", 0),
            })

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "tensors": tensors,
        "weights": {},  # LiteRT uses separate weight extraction,
                        # kept for unified interface
        "quant_scales": {},  # Quantization scales
                             # (LiteRT stores per-tensor in tensors)
    }
