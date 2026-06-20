#!/usr/bin/env python3
"""ONNX 模型解析器"""

import onnx
from onnx import helper, numpy_helper
import numpy as np
from pathlib import Path

from tinymlc.utils import fatal_error, info, warning

# ONNX 算子到 tinymlc IR 的映射
OP_MAP = {
    "Conv": "CONV_2D",
    "Gemm": "FULLY_CONNECTED",
    "MatMul": "FULLY_CONNECTED",
    "Relu": "RELU",
    "Softmax": "SOFTMAX",
    "Add": "ADD",
    "MaxPool": "MAX_POOL_2D",
    "AveragePool": "AVERAGE_POOL_2D",
    "Reshape": "RESHAPE",
    "Transpose": "TRANSPOSE",
    "Pad": "PAD",
    "Mean": "MEAN",
    "LSTM": "UNIDIRECTIONAL_SEQUENCE_LSTM",
}


def parse_model(model_path: str):
    """解析 ONNX 模型，返回 model_info"""

    # 1. 加载模型
    model = onnx.load(model_path)
    graph = model.graph

    # 2. 获取输入输出
    input_details = []
    for inp in graph.input:
        # 跳过权重（初始值）
        if inp.name in [init.name for init in graph.initializer]:
            continue
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        input_details.append({
            "name": inp.name,
            "shape": shape,
            "dtype": "float32",  # ONNX 默认浮点
        })

    output_details = []
    for out in graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        output_details.append({
            "name": out.name,
            "shape": shape,
            "dtype": "float32",
        })

    # 3. 提取权重（Initializer）
    weights = {}
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        weights[init.name] = tensor

    # 4. 解析算子
    ops = []
    for node in graph.node:
        op_info = {
            "index": len(ops),
            "op_name": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "input_indices": [],  # ONNX 没有索引，用名称代替
            "output_indices": [],
            "state": "created",
            "pass_flags": {},
            "input_details": [],
            "output_details": [],
        }

        # 映射到 tinymlc 算子名
        if node.op_type in OP_MAP:
            op_info["op_name"] = OP_MAP[node.op_type]
        else:
            warning(f"未知 ONNX 算子: {node.op_type}")
            op_info["state"] = "created"
            op_info["pass_flags"]["unknown"] = "needs_implementation"

        # 处理特殊算子
        if node.op_type == "Gemm":
            # Gemm: input, weights, bias
            # 记录权重和 bias 名称
            if len(node.input) >= 3:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = node.input[2]
            elif len(node.input) >= 2:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = None

        if node.op_type == "Conv":
            if len(node.input) >= 3:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = node.input[2]
            elif len(node.input) >= 2:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = None

        # 提取卷积参数（从 attributes）
        # 提取卷积参数
        if node.op_type in ["Conv", "MaxPool", "AveragePool"]:
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

            # 如果 kernel_shape 为空，从权重形状推断
            if not kernel_shape and node.op_type == "Conv":
                # 从权重张量推断 kernel_shape
                weights_name = node.input[1]
                if weights_name in weights:
                    weight_tensor = weights[weights_name]
                    # Conv 权重形状: [out_channels, in_channels, kernel_h, kernel_w]
                    if len(weight_tensor.shape) >= 4:
                        kernel_shape = [weight_tensor.shape[2],
                                        weight_tensor.shape[3]]

            op_info["conv_params"] = {
                "stride_h": strides[0] if len(strides) >= 2 else strides[0],
                "stride_w": strides[1] if len(strides) >= 2 else strides[0],
                "padding_h": pads[0] if len(pads) >= 2 else 0,
                "padding_w": pads[1] if len(pads) >= 2 else 0,
                "kernel_h": kernel_shape[0] if len(kernel_shape) >= 2 else
                kernel_shape[0],
                "kernel_w": kernel_shape[1] if len(kernel_shape) >= 2 else
                kernel_shape[0],
            }

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "weights": weights,
    }
