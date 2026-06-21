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


def get_tensor_shape(graph, name):
    """从 graph 中获取张量形状"""
    # 检查输入
    for inp in graph.input:
        if inp.name == name:
            return [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    # 检查输出
    for out in graph.output:
        if out.name == name:
            return [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    # 检查中间张量
    for val in graph.value_info:
        if val.name == name:
            return [dim.dim_value for dim in val.type.tensor_type.shape.dim]
    return []


def parse_model_onnx(model_path: str):
    """解析 ONNX 模型，返回 model_info"""
    # 0. 加载模型
    model = onnx.load(model_path)
    graph = model.graph

    # 1. 构建全局张量索引映射
    tensor_index_map = {}
    next_idx = 0

    # 收集所有张量信息
    tensors = {}

    # 为所有权重分配索引
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

    # 为所有输入分配索引
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
            # 计算 size
            size = 1
            for dim in shape:
                size *= dim
            tensors[next_idx]["size"] = size
            next_idx += 1

    # 为所有输出分配索引
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

    # 为所有中间张量分配索引
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
            "input_indices": [tensor_index_map.get(name, -1) for name in node.input],
            "output_indices": [tensor_index_map.get(name, -1) for name in node.output],
            "state": "created",
            "pass_flags": {},
            "input_details": [],
            "output_details": [],
        }

        # 填充 input_details
        for inp_name in node.input:
            shape = get_tensor_shape(graph, inp_name)
            size = 1
            for dim in shape:
                size *= dim
            op_info["input_details"].append({
                "index": len(op_info["input_details"]),
                "name": inp_name,
                "shape": shape,
                "size": size,
                "scale": 1.0,
                "zero_point": 0,
            })
        # 填充 output_details
        for out_name in node.output:
            shape = get_tensor_shape(graph, out_name)
            size = 1
            for dim in shape:
                size *= dim
            op_info["output_details"].append({
                "index": len(op_info["output_details"]),
                "name": out_name,
                "shape": shape,
                "size": size,
                "scale": 1.0,
                "zero_point": 0,
            })

        # 映射到 tinymlc 算子名
        if node.op_type in OP_MAP:
            op_info["op_name"] = OP_MAP[node.op_type]
            op_info["state"] = "translated"
            op_info["pass_flags"]["onnx_parse"] = "success"
        else:
            warning(f"未知 ONNX 算子: {node.op_type}")
            op_info["state"] = "created"
            op_info["pass_flags"]["unknown"] = "needs_implementation"

        # 处理特殊算子
        # Gemm 就是 FULLY_CONNECTED
        if node.op_type == "Gemm":
            op_info["data_input_idx"] = op_info["input_indices"][0]
            op_info["fc_weights_idx"] = op_info["input_indices"][1]
            if len(node.input) >= 3:
                op_info["fc_bias_idx"] = op_info["input_indices"][2]
            else:
                op_info["fc_bias_idx"] = None

            # Gemm: input, weights, bias
            # 记录权重和 bias 名称
            if len(node.input) >= 3:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = node.input[2]
            elif len(node.input) >= 2:
                op_info["weights_name"] = node.input[1]
                op_info["bias_name"] = None

        if node.op_type == "Conv":
            op_info["data_input_idx"] = op_info["input_indices"][0]  # 第一个输入是数据
            op_info["conv_weights_idx"] = op_info["input_indices"][1]  # 第二个是权重
            op_info["conv_bias_idx"] = op_info["input_indices"][2] if len(node.input) >= 3 else None

            # 从权重形状推断 kernel_shape
            weights_name = node.input[1]
            kernel_h, kernel_w = 1, 1
            if weights_name in weights:
                weight_tensor = weights[weights_name]
                if len(weight_tensor.shape) >= 4:
                    kernel_h = weight_tensor.shape[2]
                    kernel_w = weight_tensor.shape[3]

            # 获取输入形状
            input_shape = get_tensor_shape(graph, node.input[0])
            output_shape = get_tensor_shape(graph, node.output[0])

            input_h = input_shape[2] if len(input_shape) >= 4 else 1
            input_w = input_shape[3] if len(input_shape) >= 4 else 1
            input_c = input_shape[1] if len(input_shape) >= 4 else 1

            output_h = output_shape[2] if len(output_shape) >= 4 else 1
            output_w = output_shape[3] if len(output_shape) >= 4 else 1
            output_c = output_shape[1] if len(output_shape) >= 4 else 1

            # 从 attributes 提取 stride 和 padding
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
            # FIXME 临时调试
            op_info["softmax_size"] = 10  # 从输出形状获取

        if node.op_type == "Reshape":
            # 目标形状在 inputs[1]
            target_shape_name = node.input[1]
            target_shape = weights.get(target_shape_name)
            if target_shape is not None:
                op_info["reshape_target_shape"] = target_shape.tolist()

        # 提取卷积参数（从 attributes）
        # 提取卷积参数
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

            op_info["pool_params"] = {
                "stride_h": strides[0] if len(strides) >= 2 else strides[0],
                "stride_w": strides[1] if len(strides) >= 2 else strides[0],
                "padding_h": pads[0] if len(pads) >= 2 else 0,
                "padding_w": pads[1] if len(pads) >= 2 else 0,
                "kernel_h": kernel_shape[0] if len(kernel_shape) >= 2 else kernel_shape[0],
                "kernel_w": kernel_shape[1] if len(kernel_shape) >= 2 else kernel_shape[0],
            }

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "weights": weights,
        "tensors": tensors,
    }
