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
    "SVDF": "SVDF",
}


def get_tensor_shape(graph, name, initializer_map=None):
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
    # 检查 initializer（QDQ 模型中的量化权重）
    if initializer_map is not None and name in initializer_map:
        return list(initializer_map[name].shape)
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

    # 构建 initializer 名称到数组的映射（用于 QDQ 节点）
    initializer_map = {}
    for init in graph.initializer:
        initializer_map[init.name] = numpy_helper.to_array(init)

    # 为 QDQ 节点的输出分配索引（这些张量不在 value_info 中）
    for node in graph.node:
        for out_name in node.output:
            if out_name not in tensor_index_map:
                # 获取形状：如果是量化权重，从 initializer 读取；否则使用输入形状
                shape = []
                if out_name in initializer_map:
                    shape = list(initializer_map[out_name].shape)
                elif node.input:
                    inp_name = node.input[0]
                    if inp_name in tensor_index_map:
                        shape = tensors[tensor_index_map[inp_name]].get("shape", [])
                    else:
                        shape = get_tensor_shape(graph, inp_name, initializer_map)
                
                size = 1
                for dim in shape:
                    size *= dim
                
                tensor_index_map[out_name] = next_idx
                tensors[next_idx] = {
                    "name": out_name,
                    "shape": shape,
                    "dtype": "int8" if node.op_type == "QuantizeLinear" else "float32",
                    "size": size,
                    "scale": 1.0,
                    "zero_point": 0,
                }
                next_idx += 1

    # 解析 QDQ 量化参数（QuantizeLinear/DequantizeLinear）

    # QDQ 映射表：量化节点输出 -> 原始输入
    # 用于将计算算子的输入从 QuantizeLinear/DequantizeLinear 输出替换为原始张量
    qdq_map = {}

    # 遍历所有节点，提取量化参数和构建映射表
    for node in graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            input_name = node.input[0]
            output_name = node.output[0]
            scale_name = node.input[1]
            
            # 构建映射：量化节点输出 -> 原始输入
            qdq_map[output_name] = input_name
            # 递归映射：如果输入也是量化节点的输出，继续映射
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

    # 3. # 提取权重和 scale
    weights = {}
    scales = {}
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        weights[init.name] = tensor

        # 计算 scale
        min_val = tensor.min()
        max_val = tensor.max()
        scale = max(abs(min_val), abs(max_val)) / 127.0
        if scale == 0:
            scale = 1.0
        scales[init.name] = scale

    # 4. 解析算子
    ops = []
    # QDQ 模型中的伪算子，不需要生成代码
    skip_ops = {"QuantizeLinear", "DequantizeLinear", "Constant"}
    
    for node in graph.node:
        # 跳过伪算子
        if node.op_type in skip_ops:
            continue
            
        # 使用 QDQ 映射表替换输入：将 DequantizeLinear 输出替换为原始输入
        mapped_inputs = []
        for inp_name in node.input:
            # 如果输入是 DequantizeLinear 的输出，替换为原始输入
            if inp_name in qdq_map:
                mapped_inputs.append(qdq_map[inp_name])
            else:
                mapped_inputs.append(inp_name)
            
        op_info = {
            "index": len(ops),
            "op_name": node.op_type,
            "inputs": mapped_inputs,
            "outputs": list(node.output),
            "input_indices": [tensor_index_map.get(name, -1) for name in mapped_inputs],
            "output_indices": [tensor_index_map.get(name, -1) for name in node.output],
            "state": "created",
            "pass_flags": {},
            "input_details": [],
            "output_details": [],
        }

        # 填充 input_details
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
        # 填充 output_details
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
            weights_name = node.input[1]
            op_info["fc_scale"] = scales.get(weights_name, 0.01)
            op_info["data_input_idx"] = op_info["input_indices"][0]
            op_info["fc_weights_idx"] = op_info["input_indices"][1]
            op_info["fc_bias_idx"] = op_info["input_indices"][2] if len(node.input) >= 3 else None
            op_info["weights_name"] = node.input[1]
            op_info["bias_name"] = node.input[2] if len(node.input) >= 3 else None

            # 提取输出 scale（如果存在）
            output_name = node.output[0]
            output_tensor = weights.get(output_name)
            if output_tensor is not None:
                min_val = output_tensor.min()
                max_val = output_tensor.max()
                output_scale = max(abs(min_val), abs(max_val)) / 127.0
                if output_scale == 0:
                    output_scale = 1.0
            else:
                # 如果 initializer 里没有，用默认值
                output_scale = 1.0
            op_info["fc_output_scale"] = output_scale

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

            input_shape = get_tensor_shape(graph, node.input[0], initializer_map)
            output_shape = get_tensor_shape(graph, node.output[0], initializer_map)
            
            op_info["pool_params"] = {
                "input_h": input_shape[2] if len(input_shape) >= 4 else 1,
                "input_w": input_shape[3] if len(input_shape) >= 4 else 1,
                "input_c": input_shape[1] if len(input_shape) >= 4 else 1,
                "output_h": output_shape[2] if len(output_shape) >= 4 else 1,
                "output_w": output_shape[3] if len(output_shape) >= 4 else 1,
                "output_c": output_shape[1] if len(output_shape) >= 4 else 1,
                "pool_size_h": kernel_shape[0] if len(kernel_shape) >= 2 else kernel_shape[0],
                "pool_size_w": kernel_shape[1] if len(kernel_shape) >= 2 else kernel_shape[0],
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
            
            input_shape = get_tensor_shape(graph, node.input[0], initializer_map)
            output_shape = get_tensor_shape(graph, node.output[0], initializer_map)
            
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

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "weights": weights,
        "tensors": tensors,
    }
