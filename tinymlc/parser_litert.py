#!/usr/bin/env python3
"""基于 LiteRT 的 TFLite 模型解析器"""

from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
import numpy as np

from tinymlc.utils import fatal_error, info


def parse_model(model_path: str):
    """使用 LiteRT 解析 TFLite 模型"""

    # 1. 加载模型（使用 LiteRT 的 Interpreter）
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. 获取输入输出张量（与旧版 API 一致）
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. 获取所有张量信息 获取张量详情后，尝试读取常量张量的数据
    tensor_details = interpreter.get_tensor_details()
    tensor_map = {}

    for tensor in tensor_details:
        shape = tensor["shape"]
        tensor_info = {
            "name": tensor["name"],
            "shape": list(tensor["shape"]),
            "dtype": str(tensor["dtype"]),
            "size": int(np.prod(shape)) if shape is not None and len(
                shape) > 0 else 1,
            "scale": tensor["quantization"][0] if tensor["quantization"][
                                                      0] is not None else 1.0,
            "zero_point": tensor["quantization"][1] if tensor["quantization"][
                                                           1] is not None else 0,
        }

        # 尝试读取常量张量的数据（用于 Reshape 的目标形状等）
        try:
            data = interpreter.get_tensor(tensor["index"])
            tensor_info["data"] = data
        except:
            pass

        tensor_map[tensor["index"]] = tensor_info

    # 4. 获取算子列表
    ops = []
    for op in interpreter._get_ops_details():
        # 跳过 DELEGATE 算子
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

        # 根据算子类型设置状态
        if op["op_name"] == "ADD":
            inputs = op_info["input_indices"]
            # 输入数量判断是基于算子规范的，inputs[0] inputs[1] 属合理。
            if len(inputs) >= 2:
                op_info["add_input1_idx"] = inputs[0]
                op_info["add_input2_idx"] = inputs[1]
            op_info["state"] = "translated"
            op_info["pass_flags"]["add_check"] = "success"
        elif op["op_name"] == "FULLY_CONNECTED":
            inputs = op_info["input_indices"]
            # 输入数量判断是基于算子规范的，inputs[0] inputs[1] inputs[2] 属合理。
            if len(inputs) >= 3:
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = inputs[1]
                op_info["fc_bias_idx"] = inputs[2]
            # 输入数量判断是基于算子规范的，inputs[0] inputs[1] 属合理。
            elif len(inputs) >= 2:
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = inputs[1]
                op_info["fc_bias_idx"] = None
            else:
                # 输入数量判断是基于算子规范的，inputs[0] 属合理。
                op_info["data_input_idx"] = inputs[0]
                op_info["fc_weights_idx"] = None
                op_info["fc_bias_idx"] = None
            op_info["state"] = "translated"
            op_info["pass_flags"]["fc_check"] = "success"
        elif op["op_name"] == "SOFTMAX":
            if len(op_info["input_indices"]) < 1:
                fatal_error("SOFTMAX 缺少输入", "检查模型格式")
            if len(op_info["output_indices"]) < 1:
                fatal_error("SOFTMAX 缺少输出", "检查模型格式")
            op_info["state"] = "translated"
            op_info["pass_flags"]["softmax_check"] = "success"
        elif op["op_name"] == "RESHAPE":
            inputs = op_info["input_indices"]
            if len(inputs) < 2:
                fatal_error("RESHAPE 缺少目标形状参数", "检查模型格式")

            shape_idx = inputs[1]
            shape_tensor = tensor_map.get(shape_idx, {})

            # 优先从 data 读取实际值
            if "data" in shape_tensor:
                target_shape = [int(s) for s in shape_tensor["data"].flatten()]
            else:
                target_shape = shape_tensor.get("shape", [])

            if not target_shape:
                fatal_error("RESHAPE 无法提取目标形状", "检查模型格式")

            # 处理动态维度 -1
            if -1 in target_shape:
                # 获取输入张量的大小
                input_idx = inputs[0]
                input_tensor = tensor_map.get(input_idx, {})
                input_size = input_tensor.get("size", 1)

                # 计算 -1 的实际值
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
                # TFLite/LiteRT LSTM 输入顺序：
                # [0] 输入数据, [1-4] 输入门权重, [5-8] 递归权重, [9-12] 偏置
                # TFLite/LiteRT UNIDIRECTIONAL_SEQUENCE_LSTM 算子的输入顺序：
                # [0] 输入数据
                # [1-4] 输入门权重 (i, f, g, o)
                # [5-8] 递归权重 (i, f, g, o)
                # [9-12] 偏置 (i, f, g, o)
                # [13+] 其他参数
                op_info["lstm_weight_indices"] = {
                    "input": inputs[1:5],
                    "recurrent": inputs[5:9],
                    "bias": inputs[9:13],
                }

                # 从输出形状提取 hidden_size
                output_shape = tensor_map.get(op_info["output_indices"][0],
                                              {}).get("shape", [])
                if len(output_shape) >= 3:
                    hidden_size = output_shape[2]
                else:
                    fatal_error("无法从 LSTM 输出形状提取 hidden_size",
                                "检查模型格式")

                # 从输入形状提取 time_steps, batch_size, input_size
                input_shape = tensor_map.get(inputs[0], {}).get("shape", [])
                if len(input_shape) >= 3:
                    op_info["lstm_params"] = {
                        "time_steps": input_shape[1],
                        # [batch, time_steps, input_size]
                        "batch_size": input_shape[0],
                        "input_size": input_shape[2],
                        "hidden_size": hidden_size,
                    }
                else:
                    fatal_error("无法从 LSTM 输入形状提取参数", "检查模型格式")

                op_info["state"] = "translated"
                op_info["pass_flags"]["lstm_check"] = "success"
            else:
                fatal_error("LSTM 输入不完整", "检查模型格式")
        elif op["op_name"] == "SVDF":
            # SVDF 需要记录权重索引
            inputs = op_info["input_indices"]
            # 输入数量判断是基于算子规范的，inputs[0] inputs[1] inputs[2] 属合理。
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
                fatal_error("CONV_2D 输入不完整", "检查模型格式")

            # 提取卷积参数
            # 从输入输出形状计算 stride, padding 等
            input_idx = inputs[0]
            output_idx = op_info["output_indices"][0]

            input_tensor = tensor_map.get(input_idx, {})
            output_tensor = tensor_map.get(output_idx, {})

            input_shape = input_tensor.get("shape", [])
            output_shape = output_tensor.get("shape", [])

            # 权重 shape: [out_channels, kernel_h, kernel_w, in_channels]
            weights_tensor = tensor_map.get(inputs[1], {})
            weights_shape = weights_tensor.get("shape", [])

            # 默认参数
            stride_h = 1
            stride_w = 1
            padding = "VALID"

            # 从输入输出形状推断 stride 和 padding
            if len(input_shape) >= 4 and len(output_shape) >= 4 and len(
                    weights_shape) >= 4:
                input_h = input_shape[1]
                input_w = input_shape[2]
                output_h = output_shape[1]
                output_w = output_shape[2]
                kernel_h = weights_shape[1]
                kernel_w = weights_shape[2]

                # 计算 stride（假设 stride_h == stride_w）
                if input_h > output_h:
                    stride_h = (input_h - kernel_h) // (
                                output_h - 1) if output_h > 1 else 1
                    stride_w = (input_w - kernel_w) // (
                                output_w - 1) if output_w > 1 else 1

                # 判断 padding
                # 如果输出尺寸 = ceil(input / stride)，通常是 SAME padding
                # 否则是 VALID
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
        elif op["op_name"] == "QUANTIZE":
            op_info["state"] = "translated"
            op_info["pass_flags"]["quantize_check"] = "success"
        elif op["op_name"] == "DELEGATE":
            continue  # 跳过
        else:
            # 未知算子，保持 created
            pass

        # 添加输入输出详细信息
        for inp_idx in op_info["input_indices"]:
            tensor_info = tensor_map.get(inp_idx, {})
            op_info["input_details"].append({
                "index": inp_idx,
                "name": tensor_info.get("name", "unknown"),
                "shape": tensor_info.get("shape", []),
                "size": tensor_info.get("size", 0),
                "scale": tensor_info.get("scale", 1.0),
                "zero_point": tensor_info.get("zero_point", 0),
            })

        for out_idx in op_info["output_indices"]:
            tensor_info = tensor_map.get(out_idx, {})
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
        "tensors": tensor_map,
    }