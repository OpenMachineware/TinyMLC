#!/usr/bin/env python3
"""TFLite 模型解析器"""

import tensorflow as tf


# DELEGATE 是 TFLite 中一个特殊算子，它代表"委托给硬件加速器"
# （如 GPU、NPU、XNNPACK），现阶段不做，以后支持硬件加速器再做
IGNORED_OPS = {"DELEGATE"}


def _match_lstm_inputs(op, tensor_map):
    """通过名称匹配 LSTM 算子的所有输入

        返回:
            dict: {
                "input_idx": int,                    # 输入数据张量索引
                "input_weights": list[int],          # 4个输入门权重索引 (i,f,g,o)
                "recurrent_weights": list[int],      # 4个递归门权重索引 (i,f,g,o)
                "biases": list[int],                 # 4个偏置索引 (i,f,g,o)
            }
            缺失的项为 None
        """
    result = {
        "input_idx": None,
        "input_weights": [None, None, None, None],
        "recurrent_weights": [None, None, None, None],
        "biases": [None, None, None, None],
    }

    # 名称到门索引的映射
    gate_keywords = {
        0: ["input", "i"],  # 输入门
        1: ["forget", "f"],  # 遗忘门
        2: ["cell", "c", "g"],  # 候选门
        3: ["output", "o"],  # 输出门
    }

    for idx in op["inputs"]:
        if idx == -1:
            continue

        tensor_info = tensor_map.get(idx, {})
        name = tensor_info.get("name", "").lower()

        # 1. 识别输入数据张量
        if name == "input" or name.endswith("/input") or "lstm_input" in name:
            result["input_idx"] = idx
            continue

        # 2. 识别权重（input_weights 或 recurrent_weights）
        if "weight" in name or "kernel" in name:
            # 判断是输入权重还是递归权重
            is_recurrent = "recurrent" in name or "rec" in name

            for gate_idx, keywords in gate_keywords.items():
                matched = False
                for kw in keywords:
                    if kw in name:
                        matched = True
                        break
                if matched:
                    if is_recurrent:
                        result["recurrent_weights"][gate_idx] = idx
                    else:
                        result["input_weights"][gate_idx] = idx
                    break

        # 3. 识别偏置
        if "bias" in name:
            for gate_idx, keywords in gate_keywords.items():
                matched = False
                for kw in keywords:
                    if kw in name:
                        matched = True
                        break
                if matched:
                    result["biases"][gate_idx] = idx
                    break

    # 过滤 None，但保留部分匹配的结果
    result["input_weights"] = [i for i in result["input_weights"] if
                               i is not None]
    result["recurrent_weights"] = [i for i in result["recurrent_weights"] if
                                   i is not None]
    result["biases"] = [i for i in result["biases"] if i is not None]

    return result


def _fallback_lstm_positional(op):
    """当名称匹配失败时，使用 TFLite 标准位置匹配"""
    # TFLite LSTM 标准位置（参考官方文档）
    # inputs[0] = input
    # inputs[1-4] = input_weights (i,f,g,o)
    # inputs[5-8] = recurrent_weights (i,f,g,o)
    # inputs[9-12] = cell_weights (通常不用)
    # inputs[13-16] = bias (i,f,g,o)
    result = {
        "input_idx": None,
        "input_weights": [],
        "recurrent_weights": [],
        "biases": [],
    }

    inputs = op["inputs"]

    # inputs[0] = 输入数据
    if len(inputs) > 0:
        result["input_idx"] = inputs[0]

    # inputs[1-4] = 输入权重 (i, f, g, o)
    if len(inputs) >= 5:
        result["input_weights"] = [inputs[1], inputs[2], inputs[3], inputs[4]]
        result["input_weights"] = [i for i in result["input_weights"] if
                                   i != -1]

    # inputs[5-8] = 递归权重 (i, f, g, o)
    if len(inputs) >= 9:
        result["recurrent_weights"] = [inputs[5], inputs[6], inputs[7],
                                       inputs[8]]
        result["recurrent_weights"] = [i for i in result["recurrent_weights"] if
                                       i != -1]

    # inputs[13-16] = 偏置 (i, f, g, o)
    if len(inputs) >= 17:
        result["biases"] = [inputs[13], inputs[14], inputs[15], inputs[16]]
        result["biases"] = [i for i in result["biases"] if i != -1]

    return result



def _get_tensor_shape(tensor_info, default=None):
    """安全获取张量形状"""
    shape = tensor_info.get("shape", [])
    if shape and all(s > 0 for s in shape):
        return shape
    return default


def parse_model(interpreter):
    """解析 TFLite 模型，返回算子列表和张量信息"""
    interpreter.allocate_tensors()

    # 获取张量详情
    tensor_details = interpreter.get_tensor_details()
    tensor_map = {}
    for t in tensor_details:
        quant = t.get("quantization", (None, None))
        scale = quant[0] if quant[0] is not None else 1.0
        zero_point = quant[1] if quant[1] is not None else 0

        tensor_map[t["index"]] = {
            "name": t["name"],
            "shape": list(t["shape"]),
            "dtype": str(t["dtype"]),
            "size": t["shape"].num_elements() if hasattr(t["shape"], "num_elements") else 1,
            "scale": float(scale),
            "zero_point": int(zero_point),
        }

    # 获取输入输出
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 解析所有算子
    ops = []
    for op in interpreter._get_ops_details():
        op_info = {
            "index": op["index"],
            "op_name": op["op_name"],
            "inputs": list(op["inputs"]),
            "outputs": list(op["outputs"]),
            "input_details": [],
            "output_details": [],
            "state": "created",
            "pass_flags": {},
        }

        # ========== LSTM 算子处理 ==========
        if op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            # 1. 先尝试名称匹配
            matched = _match_lstm_inputs(op, tensor_map)

            # 2. 如果名称匹配失败，回退到位置匹配
            if matched["input_idx"] is None or not matched["input_weights"]:
                print("提示: 名称匹配失败，使用位置匹配（TFLite 标准格式）")
                matched = _fallback_lstm_positional(op)

            # 3. 验证匹配结果
            if matched["input_idx"] is None:
                op_info["state"] = "invalid"
                op_info["pass_flags"]["lstm_match"] = "failed_input"
                print("错误: LSTM 算子无法识别输入数据张量")
                ops.append(op_info)
                continue

            if not matched["input_weights"] and not matched["recurrent_weights"]:
                op_info["state"] = "invalid"
                op_info["pass_flags"]["lstm_match"] = "failed_weights"
                print("错误: LSTM 算子无法识别任何权重张量")
                ops.append(op_info)
                continue

            # 4. 获取形状参数
            # 此时 matched["input_idx"] 一定不是 None
            input_shape = tensor_map.get(matched["input_idx"], {}).get("shape", [])
            if len(input_shape) >= 3:
                time_steps = input_shape[0]
                batch_size = input_shape[1]
                input_size = input_shape[2]
            else:
                time_steps, batch_size, input_size = 1, 1, 1

            output_idx = op["outputs"][0]
            output_shape = tensor_map.get(output_idx, {}).get("shape", [])
            if len(output_shape) >= 3:
                hidden_size = output_shape[2]
            else:
                hidden_size = 1

            # 提取量化参数
            input_scales = []
            for idx in matched["input_weights"]:
                scale = tensor_map.get(idx, {}).get("scale", 0.01)
                input_scales.append(scale)

            recurrent_scales = []
            for idx in matched["recurrent_weights"]:
                scale = tensor_map.get(idx, {}).get("scale", 0.01)
                recurrent_scales.append(scale)

            bias_scales = []
            for idx in matched["biases"]:
                scale = tensor_map.get(idx, {}).get("scale", 1e-5)
                bias_scales.append(scale)

            # 存储 LSTM 参数
            op_info["lstm_params"] = {
                "time_steps": time_steps,
                "batch_size": batch_size,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "input_scales": input_scales,
                "recurrent_scales": recurrent_scales,
                "bias_scales": bias_scales,
                "input_scale": input_scales[0] if input_scales else 0.01,
                "input_zp": 0,
            }

            op_info["lstm_weight_indices"] = {
                "input": matched["input_weights"],
                "recurrent": matched["recurrent_weights"],
                "bias": matched["biases"],
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["lstm_match"] = "success"
            if len(matched["input_weights"]) != 4:
                op_info["pass_flags"]["lstm_input_weights_warning"] = f"found_{len(matched['input_weights'])}_expected_4"

        # ========== FC 算子处理 ==========
        elif op["op_name"] == "FULLY_CONNECTED":
            # 1. 先尝试名称匹配
            weights_idx = None
            bias_idx = None
            input_idx = None

            for idx in op["inputs"]:
                if idx == -1:
                    continue
                tensor_info = tensor_map.get(idx, {})
                name = tensor_info.get("name", "").lower()

                if "matmul" in name or "weight" in name or "kernel" in name:
                    weights_idx = idx
                    op_info["pass_flags"]["fc_weights_match"] = "name"
                elif "bias" in name:
                    bias_idx = idx
                    op_info["pass_flags"]["fc_bias_match"] = "name"
                else:
                    input_idx = idx
                    op_info["pass_flags"]["fc_input_match"] = "name"

            # 2. 如果名称匹配失败，回退到位置匹配（TFLite 标准格式）
            if weights_idx is None or bias_idx is None or input_idx is None:
                print("提示: FC 名称匹配失败，使用位置匹配（TFLite 标准格式）")
                # TFLite FC 标准位置：inputs[0]=input, inputs[1]=weight, inputs[2]=bias
                if len(op["inputs"]) >= 3:
                    if input_idx is None:
                        input_idx = op["inputs"][0]
                        op_info["pass_flags"]["fc_input_match"] = "position"
                    if weights_idx is None:
                        weights_idx = op["inputs"][1]
                        op_info["pass_flags"]["fc_weights_match"] = "position"
                    if bias_idx is None:
                        bias_idx = op["inputs"][2]
                        op_info["pass_flags"]["fc_bias_match"] = "position"

            # 3. 验证匹配结果
            if weights_idx is not None and bias_idx is not None and input_idx is not None:
                op_info["fc_weights_idx"] = weights_idx
                op_info["fc_bias_idx"] = bias_idx
                op_info["fc_input_idx"] = input_idx
                op_info["state"] = "translated"
                op_info["pass_flags"]["fc_match"] = "success"
            else:
                op_info["state"] = "invalid"
                op_info["pass_flags"]["fc_match"] = "failed"
                print(
                    f"错误: FC 算子匹配失败: weights={weights_idx}, bias={bias_idx}, input={input_idx}")

        # ========== Softmax 算子处理 ==========
        elif op["op_name"] == "SOFTMAX":
            op_info["state"] = "translated"
            op_info["pass_flags"]["softmax_check"] = "no_weights_needed"

        # ========== Reshape 算子处理 ==========
        elif op["op_name"] == "RESHAPE":
            op_info["state"] = "translated"
            op_info["pass_flags"]["reshape_check"] = "pending"

        # ========== DELEGATE 算子跳过 ==========
        elif op["op_name"] == "DELEGATE":
            continue

        # ========== 未知算子 ==========
        else:
            op_info["state"] = "created"
            op_info["pass_flags"]["unknown"] = "needs_implementation"

        # 获取输入张量详细信息（所有算子都执行）
        for inp_idx in op["inputs"]:
            if inp_idx != -1:
                info = tensor_map.get(inp_idx, {})
                op_info["input_details"].append({
                    "index": inp_idx,
                    "name": info.get("name", "unknown"),
                    "shape": info.get("shape", []),
                    "size": info.get("size", 0),
                    "scale": info.get("scale", 1.0),
                    "zero_point": info.get("zero_point", 0),
                })

        # 获取输出张量详细信息（所有算子都执行）
        for out_idx in op["outputs"]:
            if out_idx != -1:
                info = tensor_map.get(out_idx, {})
                op_info["output_details"].append({
                    "index": out_idx,
                    "name": info.get("name", "unknown"),
                    "shape": info.get("shape", []),
                    "size": info.get("size", 0),
                    "scale": info.get("scale", 1.0),
                    "zero_point": info.get("zero_point", 0),
                })

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "tensors": tensor_map,
    }
