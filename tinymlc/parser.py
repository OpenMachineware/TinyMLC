#!/usr/bin/env python3
"""TFLite 模型解析器"""

import tensorflow as tf
from tinymlc.utils import fatal_error

# DELEGATE 是 TFLite 中一个特殊算子，它代表"委托给硬件加速器"
# （如 GPU、NPU、XNNPACK），现阶段不做，以后支持硬件加速器再做
IGNORED_OPS = {"DELEGATE"}


def match_tensor_by_name(op, tensor_map, roles):
    """通用张量名称匹配

    Args:
        op: 算子信息（包含 inputs 列表）
        tensor_map: 张量索引到信息的映射
        roles: 角色定义列表，每个元素是 (role_name, [keyword_list])
               例如: [("weights", ["matmul", "weight", "kernel"]),
                     ("bias", ["bias"]),
                     ("input", [])]  # 空列表表示剩余的所有张量

    Returns:
        dict: {role_name: idx} 匹配结果
    """
    result = {}
    used_indices = set()

    # 第一步：按关键词匹配
    for role_name, keywords in roles:
        if not keywords:  # 空列表表示稍后处理剩余项
            continue
        for idx in op["inputs"]:
            if idx == -1 or idx in used_indices:
                continue
            tensor_info = tensor_map.get(idx, {})
            name = tensor_info.get("name", "").lower()
            for kw in keywords:
                if kw in name:
                    result[role_name] = idx
                    used_indices.add(idx)
                    break
            if role_name in result:
                break

    # 第二步：处理没有关键词的角色（匹配剩余张量）
    for role_name, keywords in roles:
        if keywords:  # 跳过已有关键词的
            continue
        for idx in op["inputs"]:
            if idx == -1 or idx in used_indices:
                continue
            result[role_name] = idx
            used_indices.add(idx)
            break

    return result


def match_tensor_by_position(op, positions):
    """通用位置匹配（回退方案）

    Args:
        op: 算子信息
        positions: 位置定义列表，每个元素是 (role_name, position_index)
                   例如: [("input", 0), ("weights", 1), ("bias", 2)]

    Returns:
        dict: {role_name: idx} 匹配结果，不存在的索引为 None
    """
    result = {}
    inputs = op["inputs"]
    for role_name, pos in positions:
        if pos < len(inputs):
            idx = inputs[pos]
            result[role_name] = idx if idx != -1 else None
        else:
            result[role_name] = None
    return result


def _merge_match_results(name_match, pos_match, roles):
    """合并名称匹配和位置匹配结果，名称匹配优先"""
    result = {}
    for role_name, _ in roles:
        if name_match.get(role_name) is not None:
            result[role_name] = name_match[role_name]
        elif pos_match.get(role_name) is not None:
            result[role_name] = pos_match[role_name]
        else:
            result[role_name] = None
    return result


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
            # 定义角色（12个）
            roles = [
                ("input", ["input", "lstm_input"]),
                ("input_weights_i", ["input_gate", "wi"]),
                ("input_weights_f", ["forget_gate", "wf"]),
                ("input_weights_g", ["cell_gate", "wg"]),
                ("input_weights_o", ["output_gate", "wo"]),
                ("recurrent_weights_i", ["recurrent", "input_gate", "ri"]),
                ("recurrent_weights_f", ["recurrent", "forget_gate", "rf"]),
                ("recurrent_weights_g", ["recurrent", "cell_gate", "rg"]),
                ("recurrent_weights_o", ["recurrent", "output_gate", "ro"]),
                ("bias_i", ["bias", "input"]),
                ("bias_f", ["bias", "forget"]),
                ("bias_g", ["bias", "cell"]),
                ("bias_o", ["bias", "output"]),
            ]

            # 名称匹配
            name_match = match_tensor_by_name(op, tensor_map, roles)

            # 位置匹配（回退）- TFLite 标准位置
            positions = [
                ("input", 0),
                ("input_weights_i", 1), ("input_weights_f", 2),
                ("input_weights_g", 3), ("input_weights_o", 4),
                ("recurrent_weights_i", 5), ("recurrent_weights_f", 6),
                ("recurrent_weights_g", 7), ("recurrent_weights_o", 8),
                ("bias_i", 13), ("bias_f", 14), ("bias_g", 15), ("bias_o", 16),
            ]
            pos_match = match_tensor_by_position(op, positions)

            # 合并结果
            matched = _merge_match_results(name_match, pos_match, roles)

            # 验证
            if matched["input"] is None:
                fatal_error("无法识别输入数据张量",
                            "转换时添加 converter._experimental_preserve_all_tensors = True")

            # 组装权重列表（按顺序 i, f, g, o）
            input_weights = [
                matched.get("input_weights_i"), matched.get("input_weights_f"),
                matched.get("input_weights_g"), matched.get("input_weights_o")
            ]
            input_weights = [i for i in input_weights if i is not None]

            recurrent_weights = [
                matched.get("recurrent_weights_i"),
                matched.get("recurrent_weights_f"),
                matched.get("recurrent_weights_g"),
                matched.get("recurrent_weights_o")
            ]
            recurrent_weights = [i for i in recurrent_weights if i is not None]

            biases = [
                matched.get("bias_i"), matched.get("bias_f"),
                matched.get("bias_g"), matched.get("bias_o")
            ]
            biases = [i for i in biases if i is not None]

            if not input_weights and not recurrent_weights:
                fatal_error("无法识别任何权重张量",
                            "检查模型是否为标准 TFLite LSTM 格式")

            # 获取形状参数
            input_shape = tensor_map.get(matched["input"], {}).get("shape", [])
            if len(input_shape) >= 3:
                batch_size = input_shape[0]
                time_steps = input_shape[1]
                input_size = input_shape[2]
            else:
                batch_size, time_steps, input_size = 1, 1, 1

            output_idx = op["outputs"][0]
            output_shape = tensor_map.get(output_idx, {}).get("shape", [])
            if len(output_shape) >= 3:
                hidden_size = output_shape[2]
            else:
                hidden_size = 1

            # 提取量化参数（scale）
            input_scales = []
            for idx in input_weights:
                scale = tensor_map.get(idx, {}).get("scale", 0.01)
                input_scales.append(scale)

            recurrent_scales = []
            for idx in recurrent_weights:
                scale = tensor_map.get(idx, {}).get("scale", 0.01)
                recurrent_scales.append(scale)

            bias_scales = []
            for idx in biases:
                scale = tensor_map.get(idx, {}).get("scale", 1e-5)
                bias_scales.append(scale)

            # 存储参数
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
                "input": input_weights,
                "recurrent": recurrent_weights,
                "bias": biases,
            }

            op_info["state"] = "translated"
            op_info["pass_flags"]["lstm_match"] = "success"

        # ========== FC 算子处理 ==========
        elif op["op_name"] == "FULLY_CONNECTED":
            # 定义角色
            roles = [
                ("weights", ["matmul", "weight", "kernel"]),
                ("bias", ["bias"]),
                ("input", []),  # 剩余的张量就是输入
            ]

            # 名称匹配
            name_match = match_tensor_by_name(op, tensor_map, roles)

            # 位置匹配（回退）
            positions = [("input", 0), ("weights", 1), ("bias", 2)]
            pos_match = match_tensor_by_position(op, positions)

            # 合并结果
            matched = _merge_match_results(name_match, pos_match, roles)

            # 验证
            if None in [matched.get("weights"), matched.get("bias"),
                        matched.get("input")]:
                fatal_error(f"FC 匹配失败: {matched}",
                            "检查模型是否为标准 TFLite FC 格式，或转换时保留张量名称")

            op_info["fc_weights_idx"] = matched["weights"]
            op_info["fc_bias_idx"] = matched["bias"]
            op_info["fc_input_idx"] = matched["input"]
            op_info["state"] = "translated"
            op_info["pass_flags"]["fc_match"] = "success"
            op_info["pass_flags"]["fc_weights_method"] = "name" if name_match.get(
                "weights") else "position"

        # ========== Softmax 算子处理 ==========
        elif op["op_name"] == "SOFTMAX":
            # Softmax 通常没有额外参数，只需检查输入输出
            if len(op["inputs"]) < 1 or op["inputs"][0] == -1:
                fatal_error("Softmax 缺少输入张量",
                            "检查模型转换是否完整")
            if len(op["outputs"]) < 1 or op["outputs"][0] == -1:
                fatal_error("Softmax 缺少输出张量",
                            "检查模型转换是否完整")

            op_info["state"] = "translated"
            op_info["pass_flags"]["softmax_check"] = "success"

        # ========== Reshape 算子处理 ==========
        elif op["op_name"] == "RESHAPE":
            # Reshape 需要目标形状参数（通常在 inputs[1]）
            if len(op["inputs"]) < 2 or op["inputs"][1] == -1:
                fatal_error("Reshape 缺少目标形状参数",
                            "检查模型转换是否完整")

            # 获取目标形状张量
            shape_idx = op["inputs"][1]
            shape_tensor = tensor_map.get(shape_idx, {})
            target_shape = shape_tensor.get("shape", [])

            if not target_shape or target_shape[0] == 0:
                fatal_error(f"Reshape 目标形状无效: {target_shape}",
                            "检查 Reshape 算子参数")

            op_info["reshape_target_shape"] = target_shape
            op_info["state"] = "translated"
            op_info["pass_flags"]["reshape_check"] = "success"

        # ========== ADD 算子 ==========
        elif op["op_name"] == "ADD":
            # ADD 算子：两个输入，一个输出
            if len(op["inputs"]) < 2:
                fatal_error("ADD 输入不足", "检查模型是否完整")

            op_info["add_inputs"] = [op["inputs"][0], op["inputs"][1]]
            op_info["add_output"] = op["outputs"][0]
            op_info["state"] = "translated"
            op_info["pass_flags"]["add_check"] = "success"

        # ========== DELEGATE 算子跳过 ==========
        elif op["op_name"] == "DELEGATE":
            continue

        # ========== 未知算子 ==========
        else:
            fatal_error(f"不支持的算子类型: {op['op_name']}",
                        f"当前支持的算子: UNIDIRECTIONAL_SEQUENCE_LSTM, FULLY_CONNECTED, SOFTMAX, RESHAPE, ADD")

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

        # 添加索引信息
        op_info["input_indices"] = [idx for idx in op["inputs"] if idx != -1]
        op_info["output_indices"] = [idx for idx in op["outputs"] if idx != -1]

        ops.append(op_info)

    return {
        "input": input_details,
        "output": output_details,
        "ops": ops,
        "tensors": tensor_map,
    }
