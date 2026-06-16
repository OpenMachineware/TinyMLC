#!/usr/bin/env python3
"""测试 extract_weights 模块"""

import sys
from pathlib import Path

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.extract_weights import extract_fc_weights, extract_lstm_weights
from tinymlc.parser import parse_model


@pytest.fixture
def interpreter():
    """创建 interpreter fixture"""
    model_path = Path(__file__).parent / "fixtures" / "trained_lstm_int8.tflite"
    if not model_path.exists():
        pytest.skip(f"测试模型不存在: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


@pytest.fixture
def model_info(interpreter):
    """解析模型信息"""
    return parse_model(interpreter)


@pytest.fixture
def fc_op_info(model_info):
    """获取 FC 算子信息"""
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            return op
    pytest.fail("未找到 FC 算子")


@pytest.fixture
def lstm_op_info(model_info):
    """获取 LSTM 算子信息"""
    for op in model_info["ops"]:
        if op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            return op
    pytest.fail("未找到 LSTM 算子")


def test_extract_fc_weights_returns_tuple(interpreter, fc_op_info):
    """测试 extract_fc_weights 返回元组"""
    result = extract_fc_weights(interpreter, fc_op_info)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_fc_weights_shape(interpreter, fc_op_info):
    """测试 FC 权重形状正确"""
    weights, bias = extract_fc_weights(interpreter, fc_op_info)
    # 根据你的模型：权重 [10, 560], bias [10]
    assert weights.shape == (10, 560)
    assert bias.shape == (10,)


def test_extract_lstm_weights_returns_dict(interpreter, lstm_op_info):
    """测试 extract_lstm_weights 返回字典"""
    result = extract_lstm_weights(interpreter, lstm_op_info)
    assert isinstance(result, dict)
    assert "input" in result
    assert "recurrent" in result
    assert "bias" in result


def test_extract_lstm_weights_gates(interpreter, lstm_op_info):
    """测试 LSTM 四个门的权重都存在"""
    result = extract_lstm_weights(interpreter, lstm_op_info)
    gates = ['i', 'f', 'g', 'o']
    for gate in gates:
        assert gate in result["input"], f"input_{gate} 缺失"
        assert gate in result["recurrent"], f"recurrent_{gate} 缺失"
