#!/usr/bin/env python3
"""测试 parser 模块"""

import sys
from pathlib import Path

import pytest
import tensorflow as tf

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.parser import parse_model


@pytest.fixture
def lstm_model_path():
    """LSTM 测试模型路径"""
    path = Path(__file__).parent.parent / "test_models" / "trained_lstm_int8.tflite"
    if not path.exists():
        pytest.skip(f"测试模型不存在: {path}")
    return path


def test_parse_model_returns_dict(lstm_model_path):
    """测试 parse_model 返回字典"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    assert isinstance(result, dict)
    assert "input" in result
    assert "output" in result
    assert "ops" in result
    assert "tensors" in result


def test_parse_model_ops_count(lstm_model_path):
    """测试解析出的算子数量正确"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    # LSTM 模型应该有 LSTM, RESHAPE, FC, SOFTMAX
    ops = result["ops"]
    assert len(ops) == 4


def test_parse_model_has_lstm(lstm_model_path):
    """测试能识别 LSTM 算子"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in op_names


def test_parse_model_has_fc(lstm_model_path):
    """测试能识别 FC 算子"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "FULLY_CONNECTED" in op_names


def test_parse_model_has_softmax(lstm_model_path):
    """测试能识别 Softmax 算子"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "SOFTMAX" in op_names


def test_parse_model_lstm_params(lstm_model_path):
    """测试 LSTM 参数被正确提取"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    for op in result["ops"]:
        if op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            assert "lstm_params" in op
            params = op["lstm_params"]
            assert params["batch_size"] == 1
            assert params["time_steps"] == 28
            assert params["input_size"] == 28
            assert params["hidden_size"] == 20
            break
    else:
        pytest.fail("未找到 LSTM 算子")
