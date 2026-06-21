#!/usr/bin/env python3
"""Test parser module"""

import sys
from pathlib import Path

import pytest
import tensorflow as tf

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.parser import parse_model


@pytest.fixture
def lstm_model_path():
    """LSTM test model path"""
    path = Path(__file__).parent.parent / "test_models" / "trained_lstm_int8.tflite"
    if not path.exists():
        pytest.skip(f"Test model not found: {path}")
    return path


def test_parse_model_returns_dict(lstm_model_path):
    """Test parse_model returns dict"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    assert isinstance(result, dict)
    assert "input" in result
    assert "output" in result
    assert "ops" in result
    assert "tensors" in result


def test_parse_model_ops_count(lstm_model_path):
    """Test parsed operator count is correct"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    # LSTM model should have LSTM, RESHAPE, FC, SOFTMAX
    ops = result["ops"]
    assert len(ops) == 4


def test_parse_model_has_lstm(lstm_model_path):
    """Test can identify LSTM operator"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in op_names


def test_parse_model_has_fc(lstm_model_path):
    """Test can identify FC operator"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "FULLY_CONNECTED" in op_names


def test_parse_model_has_softmax(lstm_model_path):
    """Test can identify Softmax operator"""
    interpreter = tf.lite.Interpreter(model_path=str(lstm_model_path))
    interpreter.allocate_tensors()

    result = parse_model(interpreter)

    op_names = [op["op_name"] for op in result["ops"]]
    assert "SOFTMAX" in op_names


def test_parse_model_lstm_params(lstm_model_path):
    """Test LSTM params are correctly extracted"""
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
        pytest.fail("LSTM operator not found")
