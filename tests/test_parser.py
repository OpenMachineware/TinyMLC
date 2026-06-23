#!/usr/bin/env python3
"""Test parser module"""

from pathlib import Path

import pytest

from tinymlc.converter.parser_litert import parse_model_tflite


@pytest.fixture
def lstm_model_path():
    """LSTM test model path"""
    path = Path(__file__).parent.parent / "test_models" / "trained_lstm_int8.tflite"
    if not path.exists():
        pytest.skip(f"Test model not found: {path}")
    return path


def test_parse_model_returns_dict(lstm_model_path):
    """Test parse_model returns dict"""
    result = parse_model_tflite(str(lstm_model_path))

    assert isinstance(result, dict)
    assert "input" in result
    assert "output" in result
    assert "ops" in result
    assert "tensors" in result


def test_parse_model_ops_count(lstm_model_path):
    """Test parsed operator count is correct"""
    result = parse_model_tflite(str(lstm_model_path))

    # LSTM model should have LSTM, RESHAPE, FC, SOFTMAX
    ops = result["ops"]
    assert len(ops) == 4


def test_parse_model_has_lstm(lstm_model_path):
    """Test can identify LSTM operator"""
    result = parse_model_tflite(str(lstm_model_path))

    op_names = [op["op_name"] for op in result["ops"]]
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in op_names


def test_parse_model_has_fc(lstm_model_path):
    """Test can identify FC operator"""
    result = parse_model_tflite(str(lstm_model_path))

    op_names = [op["op_name"] for op in result["ops"]]
    assert "FULLY_CONNECTED" in op_names


def test_parse_model_has_softmax(lstm_model_path):
    """Test can identify Softmax operator"""
    result = parse_model_tflite(str(lstm_model_path))

    op_names = [op["op_name"] for op in result["ops"]]
    assert "SOFTMAX" in op_names


def test_parse_model_lstm_params(lstm_model_path):
    """Test LSTM params are correctly extracted"""
    result = parse_model_tflite(str(lstm_model_path))

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
