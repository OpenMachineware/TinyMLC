#!/usr/bin/env python3
"""Test extract_litert_weights module"""

from pathlib import Path

import pytest
from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

from tinymlc.model_converter.parser_litert import (
    parse_model_tflite,
    extract_fc_weights,
    extract_lstm_weights,
)


@pytest.fixture
def interpreter():
    """Create interpreter fixture"""
    model_path = Path(__file__).parent.parent / "test_models" / \
        "trained_lstm_int8.tflite"
    if not model_path.exists():
        pytest.skip(f"Test model not found: {model_path}")
    interpreter = LiteRTInterpreter(model_path=str(model_path))
    return interpreter


@pytest.fixture
def model_info(interpreter):
    """Parse model info"""
    model_path = Path(__file__).parent.parent / "test_models" / \
        "trained_lstm_int8.tflite"
    return parse_model_tflite(str(model_path))


@pytest.fixture
def fc_op_info(model_info):
    """Get FC operator info"""
    for op in model_info["ops"]:
        if op["op_name"] == "FULLY_CONNECTED":
            return op
    pytest.fail("FC operator not found")


@pytest.fixture
def lstm_op_info(model_info):
    """Get LSTM operator info"""
    for op in model_info["ops"]:
        if op["op_name"] == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            return op
    pytest.fail("LSTM operator not found")


def test_extract_fc_weights_returns_tuple(interpreter, fc_op_info):
    """Test extract_fc_weights returns tuple"""
    result = extract_fc_weights(interpreter, fc_op_info)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_fc_weights_shape(interpreter, fc_op_info):
    """Test FC weights shape is correct"""
    weights, bias = extract_fc_weights(interpreter, fc_op_info)
    # Based on model: weights [10, 560], bias [10]
    assert weights.shape == (10, 560)
    assert bias.shape == (10,)


def test_extract_lstm_weights_returns_dict(interpreter, lstm_op_info):
    """Test extract_lstm_weights returns dict"""
    result = extract_lstm_weights(interpreter, lstm_op_info)
    assert isinstance(result, dict)
    assert "input" in result
    assert "recurrent" in result
    assert "bias" in result


def test_extract_lstm_weights_gates(interpreter, lstm_op_info):
    """Test LSTM four gate weights all exist"""
    result = extract_lstm_weights(interpreter, lstm_op_info)
    gates = ['i', 'f', 'g', 'o']
    for gate in gates:
        assert gate in result["input"], f"input_{gate} missing"
        assert gate in result["recurrent"], f"recurrent_{gate} missing"
