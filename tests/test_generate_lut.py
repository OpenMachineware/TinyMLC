#!/usr/bin/env python3
"""Test generate_lut module"""

from pathlib import Path

import pytest
import numpy as np

from TinyMLC.generate_lut import generate_sigmoid_lut, generate_tanh_lut


def test_sigmoid_lut_length():
    """Test sigmoid LUT length"""
    lut = generate_sigmoid_lut()
    assert len(lut) == 256
    assert isinstance(lut, np.ndarray)


def test_sigmoid_lut_range():
    """Test sigmoid LUT value range"""
    lut = generate_sigmoid_lut()
    # Sigmoid output range [0, 32767]
    assert lut.min() >= 0
    assert lut.max() <= 32767
    # Should be monotonically increasing
    assert np.all(np.diff(lut) >= 0)


def test_tanh_lut_length():
    """Test tanh LUT length"""
    lut = generate_tanh_lut()
    assert len(lut) == 256
    assert isinstance(lut, np.ndarray)


def test_tanh_lut_range():
    """Test tanh LUT value range"""
    lut = generate_tanh_lut()
    # Tanh output range [-32768, 32767]
    assert lut.min() >= -32768
    assert lut.max() <= 32767
    # Should be monotonically increasing
    assert np.all(np.diff(lut) >= 0)


def test_sigmoid_lut_endpoints():
    """Test sigmoid LUT endpoint values (with tolerance)"""
    lut = generate_sigmoid_lut()
    # Input near -8 -> close to 0 (allow 0-20 range)
    assert lut[0] < 20, f"Endpoint value {lut[0]} too large"
    # Input near 8 -> close to 32767 (allow 32747-32767 range)
    assert lut[-1] > 32700, f"Endpoint value {lut[-1]} too small"


def test_tanh_lut_endpoints():
    """Test tanh LUT endpoint values (with tolerance)"""
    lut = generate_tanh_lut()
    # Input near -8 -> close to -32768 (allow -32768 to -32700)
    assert lut[0] < -32700, f"Endpoint value {lut[0]} too large"
    # Input near 8 -> close to 32767 (allow 32700-32767 range)
    assert lut[-1] > 32700, f"Endpoint value {lut[-1]} too small"
