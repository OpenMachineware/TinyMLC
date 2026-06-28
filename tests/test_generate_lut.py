# -*- coding: utf-8 -*-
# TinyMLC - Tiny Machine Learning Compiler
#
# Copyright (c) 2026 Jia Liu & TinyMLC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of TinyMLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
