#!/usr/bin/env python3
"""测试 generate_lut 模块"""

import sys
from pathlib import Path

import pytest
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinymlc.generate_lut import generate_sigmoid_lut, generate_tanh_lut


def test_sigmoid_lut_length():
    """测试 sigmoid LUT 长度"""
    lut = generate_sigmoid_lut()
    assert len(lut) == 256
    assert isinstance(lut, np.ndarray)


def test_sigmoid_lut_range():
    """测试 sigmoid LUT 值范围"""
    lut = generate_sigmoid_lut()
    # Sigmoid 输出范围 [0, 32767]
    assert lut.min() >= 0
    assert lut.max() <= 32767
    # 应该是递增的
    assert np.all(np.diff(lut) >= 0)


def test_tanh_lut_length():
    """测试 tanh LUT 长度"""
    lut = generate_tanh_lut()
    assert len(lut) == 256
    assert isinstance(lut, np.ndarray)


def test_tanh_lut_range():
    """测试 tanh LUT 值范围"""
    lut = generate_tanh_lut()
    # Tanh 输出范围 [-32768, 32767]
    assert lut.min() >= -32768
    assert lut.max() <= 32767
    # 应该是递增的
    assert np.all(np.diff(lut) >= 0)


def test_sigmoid_lut_endpoints():
    """测试 sigmoid LUT 端点值（允许误差）"""
    lut = generate_sigmoid_lut()
    # 输入 -8 附近 → 接近 0（允许 0-20 范围）
    assert lut[0] < 20, f"端点值 {lut[0]} 太大"
    # 输入 8 附近 → 接近 32767（允许 32747-32767 范围）
    assert lut[-1] > 32700, f"端点值 {lut[-1]} 太小"


def test_tanh_lut_endpoints():
    """测试 tanh LUT 端点值（允许误差）"""
    lut = generate_tanh_lut()
    # 输入 -8 附近 → 接近 -32768（允许 -32768 到 -32700）
    assert lut[0] < -32700, f"端点值 {lut[0]} 太大"
    # 输入 8 附近 → 接近 32767（允许 32700-32767 范围）
    assert lut[-1] > 32700, f"端点值 {lut[-1]} 太小"
