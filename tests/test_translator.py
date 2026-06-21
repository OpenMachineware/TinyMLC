#!/usr/bin/env python3
"""测试 translator 模块"""

import subprocess
import sys
from pathlib import Path

import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "test_models"
OUTPUT_DIR = PROJECT_ROOT / "test_output"


@pytest.fixture
def lstm_model():
    """LSTM 测试模型 fixture"""
    model_path = FIXTURES_DIR / "trained_lstm_int8.tflite"
    if not model_path.exists():
        pytest.skip(f"测试模型不存在: {model_path}")
    return model_path


@pytest.fixture
def output_dir(tmp_path):
    """临时输出目录 fixture"""
    return tmp_path / "tinymlc_generated"


def test_translator_runs(lstm_model, output_dir):
    """测试 translator 能正常执行（生成测试入口）"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tinymlc.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "--with-test-main",  # 添加这个参数
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Translator 失败: {result.stderr}"
    assert (output_dir / "model.c").exists(), "model.c 未生成"
    assert (output_dir / "model.h").exists(), "model.h 未生成"
    assert (output_dir / "main_test.c").exists(), "main_test.c 未生成"


def test_generated_files_content(lstm_model, output_dir):
    """测试生成的文件非空"""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tinymlc.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "--with-test-main",  # 添加这个参数
        ],
        capture_output=True,
        check=True,
    )

    # 检查生成的文件非空
    for filename in ["model.c", "model.h", "main_test.c"]:
        file_path = output_dir / filename
        assert file_path.exists(), f"{filename} 不存在"
        assert file_path.stat().st_size > 0, f"{filename} 为空"


def test_translator_with_verbose(lstm_model, output_dir):
    """测试 verbose 模式"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tinymlc.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "-v",
            "--with-test-main",  # 可选，加不加都可以
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # verbose 模式下 stdout 应该包含模型信息
    assert "模型信息" in result.stdout or "算子数量" in result.stdout
