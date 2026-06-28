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
"""Test translator module"""

import subprocess
import sys
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "test_models"
OUTPUT_DIR = PROJECT_ROOT / "test_output"


@pytest.fixture
def lstm_model():
    """LSTM test model fixture"""
    model_path = FIXTURES_DIR / "trained_lstm_int8.tflite"
    if not model_path.exists():
        pytest.skip(f"Test model not found: {model_path}")
    return model_path


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory fixture"""
    return tmp_path / "tinymlc_generated"


def test_translator_runs(lstm_model, output_dir):
    """Test translator can run normally (generate test entry)"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "TinyMLC.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "--with-test-main",  # Add this parameter
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Translator failed: {result.stderr}"
    assert (output_dir / "model.c").exists(), "model.c not generated"
    assert (output_dir / "model.h").exists(), "model.h not generated"
    assert (output_dir / "main_test.c").exists(), "main_test.c not generated"


def test_generated_files_content(lstm_model, output_dir):
    """Test generated files are non-empty"""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "TinyMLC.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "--with-test-main",  # Add this parameter
        ],
        capture_output=True,
        check=True,
    )

    # Check generated files are non-empty
    for filename in ["model.c", "model.h", "main_test.c"]:
        file_path = output_dir / filename
        assert file_path.exists(), f"{filename} not found"
        assert file_path.stat().st_size > 0, f"{filename} is empty"


def test_translator_with_verbose(lstm_model, output_dir):
    """Test verbose mode"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "TinyMLC.translator",
            str(lstm_model),
            "--output-dir",
            str(output_dir),
            "-v",
            "--with-test-main",  # Optional
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # In verbose mode, stdout should contain model info
    assert "Model Info" in result.stdout or "Operator count" in result.stdout
