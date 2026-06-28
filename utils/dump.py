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

import sys
from termcolor import colored


IS_WINDOWS = sys.platform.startswith('win')
if IS_WINDOWS:
    try:
        from colorama import init
        init(autoreset=True)
    except ImportError:
        print("[WARNING]: colorama is required under Windows,"
              "please run 'pip install colorama' to install it.")


def info(msg):
    """Print info message"""
    print(colored(f"[INFO] {msg}", "cyan"))


def warning(msg, suggestion=None):
    """Print warning message (does not exit)"""
    print(colored(f"[WARNING] {msg}", "yellow"))
    if suggestion:
        print(colored(f"SUGGESTION: {suggestion}", "yellow"))


def fatal_error(msg, suggestion=None):
    """Print fatal error message and exit"""
    print(colored("\n" + "=" * 60, "red"))
    print(colored(f"[ERROR] {msg}", "red", attrs=["bold"]))
    if suggestion:
        print(colored(f"SUGGESTION: {suggestion}", "red"))
    print(colored("\n" + "=" * 60, "red"))
    sys.exit(1)


def dump_model_info(model_info):
    """Dump model info"""
    info("\n=== Model Info ===")
    info(f"Input tensors: {len(model_info['input'])}")
    for inp in model_info["input"]:
        info(f"  - {inp.get('name', 'unnamed')}: "
             f"shape={inp['shape']}, dtype={inp['dtype']}")
    info(f"Output tensors: {len(model_info['output'])}")
    for out in model_info["output"]:
        info(f"  - {out.get('name', 'unnamed')}: "
             f"shape={out['shape']}, dtype={out['dtype']}")
    info(f"\nOperator count: {len(model_info['ops'])}")
    for op in model_info["ops"]:
        info(f"\n  [{op['index']}] {op['op_name']}")
        info(f"      Inputs:")
        for inp in op.get("input_details", []):
            info(f"        - [{inp.get('index', '?')}] "
                 f"{inp.get('name', 'unknown')}: "
                 f"shape={inp.get('shape', [])}, "
                 f"size={inp.get('size', 0)}")
        info(f"      Outputs:")
        for out in op.get("output_details", []):
            info(f"        - [{out.get('index', '?')}] "
                 f"{out.get('name', 'unknown')}: "
                 f"shape={out.get('shape', [])}, "
                 f"size={out.get('size', 0)}")
