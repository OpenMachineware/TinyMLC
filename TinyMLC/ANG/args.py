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
"""Command-line argument utilities for ANG and TinyMLC CLI"""

import argparse
from typing import List

from TinyMLC.ANG.estimator import Estimator
from TinyMLC.ANG.estimator_software import SoftwareEstimator
from TinyMLC.ANG.estimator_qemu import QemuEstimator
from TinyMLC.ANG.estimator_hal import HardwareHALEstimator
from utils.dump import fatal_error

def parse_shape(shape_str: str) -> List[int]:
    """Parse shape string like '1,28,28,1' to list of integers"""
    return [int(x.strip()) for x in shape_str.split(",")]


def create_estimator(args: argparse.Namespace) -> Estimator:
    """
    Create an estimator based on command-line arguments.
    """
    estimator_type = getattr(args, "estimator", "software")

    # Read from args
    max_macs = getattr(args, "max_macs", 100000)
    max_ram_kb = getattr(args, "max_ram", 30)
    max_flash_kb = getattr(args, "max_flash", 64)

    if estimator_type == "software":
        return SoftwareEstimator(
            {
                "max_macs": max_macs,
                "max_params": 50000,  # TODO: Independently configurable.
                "max_ram": max_ram_kb * 1024,
                "clock_speed": getattr(args, "clock_speed", 100000000),
            }
        )

    elif estimator_type == "qemu":
        return QemuEstimator(
            {
                "max_macs": max_macs,
                "max_params": 50000,
                "max_ram": max_ram_kb * 1024,
                "qemu_binary": "qemu-system-" + getattr(args, "target", "arm"),
                "cpu": getattr(args, "qemu_cpu", "cortex-m4"),
                "icount_shift": getattr(args, "icount_shift", 0),
                "clock_speed": getattr(args, "clock_speed", 100000000),
            }
        )

    elif estimator_type == "hardware":
        return HardwareHALEstimator(
            {
                "max_macs": max_macs,
                "max_params": 50000,
                "max_ram": max_ram_kb * 1024,
                "script_path": getattr(args, "estimator_script", None),
                "function_name": getattr(
                    args, "estimator_function", "estimate"
                ),
                "clock_speed": getattr(args, "clock_speed", 100000000),
            }
        )

    else:
        fatal_error(f"Unknown estimator type: {estimator_type}")
