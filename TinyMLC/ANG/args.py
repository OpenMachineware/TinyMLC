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
