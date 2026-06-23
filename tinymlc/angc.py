#!/usr/bin/env python3
# angc.py - ANG CLI entry point
# ANG (Automatic Network Generator) - Main entry point.
#
# ANG is a frontend for TinyMLC that automatically generates
# network structures optimized for given hardware constraints.

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from tinymlc.ang.model_info import (ModelInfo, TensorSpec, Op,
                                    create_default_tensor_spec)
from tinymlc.ang.model_builder import ModelBuilder
from tinymlc.ang.estimator import Estimator
from tinymlc.ang.estimator_software import SoftwareEstimator
from tinymlc.ang.estimator_qemu import QemuEstimator
from tinymlc.ang.estimator_hal import HardwareHALEstimator
from tinymlc.ang.model_generator import ModelGenerator
from tinymlc.ang.table import TableManager
from tinymlc.ang.utils import (calculate_macs, calculate_params,
                               calculate_peak_ram)
from utils.dump import fatal_error, warning, info


def create_estimator(args: argparse.Namespace) -> Estimator:
    """
    Create an estimator based on command-line arguments.
    """
    estimator_type = getattr(args, "estimator", "software")

    # 直接读独立参数
    max_macs = getattr(args, "max_macs", 100000)
    max_ram_kb = getattr(args, "max_ram", 30)
    max_flash_kb = getattr(args, "max_flash", 64)

    if estimator_type == "software":
        return SoftwareEstimator({
            "max_macs": max_macs,
            "max_params": 50000,  # TODO: Independently configurable.
            "max_ram": max_ram_kb * 1024,
            "clock_speed": getattr(args, "clock_speed", 100000000),
        })

    elif estimator_type == "qemu":
        return QemuEstimator({
            "max_macs": max_macs,
            "max_params": 50000,
            "max_ram": max_ram_kb * 1024,
            "qemu_binary": getattr(args, "qemu_binary", "qemu-system-arm"),
            "cpu": getattr(args, "qemu_cpu", "cortex-m4"),
            "icount_shift": getattr(args, "icount_shift", 0),
            "clock_speed": getattr(args, "clock_speed", 100000000),
        })

    elif estimator_type == "hardware":
        return HardwareHALEstimator({
            "max_macs": max_macs,
            "max_params": 50000,
            "max_ram": max_ram_kb * 1024,
            "script_path": getattr(args, "estimator_script", None),
            "function_name": getattr(args, "estimator_function", "estimate"),
            "clock_speed": getattr(args, "clock_speed", 100000000),
        })

    else:
        fatal_error(f"Unknown estimator type: {estimator_type}")


def do_generate(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute network generation (generate + build).
    """
    estimator = create_estimator(args)

    generator = ModelGenerator(
        estimator=estimator,
        config={
            "task_type": getattr(args, "task_type", "classification"),
            "population_size": getattr(args, "population", 50),
            "generations": getattr(args, "generations", 50),
            "mutation_rate": getattr(args, "mutation_rate", 0.1),
            "crossover_rate": getattr(args, "crossover_rate", 0.8),
            "early_stop": getattr(args, "early_stop", 10),
            "input_shape": getattr(args, "input_shape", [1, 28, 28, 1]),
            "output_shape": getattr(args, "output_shape", [1, 10]),
            "max_macs": getattr(args, "max_macs", 100000),
            "max_ram": getattr(args, "max_ram", 30) * 1024,
            "max_flash": getattr(args, "max_flash", 64) * 1024,
            "num_samples": getattr(args, "num_samples", 100),
        }
    )

    mode = getattr(args, "mode", "genetic")
    model_info = generator.generate(mode=mode)

    return model_info


def do_table_lookup(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Look up a network from a pre-built table.
    """
    table_file = getattr(args, "table_file", None)
    if not table_file:
        fatal_error("--table-file is required for table mode")

    table_mgr = TableManager(table_file)

    lookup_key = {
        "max_macs": getattr(args, "max_macs", 100000),
        "max_ram": getattr(args, "max_ram", 30) * 1024,
        "max_flash": getattr(args, "max_flash", 64) * 1024,
        "input_shape": getattr(args, "input_shape", [1, 28, 28, 1]),
        "output_shape": getattr(args, "output_shape", [1, 10]),
    }

    return table_mgr.lookup(lookup_key)


def do_build_table(args: argparse.Namespace) -> None:
    """
    Build a hardware profile table.
    """
    board = getattr(args, "board", "unknown")
    output = getattr(args, "output", "table.json")
    estimator = create_estimator(args)

    input_shape = getattr(args, "input_shape", [1, 28, 28, 1])
    output_shape = getattr(args, "output_shape", [1, 10])

    table_mgr = TableManager(output)
    table_mgr.build_table(
        board=board,
        estimator=estimator,
        num_entries=100,
        max_layers=10,
        input_shape=input_shape,
        output_shape=output_shape,
    )
    table_mgr.save(output)
    info(f"Table saved to: {output}")


def do_update_table(args: argparse.Namespace) -> None:
    """
    Update an existing table.
    """
    table_file = getattr(args, "table_file", None)
    if not table_file:
        fatal_error("--table-file is required for update")

    output = getattr(args, "output", table_file)
    add_ops = getattr(args, "add_ops", "").split(",") if args.add_ops else []
    recalibrate = getattr(args, "recalibrate", False)

    table_mgr = TableManager(table_file)
    table_mgr.update_table(
        add_ops=add_ops,
        recalibrate=recalibrate,
        entries=recalibrate,
    )
    table_mgr.save(output)
    info(f"Table updated and saved to: {output}")


def main() -> int:
    parent_parser = argparse.ArgumentParser(add_help=False)
    # ========== Global arguments ==========
    parent_parser.add_argument(
        "--input-shape",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 28, 28, 1],
        help='Input shape: "1,28,28,1"',
    )
    parent_parser.add_argument(
        "--output-shape",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 10],
        help='Output shape: "1,10"',
    )
    parent_parser.add_argument(
        "--max-macs",
        type=int,
        default=100000,
        help="Maximum MACs",
    )
    parent_parser.add_argument(
        "--max-ram",
        type=int,
        default=30,
        help="Maximum RAM in KB",
    )
    parent_parser.add_argument(
        "--max-flash",
        type=int,
        default=64,
        help="Maximum Flash in KB",
    )
    parent_parser.add_argument(
        "--clock-speed",
        type=int,
        default=100000000,
        help="Clock speed in Hz",
    )
    parent_parser.add_argument(
        "--dump-model",
        type=str,
        default="model_info.json",
        help="Output file",
    )
    parent_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parent_parser.add_argument(
        "--qemu-cpu",
        type=str,
        default="cortex-m4",
        help="QEMU CPU model",
    )
    parent_parser.add_argument(
        "--icount-shift",
        type=int,
        default=0,
        help="QEMU icount shift",
    )
    parent_parser.add_argument(
        "--qemu-binary",
        type=str,
        default="qemu-system-arm",
        help="QEMU binary path",
    )
    parent_parser.add_argument(
        "--estimator",
        choices=["software", "qemu", "hardware"],
        default="software",
        help="Estimator type",
    )
    parent_parser.add_argument(
        "--estimator-script",
        type=str,
        help="Path to hardware estimator script",
    )
    parent_parser.add_argument(
        "--estimator-function",
        type=str,
        default="estimate",
        help="Function name in estimator script",
    )

    parser = argparse.ArgumentParser(
        description="ANG - Automatic Network Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent_parser],
    )

    # ========== Subcommands ==========
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ---- generate command ----
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a network optimized for the given constraints",
        parents=[parent_parser],
    )
    gen_parser.add_argument(
        "--mode",
        choices=["random", "genetic"],
        default="genetic",
        help="Generation mode",
    )
    gen_parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size",
    )
    gen_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations",
    )
    gen_parser.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="Stop if no improvement for N generations",
    )
    gen_parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples for random mode",
    )
    # Hardware HAL
    gen_parser.add_argument(
        "--task-type",
        choices=["classification", "detection", "segmentation"],
        default="classification")

    # ---- table command ----
    table_parser = subparsers.add_parser(
        "table",
        help="Look up a network from a pre-built table",
        parents=[parent_parser],
    )
    table_parser.add_argument(
        "--table-file",
        type=str,
        required=True,
        help="Path to the table file",
    )

    # ---- build-table command ----
    build_parser = subparsers.add_parser(
        "build-table",
        help="Build a hardware profile table",
        parents=[parent_parser],
    )
    build_parser.add_argument(
        "--board",
        type=str,
        required=True,
        help="Board name (e.g., stm32f4)",
    )
    build_parser.add_argument(
        "--dump-table",
        type=str,
        default="table.json",
        help="Output file for the table",
    )

    # ---- update-table command ----
    update_parser = subparsers.add_parser(
        "update-table",
        help="Update an existing table",
        parents=[parent_parser],
    )
    update_parser.add_argument(
        "--table-file",
        type=str,
        required=True,
        help="Path to the table file to update",
    )
    update_parser.add_argument(
        "--update-table",
        type=str,
        help="Output file (defaults to --table-file)",
    )
    update_parser.add_argument(
        "--add-ops",
        type=str,
        help="Comma-separated list of ops to add: dwconv,group_conv",
    )
    update_parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recalibrate existing entries",
    )
    update_parser.add_argument(
        "--entries",
        type=str,
        help="Comma-separated list of entries to recalibrate",
    )

    # ---- show-table command ----
    show_parser = subparsers.add_parser(
        "show-table",
        help="Show table information",
        parents = [parent_parser],
    )
    show_parser.add_argument(
        "--table-file",
        type=str,
        required=True,
        help="Path to the table file",
    )
    show_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics",
    )

    args = parser.parse_args()

    try:
        if args.command == "generate":
            model_info = do_generate(args)
            with open(args.dump_model, "w") as f:
                json.dump(model_info, f, indent=2, default=str)
            info(f"Model info saved to: {args.dump_model}")

        elif args.command == "table":
            model_info = do_table_lookup(args)
            with open(args.dump_table, "w") as f:
                json.dump(model_info, f, indent=2, default=str)
            info(f"Model info saved to: {args.dump_table}")

        elif args.command == "build-table":
            do_build_table(args)

        elif args.command == "update-table":
            do_update_table(args)

        elif args.command == "show-table":
            table_mgr = TableManager(args.table_file)
            if args.stats:
                stats = table_mgr.get_stats()
                print(json.dumps(stats, indent=2))
            else:
                info(f"Table: {args.table_file}")
                info(f"  Entries: {len(table_mgr.entries)}")

        else:
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        fatal_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
