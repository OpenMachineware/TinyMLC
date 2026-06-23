#!/usr/bin/env python3
# cli.py - CLI TinyMLC

import argparse
import json
import sys
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from tinymlc.ang.model_info import (
    ModelInfo, TensorSpec, Op, create_default_tensor_spec)
from tinymlc.ang.model_generator import ModelGenerator
from tinymlc.ang.model_builder import ModelBuilder
from tinymlc.ang.estimator import Estimator
from tinymlc.ang.estimator_software import SoftwareEstimator
from tinymlc.ang.estimator_qemu import QemuEstimator
from tinymlc.ang.estimator_hal import HardwareHALEstimator
from tinymlc.ang.table import TableManager
from tinymlc.codegen import generate_c_code, copy_files_to_build
from tinymlc.generate_lut import generate_lut
from tinymlc.ang.utils import (
    calculate_macs, calculate_params, calculate_peak_ram)
from tinymlc.converter.parser_litert import (
    parse_model_tflite, extract_all_weights_litert)
from tinymlc.converter.parser_onnx import (
    parse_model_onnx, extract_all_weights_onnx)
from tinymlc.converter.export_weights import export_model_weights
from utils.dump import fatal_error, warning, info, dump_model_info


def parse_shape(shape_str: str) -> List[int]:
    return [int(x.strip()) for x in shape_str.split(",")]


def create_estimator(args: argparse.Namespace) -> Estimator:
    """
    Create an estimator based on command-line arguments.
    """
    estimator_type = getattr(args, "estimator", "software")

    # Read form args
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
                "function_name": getattr(args, "estimator_function", "estimate"),
                "clock_speed": getattr(args, "clock_speed", 100000000),
            }
        )

    else:
        fatal_error(f"Unknown estimator type: {estimator_type}")


def get_output_dir(args: argparse.Namespace) -> Path:
    out_dir = Path(getattr(args, "output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_table_name(args: argparse.Namespace) -> Optional[str]:
    return getattr(args, "table_name", None)


def handle_generate(args: argparse.Namespace) -> int:
    table_name = get_table_name(args)

    if table_name:
        info(f"Looking up network from table: {table_name}")
        try:
            table_mgr = TableManager(table_name)
            constraints = {
                "max_macs": getattr(args, "max_macs", 100000),
                "max_ram": getattr(args, "max_ram", 30) * 1024,
                "max_flash": getattr(args, "max_flash", 64) * 1024,
                "input_shape": getattr(args, "input_shape", [1, 28, 28, 1]),
                "output_shape": getattr(args, "output_shape", [1, 10]),
            }
            model_info = table_mgr.lookup(constraints)
            info("Table lookup successful.")
        except Exception as e:
            warning(f"Table lookup failed: {e}")
            info("Falling back to network generation...")
            model_info = None
    else:
        model_info = None

    if model_info is None:
        info("Running network generation...")
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
            },
        )

        mode = getattr(args, "generate_mode", "genetic")
        model_info = generator.generate(mode=mode)

    dump_model = getattr(args, "dump_model", None)
    if dump_model:
        with open(dump_model, "w") as f:
            json.dump(model_info, f, indent=2, default=str)
        info(f"Model info saved to: {dump_model}")

    info("Generating C code...")
    out_dir = get_output_dir(args)
    result = generate_c_code(
        model_info=model_info,
        output_dir=str(out_dir),
        target=getattr(args, "target", "riscv"),
        inference_func=getattr(args, "inference_function_name", "tinymlc_inference"),
        with_test_main=getattr(args, "with_test_main", False),
    )

    for filename, content in result.items():
        filepath = out_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        info(f"Generated: {filepath}")

    if getattr(args, "run", False):
        info("Running build...")

    return 0


def handle_table(args: argparse.Namespace) -> int:
    table_name = getattr(args, "table_name", None)
    if not table_name:
        fatal_error("--table-name is required for table operations")

    table_mgr = TableManager(table_name)

    if getattr(args, "build", False):
        info(f"Building table: {table_name}")
        board = getattr(args, "board", "unknown")
        estimator = create_estimator(args)
        input_shape = getattr(args, "input_shape", [1, 28, 28, 1])
        output_shape = getattr(args, "output_shape", [1, 10])

        table_mgr.build_table(
            board=board,
            estimator=estimator,
            num_entries=100,
            max_layers=10,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        table_mgr.save(table_name)
        info(f"Table saved to: {table_name}")

    elif getattr(args, "update", False):
        info(f"Updating table: {table_name}")
        add_ops = getattr(args, "add_ops", "").split(",") if args.add_ops else []
        recalibrate = getattr(args, "recalibrate", False)

        table_mgr.update_table(
            add_ops=add_ops,
            recalibrate=recalibrate,
            entries=recalibrate,
        )
        table_mgr.save(table_name)
        info(f"Table updated and saved to: {table_name}")

    elif getattr(args, "show", False):
        info(f"Table: {table_name}")
        info(f"  Entries: {len(table_mgr.entries)}")
        if getattr(args, "stats", False):
            stats = table_mgr.get_stats()
            print(json.dumps(stats, indent=2))

    else:
        fatal_error("Table operation not specified. Use --build, --update, or --show")

    return 0


def handle_convert(args: argparse.Namespace) -> int:
    target = getattr(args, "target", None)
    mode = getattr(args, "mode", None)
    accel = getattr(args, "accel", None)
    model_path = getattr(args, "model", None)
    output_dir = getattr(args, "output_dir", None)
    output_dir = Path(output_dir)
    with_test_main = getattr(args, "with_test_main", None)
    inference_function_name = getattr(args, "inference_function_name", None)
    verbose = getattr(args, "verbose", None)
    run = getattr(args, "run", None)

    if not model_path:
        fatal_error("--model is required for convert")
    if not Path(model_path).exists():
        fatal_error(f"Model file not found: {model_path}", "Please check file path")
    if model_path.endswith(".tflite"):
        model_info = parse_model_tflite(model_path)
        # Extract weights (interpreter created internally)
        extract_all_weights_litert(model_path, model_info)
        # Export weights using unified function
        quant_scales = export_model_weights(output_dir, model_info)
        model_info["quant_scales"] = quant_scales
    elif model_path.endswith(".onnx"):
        model_info = parse_model_onnx(model_path)
        # Extract weights (model_path for consistency)
        extract_all_weights_onnx(model_path, model_info)
        # Export weights using unified function
        quant_scales = export_model_weights(output_dir, model_info)
        model_info["quant_scales"] = quant_scales
    else:
        fatal_error("Unsupported model format", "Supported: .tflite and .onnx")
    info(f"Converting model: {model_path}")

    if verbose:
        dump_model_info(model_info)

    if target == "arm":
        if accel not in ("pure-c", "cmsis-nn"):
            fatal_error(
                f"Invalid --accel '{accel}' for --target arm",
                "Supported accel for ARM: pure-c, cmsis-nn")
    elif target == "riscv":
        if accel not in ("pure-c", "nmsis-nn", "nuclei-ai"):
            fatal_error(
                f"Invalid --accel '{accel}' for --target riscv",
                "Supported accel for RISC-V: pure-c, nmsis-nn, nuclei-ai")
    else:
        fatal_error(
            f"Invalid --target '{target}'",
            "Supported targets: arm, riscv")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = generate_c_code(
        model_info, output_dir, target,
        inference_func=inference_function_name,
        with_test_main=with_test_main)

    for filename, content in generated_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        info(f"Generated: {output_path}")

    generate_lut(output_dir)
    copy_files_to_build(output_dir, target, mode, accel)
    if accel != 'none':
        script_name = (
            f"build_{target}_{accel.replace('-', '_')}_{mode}.sh")
    else:
        script_name = f"build_{target}_{mode}.sh"

    if run:
        script_path = output_dir / script_name
        try:
            script_path.chmod(0o755)
        except OSError:
            pass
        info(f"Executing: {script_path} {model_path}")
        result = subprocess.run(
            [str(script_path.resolve()), model_path],
            cwd=output_dir)
        sys.exit(result.returncode)

    info(f"Done! Output directory: {output_dir}")
    info("\nNext steps:")
    info(f"  cd {output_dir}")
    info(f"  ./{script_name} {model_path}")


global_parent = argparse.ArgumentParser(add_help=False)
global_parent.add_argument("--verbose", "-v", action="store_true")
global_parent.add_argument(
    "--target", "-t", type=str, default="riscv", choices=["riscv", "arm"]
)
global_parent.add_argument(
    "--mode", type=str, default="release", choices=["debug", "release"]
)
global_parent.add_argument(
    "--accel", type=str, default="pure-c", choices=["cmsis-nn", "nmsis-nn", "pure-c"]
)
global_parent.add_argument("--with-test-main", action="store_true")
global_parent.add_argument("--run", action="store_true")
global_parent.add_argument(
    "--inference-function-name", type=str, default="tinymlc_inference"
)
global_parent.add_argument("--output-dir", type=str, default=".")
global_parent.add_argument("--table-name", type=str)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TinyMLC - TinyML Compiler CLI",
        parents=[global_parent],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ---- generate ----
    gen_parser = subparsers.add_parser(
        "generate", parents=[global_parent], help="Generate a network"
    )
    gen_parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "detection", "segmentation"],
    )
    gen_parser.add_argument("--input-shape", type=parse_shape, default=[1, 28, 28, 1])
    gen_parser.add_argument("--output-shape", type=parse_shape, default=[1, 10])
    gen_parser.add_argument("--max-macs", type=int, default=100000)
    gen_parser.add_argument("--max-ram", type=int, default=30)
    gen_parser.add_argument("--max-flash", type=int, default=64)
    gen_parser.add_argument("--clock-speed", type=int, default=100000000)
    gen_parser.add_argument("--dump-model", type=str)
    gen_parser.add_argument("--icount-shift", type=int, default=0)
    gen_parser.add_argument("--qemu-cpu", type=str, default="cortex-m4")
    gen_parser.add_argument(
        "--estimator",
        type=str,
        default="software",
        choices=["software", "qemu", "hardware"],
    )
    gen_parser.add_argument("--estimator-script", type=str)
    gen_parser.add_argument("--estimator-function", type=str, default="estimate")
    gen_parser.add_argument(
        "--generate-mode", type=str, default="genetic", choices=["random", "genetic"]
    )
    gen_parser.add_argument("--population", type=int, default=50)
    gen_parser.add_argument("--generations", type=int, default=50)
    gen_parser.add_argument("--early-stop", type=int, default=10)
    gen_parser.add_argument("--num-samples", type=int, default=100)

    # ---- table ----
    table_parser = subparsers.add_parser(
        "table", parents=[global_parent], help="Manage hardware profile tables"
    )
    table_parser.add_argument("--build", action="store_true")
    table_parser.add_argument("--update", action="store_true")
    table_parser.add_argument("--show", action="store_true")
    table_parser.add_argument("--stats", action="store_true")
    table_parser.add_argument("--add-ops", type=str)
    table_parser.add_argument("--recalibrate", action="store_true")
    table_parser.add_argument("--entries", type=str)
    table_parser.add_argument("--board", type=str, default="unknown")

    # ---- convert ----
    convert_parser = subparsers.add_parser(
        "convert", parents=[global_parent], help="Convert an existing model"
    )
    convert_parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "generate":
        return handle_generate(args)
    elif args.command == "table":
        return handle_table(args)
    elif args.command == "convert":
        return handle_convert(args)
    else:
        fatal_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
