#!/usr/bin/env python3
"""
Command handlers for TinyMLC CLI.
Contains handle_generate, handle_table, handle_convert.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tinymlc.ANG.model_generator import ModelGenerator
from tinymlc.ANG.table import TableManager
from tinymlc.ANG.args import create_estimator, get_table_name, parse_shape
from utils.path import get_output_dir
from tinymlc.codegen import generate_c_code, copy_files_to_build
from tinymlc.generate_lut import generate_lut
from tinymlc.transform.pass_manager import PassManager
from utils.dump import fatal_error, warning, info, dump_model_info


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

    # Get target, mode, accel for code generation and build
    target = getattr(args, "target", "riscv")
    mode = getattr(args, "mode", "release")
    accel = getattr(args, "accel", "pure-c")
    run = getattr(args, "run", False)

    # Validate target/accel/mode combination
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
    elif target == "host":
        # Host only supports pure-c and debug mode
        if accel != "pure-c":
            fatal_error(
                f"Invalid --accel '{accel}' for --target host",
                "Host only supports pure-c (no hardware acceleration)")
        if mode != "debug":
            warning("Host target only supports debug mode, forcing mode=debug")
            mode = "debug"

    # ---- Optimization passes ----
    pm = PassManager.default_pipeline()
    info("Running optimization passes...")
    optimized_model_info = pm.run(model_info)
    pm.dump_summary()

    info("Generating C code...")
    out_dir = get_output_dir(args)
    result = generate_c_code(
        model_info=optimized_model_info,
        output_dir=str(out_dir),
        target=target,
        inference_func=getattr(args, "inference_function_name", "tinymlc_inference"),
        with_test_main=True,  # Always generate test main for generated networks
    )

    for filename, content in result.items():
        filepath = out_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        info(f"Generated: {filepath}")

    # Export weights (needed for ANG-generated models)
    from tinymlc.converter.export_weights import export_model_weights
    quant_scales = export_model_weights(out_dir, optimized_model_info)
    optimized_model_info["quant_scales"] = quant_scales

    # Generate LUT and copy build files
    generate_lut(out_dir)

    # Compute accel library paths if not provided
    project_root = Path(__file__).parent
    accel_lib_inc = getattr(args, "accel_lib_inc", None)
    accel_lib_lib = getattr(args, "accel_lib_lib", None)

    if accel == "cmsis-nn" and (not accel_lib_inc or not accel_lib_lib):
        accel_lib_inc = str(project_root / "third_party" / "CMSIS-NN-7.0.0" / "Include")
        accel_lib_lib = str(project_root / "third_party" / "CMSIS-NN-7.0.0" / "Lib" / "libcmsis-nn.a")
    elif accel == "nmsis-nn" and (not accel_lib_inc or not accel_lib_lib):
        accel_lib_inc = str(project_root / "third_party" / "NMSIS-1.6.0" / "Include")
        accel_lib_lib = str(project_root / "third_party" / "NMSIS-1.6.0" / "Lib" / "libNMSISNN.a")

    if accel == "pure-c":
        copy_files_to_build(out_dir, target, mode, accel)
    copy_files_to_build(out_dir, target, mode, accel, accel_lib_inc, accel_lib_lib)

    # Determine script name (must match copy_files_to_build logic)
    if target == "host":
        script_name = "build_host_debug.sh"
    elif accel != 'none':
        script_name = f"build_{target}_{accel.replace('-', '_')}_{mode}.sh"
    else:
        script_name = f"build_{target}_{mode}.sh"

    if run:
        script_path = out_dir / script_name
        try:
            script_path.chmod(0o755)
        except OSError:
            pass
        info(f"Executing: {script_path}")
        result = subprocess.run(
            [str(script_path.resolve())],
            cwd=out_dir)
        sys.exit(result.returncode)

    info(f"Done! Output directory: {out_dir}")
    info("\nNext steps:")
    info(f"  cd {out_dir}")
    info(f"  ./{script_name}")

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
    output_dir = Path(getattr(args, "output_dir", "."))
    with_test_main = getattr(args, "with_test_main", None)
    inference_function_name = getattr(args, "inference_function_name", None)
    verbose = getattr(args, "verbose", None)
    run = getattr(args, "run", None)

    if not model_path:
        fatal_error("--model is required for convert")
    if not Path(model_path).exists():
        fatal_error(f"Model file not found: {model_path}", "Please check file path")

    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path.endswith(".tflite"):
        # Lazy import to avoid dependency when not using tflite
        from tinymlc.converter.parser_litert import (
            parse_model_tflite, extract_all_weights_litert)
        from tinymlc.converter.export_weights import export_model_weights
        model_info = parse_model_tflite(model_path)
        # Extract weights (interpreter created internally)
        extract_all_weights_litert(model_path, model_info)
        # Export weights using unified function
        quant_scales = export_model_weights(output_dir, model_info)
        model_info["quant_scales"] = quant_scales
    elif model_path.endswith(".onnx"):
        from tinymlc.converter.parser_onnx import (
            parse_model_onnx, extract_all_weights_onnx)
        from tinymlc.converter.export_weights import export_model_weights
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
    elif target == "host":
        # Host only supports pure-c and debug mode
        if accel != "pure-c":
            fatal_error(
                f"Invalid --accel '{accel}' for --target host",
                "Host only supports pure-c (no hardware acceleration)")
        if mode != "debug":
            warning("Host target only supports debug mode, forcing mode=debug")
            mode = "debug"
    else:
        fatal_error(
            f"Invalid --target '{target}'",
            "Supported targets: arm, riscv, host")

        # ---- Optimization passes ----
        pm = PassManager.default_pipeline()
        info("Running optimization passes...")
        optimized_model_info = pm.run(model_info)
        pm.dump_summary()

    info("Generating C code...")
    generated_files = generate_c_code(
        optimized_model_info, output_dir, target,
        inference_func=inference_function_name,
        with_test_main=with_test_main)

    for filename, content in generated_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        info(f"Generated: {output_path}")

    generate_lut(output_dir)

    # Compute accel library paths if not provided
    project_root = Path(__file__).parent
    accel_lib_inc = getattr(args, "accel_lib_inc", None)
    accel_lib_lib = getattr(args, "accel_lib_lib", None)

    if accel == "cmsis-nn" and (not accel_lib_inc or not accel_lib_lib):
        accel_lib_inc = str(project_root / "third_party" / "CMSIS-NN-7.0.0" / "Include")
        accel_lib_lib = str(project_root / "third_party" / "CMSIS-NN-7.0.0" / "Lib" / "libcmsis-nn.a")
    elif accel == "nmsis-nn" and (not accel_lib_inc or not accel_lib_lib):
        accel_lib_inc = str(project_root / "third_party" / "NMSIS-1.6.0" / "Include")
        accel_lib_lib = str(project_root / "third_party" / "NMSIS-1.6.0" / "Lib" / "libNMSISNN.a")

    copy_files_to_build(output_dir, target, mode, accel, accel_lib_inc, accel_lib_lib)

    # Determine script name
    if target == "host":
        script_name = "build_host_debug.sh"
    elif accel != 'none':
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
