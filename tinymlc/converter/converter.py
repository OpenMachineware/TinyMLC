#!/usr/bin/env python3
"""
TinyMLC - TinyML Compiler
Convert TFLite/ONNX models to C code executable on MCU

This module is a thin wrapper around handlers.handle_convert.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from handlers import handle_convert


def main():
    """
    Simple CLI wrapper that calls handle_convert.
    Uses minimal argument parsing and delegates to the unified handler.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="tinymlc - TinyML Compiler")
    parser.add_argument(
        "model",
        help="TFLite or ONNX model file path")
    parser.add_argument(
        "--entry-point", default="tinymlc_inference",
        help="Inference function name (default: tinymlc_inference)")
    parser.add_argument(
        "--with-test-main", action="store_true",
        help="Generate test main function")
    parser.add_argument(
        "--run", action="store_true",
        help="Run build script after generation")
    parser.add_argument(
        "--arch", default="riscv",
        help="Target chip architecture (default: riscv)")
    parser.add_argument(
        "--accel", default="none",
        help="Acceleration library type: none, cmsis-nn, "
             "nmsis-nn, nuclei-ai, ...")
    parser.add_argument(
        "--acc-lib-inc",
        default="third_party/CMSIS-NN-7.0.0/Include",
        help="Operator acceleration library header path")
    parser.add_argument(
        "--acc-lib-lib",
        default="third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a",
        help="Operator acceleration library static library path")
    parser.add_argument(
        "-o", "--output-dir", default="tinymlc_generated",
        help="Output directory (default: tinymlc_generated)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed info")

    args = parser.parse_args()

    # Map old arch to new target
    target = args.arch

    # Map old accel to new accel (map 'none' to 'pure-c')
    accel = args.accel if args.accel != 'none' else 'pure-c'

    # Map old entry-point to new inference-function-name
    inference_function_name = args.entry_point

    # Map old with_test_main to new with_test_main
    with_test_main = args.with_test_main

    # Map old run to new run
    run = args.run

    # Map old verbose to new verbose
    verbose = args.verbose

    # For this legacy CLI, mode is determined by with_test_main
    mode = "debug" if with_test_main else "release"

    # Build namespace for handle_convert
    class Args:
        pass
    ns = Args()
    ns.model = args.model
    ns.target = target
    ns.accel = accel
    ns.mode = mode
    ns.inference_function_name = inference_function_name
    ns.with_test_main = with_test_main
    ns.output_dir = args.output_dir
    ns.verbose = verbose
    ns.run = run

    return handle_convert(ns)


if __name__ == "__main__":
    sys.exit(main())
