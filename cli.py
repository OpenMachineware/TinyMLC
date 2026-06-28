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
# Entry point for TinyMLC command line interface

import argparse
import sys

from handlers import handle_generate, handle_convert
from utils.dump import fatal_error
from TinyMLC.ANG.args import parse_shape


global_parent = argparse.ArgumentParser(add_help=False)
global_parent.add_argument("--verbose", "-v", action="store_true")
global_parent.add_argument(
    "--target", "-t", type=str, default="riscv",
    choices=["riscv", "arm", "host"]
)
global_parent.add_argument(
    "--mode", type=str, default="release", choices=["debug", "release"])
global_parent.add_argument(
    "--accel", type=str, default="pure-c",
    choices=["cmsis-nn", "nmsis-nn", "pure-c"])
global_parent.add_argument(
    "--accel-lib-inc", type=str,
    help="Path to accelerator library include directory")
global_parent.add_argument(
    "--accel-lib-lib", type=str, help="Path to accelerator library lib file")

global_parent.add_argument("--with-test-main", action="store_true")
global_parent.add_argument("--run", action="store_true")
global_parent.add_argument(
    "--inference-function-name", type=str, default="tinymlc_inference")
global_parent.add_argument("--output-dir", type=str, default=".")
global_parent.add_argument("--dump-model", type=str)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TinyMLC - TinyML Compiler CLI",
        parents=[global_parent],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute"
    )

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
    gen_parser.add_argument(
        "--input-shape", type=parse_shape, default=[1, 28, 28, 1]
    )
    gen_parser.add_argument("--output-shape", type=parse_shape, default=[1, 10])
    gen_parser.add_argument("--max-macs", type=int, default=100000)
    gen_parser.add_argument("--max-ram", type=int, default=30)
    gen_parser.add_argument("--max-flash", type=int, default=64)
    gen_parser.add_argument("--clock-speed", type=int, default=100000000)
    gen_parser.add_argument("--icount-shift", type=int, default=0)
    gen_parser.add_argument("--qemu-cpu", type=str, default="cortex-m4")
    gen_parser.add_argument(
        "--estimator",
        type=str,
        default="software",
        choices=["software", "qemu", "hardware"],
    )
    gen_parser.add_argument("--estimator-script", type=str)
    gen_parser.add_argument(
        "--estimator-function", type=str, default="estimate"
    )
    gen_parser.add_argument(
        "--generate-mode", type=str, default="genetic",
        choices=["random", "genetic"])
    gen_parser.add_argument("--population", type=int, default=50)
    gen_parser.add_argument("--generations", type=int, default=50)
    gen_parser.add_argument("--early-stop", type=int, default=10)
    gen_parser.add_argument("--num-samples", type=int, default=100)

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
    elif args.command == "convert":
        return handle_convert(args)
    else:
        fatal_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
