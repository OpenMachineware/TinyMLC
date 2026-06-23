#!/usr/bin/env python3
# cli.py - CLI TinyMLC

import argparse
import sys
from typing import List

from handlers import handle_generate, handle_table, handle_convert
from utils.dump import fatal_error
from tinymlc.ang.args import parse_shape


global_parent = argparse.ArgumentParser(add_help=False)
global_parent.add_argument("--verbose", "-v", action="store_true")
global_parent.add_argument(
    "--target", "-t", type=str, default="riscv", choices=["riscv", "arm", "host"]
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
