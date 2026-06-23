# tinymlc/dump.py
import sys

try:
    from termcolor import cprint
except ImportError:
    # Fallback: use plain print when termcolor is unavailable
    def cprint(msg, color=None, attrs=None):
        print(msg)


def info(msg):
    """Print info message"""
    cprint(f"[INFO] {msg}", "cyan")


def warning(msg, suggestion=None):
    """Print warning message (does not exit)"""
    cprint(f"[WARNING] {msg}", "yellow")
    if suggestion:
        cprint(f"  SUGGESTION: {suggestion}", "yellow")


def fatal_error(msg, suggestion=None):
    """Print fatal error message and exit"""
    cprint("\n" + "=" * 60, "red")
    cprint(f"[ERROR] {msg}", "red", attrs=["bold"])
    if suggestion:
        cprint(f"\n  SUGGESTION: {suggestion}", "red")
    cprint("\n" + "=" * 60, "red")
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
