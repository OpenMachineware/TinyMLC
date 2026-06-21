# tinymlc/utils.py
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
