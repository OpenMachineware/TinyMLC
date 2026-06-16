# tinymlc/utils.py
import sys

try:
    from termcolor import cprint
except ImportError:
    # 回退：没有 termcolor 时使用普通 print
    def cprint(msg, color=None, attrs=None):
        print(msg)


def info(msg):
    """普通信息"""
    cprint(f"[INFO] {msg}", "cyan")


def warning(msg, suggestion=None):
    """警告：打印但不退出"""
    cprint(f"[WARNING] {msg}", "yellow")
    if suggestion:
        cprint(f"  SUGGESTION: {suggestion}", "yellow")


def fatal_error(msg, suggestion=None):
    """致命错误：打印错误信息并退出"""
    cprint("\n" + "=" * 60, "red")
    cprint(f"[ERROR] {msg}", "red", attrs=["bold"])
    if suggestion:
        cprint(f"\n  SUGGESTION: {suggestion}", "red")
    cprint("\n" + "=" * 60, "red")
    sys.exit(1)
