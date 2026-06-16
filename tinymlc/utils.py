# tinymlc/utils.py
import sys

def info(msg):
    """普通信息"""
    print(f"Info: {msg}")

def warning(msg, suggestion=None):
    """警告：打印但不退出"""
    print(f"警告: {msg}")
    if suggestion:
        print(f"  建议: {suggestion}")

def fatal_error(msg, suggestion=None):
    """致命错误：打印错误信息并退出"""
    print("\n" + "=" * 60)
    print(f"错误: {msg}")
    if suggestion:
        print(f"\n建议: {suggestion}")
    print("=" * 60)
    sys.exit(1)
