import sys


def fatal_error(op_name, reason, suggestion=None):
    """致命错误：打印错误信息并退出"""
    print("\n" + "=" * 60)
    print(f"错误: {op_name} 算子解析失败")
    print(f"原因: {reason}")
    if suggestion:
        print(f"\n建议: {suggestion}")
    print("=" * 60)
    sys.exit(1)


def warning(msg, suggestion=None):
    """警告：打印但不退出"""
    print(f"警告: {msg}")
    if suggestion:
        print(f"  {suggestion}")
