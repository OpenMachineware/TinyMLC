#!/usr/bin/env python3
"""
tinymlc - TinyML Compiler
命令行入口
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 main.py 中的 main 函数
from src.tinymlc.main import main

if __name__ == "__main__":
    sys.exit(main())
