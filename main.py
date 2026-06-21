#!/usr/bin/env python3
"""
tinymlc - TinyML Compiler
Command line entry point
"""

import sys
from tinymlc.translator import main

if __name__ == "__main__":
    sys.exit(main())
