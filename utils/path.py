#!/usr/bin/env python3
"""Path utilities for TinyMLC CLI"""

from pathlib import Path
from argparse import Namespace


def get_output_dir(args: Namespace) -> Path:
    """Get and create output directory from args"""
    out_dir = Path(getattr(args, "output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
