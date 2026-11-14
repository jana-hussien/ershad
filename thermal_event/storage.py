"""Free-space checks, so every stage stops before it can fill the disk.

MARGIN is kept free on top of what a stage writes: with 8 GB of RAM, macOS swaps to disk while YOLO
or training runs, and a full disk makes the whole machine stall.
"""
import os
import shutil
import sys
from pathlib import Path

GB = 1024 ** 3
MARGIN = float(os.environ.get("THERMAL_MARGIN_GB", 2)) * GB  # lower it only when the disk is tight


def free_bytes(path):
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def require_free(path, need, what):
    """Exit with a clear message unless `need` bytes plus MARGIN are free where `path` lives."""
    free = free_bytes(path)
    if free < need + MARGIN:
        sys.exit(f"Not enough disk space for {what}: needs about {need / GB:.1f} GB plus "
                 f"{MARGIN / GB:.1f} GB spare, but only {free / GB:.1f} GB is free. "
                 f"Free up {(need + MARGIN - free) / GB:.1f} GB and run again.")
    print(f"disk check ok for {what}: {free / GB:.1f} GB free, needs about {need / GB:.1f} GB "
          f"(+{MARGIN / GB:.1f} GB spare)")
