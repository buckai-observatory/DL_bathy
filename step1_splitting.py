"""Backward-compatible entry point; equivalent to `sdb split --config <file>`.

The patch-preparation code now lives in dl_bathy/split.py and its settings
in a YAML config file (write an example with `sdb init-config`).
"""
import sys

from dl_bathy.cli import main

if __name__ == "__main__":
    sys.exit(main(["split", *sys.argv[1:]]))
