"""Backward-compatible entry point; equivalent to `sdb train --config <file>`.

The training code now lives in dl_bathy/train.py and its settings in a YAML
config file (write an example with `sdb init-config`).
"""
import sys

from dl_bathy.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
