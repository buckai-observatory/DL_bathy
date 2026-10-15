"""Command-line interface: `sdb split`, `sdb train` and `sdb init-config`."""

import argparse
import os
import sys

from . import __version__
from .config import example_config, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sdb",
        description="DL_bathy: prepare Sentinel-2/LiDAR patches and train a DeepLabV3+ bathymetry model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="align, split, balance and augment training patches")
    p_split.add_argument("-c", "--config", help="YAML config file (default: built-in defaults)")

    p_train = sub.add_parser("train", help="train the model, evaluate it and write predictions")
    p_train.add_argument("-c", "--config", help="YAML config file (default: built-in defaults)")

    p_init = sub.add_parser("init-config", help="write an example config file with every setting")
    p_init.add_argument("path", nargs="?", default="dl_bathy_config.yml")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")

    args = parser.parse_args(argv)

    if args.command == "init-config":
        if os.path.exists(args.path) and not args.force:
            parser.error(f"{args.path} exists; use --force to overwrite")
        with open(args.path, "w") as f:
            f.write(example_config())
        print(f"Wrote {args.path}")
        return 0

    if args.command == "split":
        from .split import run
        run(load_config(args.config, "split"))
    else:
        from .train import run
        run(load_config(args.config, "train"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
