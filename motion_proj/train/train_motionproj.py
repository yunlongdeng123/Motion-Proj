"""投影蒸馏微调（阶段 3）的命令行入口。"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..utils.logging import get_logger
from .trainer import Trainer

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
