"""投影蒸馏微调（阶段 3）的命令行入口。"""
from __future__ import annotations

import argparse

import torch

from ..config import ConfigError, load_config
from ..utils.logging import get_logger
from .trainer import Trainer

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="auto | none | <checkpoint directory>")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    overrides = list(args.overrides)
    if args.resume is not None:
        overrides.append(f"train.resume={args.resume}")
    try:
        cfg = load_config(args.config, overrides)
        Trainer(cfg).train()
    except ConfigError:
        log.exception("配置校验失败，不自动重试")
        raise SystemExit(2)
    except FloatingPointError:
        log.exception("训练因 NaN/Inf 终止；必须创建 lr*0.5 的派生 run，不能修改原 run")
        raise SystemExit(42)
    except torch.cuda.OutOfMemoryError:
        log.exception("训练 OOM；原实验标记失败，不自动减小 batch")
        raise SystemExit(43)


if __name__ == "__main__":
    main()
