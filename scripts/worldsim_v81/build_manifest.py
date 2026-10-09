"""建立 split 明确的 RGB 清单；结果保存到仓库外。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motion_proj.worldsim_v81.data import discover_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=25)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repo):
        parser.error("数据清单请保存到仓库外的 runs 目录")
    accepted, rejected = discover_videos(args.rgb_root, args.split, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in accepted), encoding="utf-8")
    args.output.with_suffix(".rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "split": args.split}))
    if not accepted:
        raise SystemExit("没有可用 RGB clip；不启动训练。")


if __name__ == "__main__":
    main()
