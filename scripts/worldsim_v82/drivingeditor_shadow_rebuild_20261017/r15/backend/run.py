"""统一命令入口。preflight 强制隐藏 CUDA；其余 GPU 动作都须显式 --execute。"""
import os
import sys
import runpy


def main():
    tasks = {"prepare": "prepare", "preflight": "train", "train": "train", "benchmark": "benchmark", "evaluate": "evaluate", "progress": "progress"}
    if len(sys.argv) < 2 or sys.argv[1] not in tasks:
        raise SystemExit("usage: python run.py {prepare,preflight,benchmark,train,evaluate,progress} ...")
    task = sys.argv.pop(1)
    if task in ("prepare", "preflight", "progress"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if task == "preflight" and "--execute" in sys.argv:
        raise SystemExit("preflight 不接受 --execute；不得从输入审计隐式启动 GPU")
    runpy.run_module("shadow_rebuild."+tasks[task], run_name="__main__")


if __name__ == "__main__":
    main()
