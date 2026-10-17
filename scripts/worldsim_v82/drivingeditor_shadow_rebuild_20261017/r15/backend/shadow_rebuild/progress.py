"""只读进度，不启动训练；控制器可按此输出调整下一次检查间隔。"""
import argparse
import json
import time
from pathlib import Path
from .data import read_json


def status(root):
    root = Path(root)
    if not (root/"progress.json").exists():
        return dict(status="NOT_STARTED_OR_BEFORE_FIRST_STEP", action="read only; do not auto-launch")
    latest = read_json(root/"progress.json")
    records = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines() if line.strip()]
    stage = [r for r in records if r["stage"] == latest["stage"]]
    eta = None
    if len(stage) >= 2:
        a, b = stage[max(0, len(stage)-6)], stage[-1]
        delta = b["global_step"]-a["global_step"]
        elapsed = b["elapsed_seconds"]-a["elapsed_seconds"]
        # resume 会重置计时，发现非正 delta 不报错误 ETA。
        if delta > 0 and elapsed > 0:
            eta = elapsed/delta*(latest["stage_steps"]-latest["stage_step"])
    return dict(status="TRAINING_FINISHED" if (root/"completion.json").exists() else "LATEST_SAVED_PROGRESS",
                **latest, current_stage_remaining_seconds=eta,
                last_progress_age_seconds=max(0, time.time()-(root/"progress.json").stat().st_mtime),
                recommended_next_check_seconds=300 if eta is None else max(60, min(900, int(eta/4))),
                note="saved progress alone cannot prove worker liveness; no controller started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(status(parser.parse_args().output), ensure_ascii=False, indent=2))
