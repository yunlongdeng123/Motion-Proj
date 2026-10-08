"""将 R001 已评分的单帧伪标签制成静态重复窗；保留全部候选评分。"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def checked_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"score 必须是有限数值: {value!r}")
    return value


def prepare(manifest: Path, out: Path, select: str, selection_review: Path):
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest 必须是非空候选列表；无合格项时仍保留低分记录")
    choice=json.loads(selection_review.read_text(encoding='utf-8'))
    if choice.get('selected_candidate_id') != select:
        raise ValueError('显式选择与独立候选取舍记录不一致')
    accepted, excluded, eligible, ids = [], [], [], set()
    for row in rows:
        cid = row["candidate_id"]
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cid) or cid in ids:
            raise ValueError(f"candidate_id 非法或重复: {cid!r}")
        ids.add(cid)
        frame = row["source_frame_index"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame not in range(10):
            raise ValueError(f"{cid}: source_frame_index 必须为 0..9")
        score = checked_score(row["score"])
        qa_path = Path(row["qa_record"])
        if not qa_path.is_absolute() or not qa_path.is_file():
            raise ValueError(f"{cid}: qa_record 必须是已有绝对路径")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if checked_score(qa.get("score")) != score or qa.get("candidate_id", cid) != cid:
            raise ValueError(f"{cid}: QA 的 candidate_id/score 与 manifest 不一致")
        if score <= 1:
            excluded.append({"candidate_id": cid, "source_frame_index": frame, "score": score, "reason": "score<=1"})
            continue
        eligible.append(cid)
        if cid != select:
            excluded.append({'candidate_id':cid,'source_frame_index':frame,'score':score,'reason':'eligible_but_not_selected_best1'})
            continue
        if choice.get('selected_source_frame_index') != frame:
            raise ValueError('入选来源帧与独立选择记录不一致')
        images = {}
        for role in ("x", "y", "hole"):
            path = Path(row[role])
            if not path.is_absolute() or path.suffix.lower() != ".png" or not path.is_file():
                raise ValueError(f"{cid}: {role} 必须是已有绝对 PNG 路径")
            images[role] = np.asarray(Image.open(path).convert("L" if role == "hole" else "RGB"))
        x, y, h = images["x"], images["y"], images["hole"]
        if x.shape != y.shape or x.shape != (576, 1024, 3) or h.shape != x.shape[:2]:
            raise ValueError(f"{cid}: R001 X/Y/H 必须是同尺寸 576x1024")
        if not set(np.unique(h)).issubset({0, 255}) or not h.any():
            raise ValueError(f"{cid}: 洞必须是非空二值 mask")
        if np.any((x != y).any(-1) & (h == 0)) or not np.any((x != y).any(-1) & (h > 0)):
            raise ValueError(f"{cid}: Y 必须仅在洞内与 X 不同，且洞内不能原样复制 X")
        accepted.append(row)
    if len(accepted)!=1:
        raise ValueError('第一关必须且仅选择一个 score>1 的已审核候选')
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "selection.json").write_text(json.dumps({"rule": "score>1", "eligible":eligible,"selected": [r["candidate_id"] for r in accepted],
        "sampling_strategy":"best1_per_case","selection_review":choice,"excluded": excluded, "static_repeat_capacity_only": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "qa_all").mkdir()
    for row in rows:
        shutil.copy2(row["qa_record"], out / "qa_all" / f"{row['candidate_id']}.json")
    for row in accepted:
        cid = row["candidate_id"]
        root = out / "pairs" / cid
        for role in ("x", "y", "hole"):
            folder = root / role
            folder.mkdir(parents=True)
            for i in range(10):
                shutil.copy2(row[role], folder / f"{i:05}.png")
        shutil.copy2(row["qa_record"], root / "quality_review.json")
        shutil.copy2(row["y"], root / "f05.png")
        pair = {"candidate_id": cid, "source_case": "R001", "source_frame_index": row["source_frame_index"],
            "review_slot": 5, "frame_index": 5, "num_frames": 10, "static_repeat_capacity_only": True,
            "supervision_kind": "reviewed_pseudo_clean_target", "score": row["score"], "qa_pass": True,
            "qa_record": str((root / "quality_review.json").resolve()),
            **{role: str((root / role).resolve()) for role in ("x", "y", "hole")}}
        (root / "pair.json").write_text(json.dumps(pair, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"R001 候选 {len(rows)}，严格 score>1 准入 {len(eligible)}，best1 实际入训 {len(accepted)}，其余保留；仅静态容量诊断")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument('--select',required=True)
    parser.add_argument('--selection-review',required=True,type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.outdir, args.select, args.selection_review)
