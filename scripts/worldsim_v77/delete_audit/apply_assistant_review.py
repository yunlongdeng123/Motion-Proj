"""写入有图像依据的单帧粗分类，保持用户评分为空。"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    manifest = root / "review/review_manifest.json"
    data = json.loads(manifest.read_text())
    notes = json.loads(args.annotations.read_text(encoding="utf-8"))
    by_id = {r["clip_id"]: r for r in notes["reviews"]}
    assert len(by_id) == 70 and set(by_id) == {r["clip_id"] for r in data["clips"]}
    backup = root / "review_manifest.before_assistant_review.json"
    if not backup.exists():
        shutil.copy2(manifest, backup)
        shutil.copy2(root / "review/index.html", root / "index.before_assistant_review.html")
    for row in data["clips"]:
        note = by_id[row["clip_id"]]
        assert row["human_verdict"] is None
        row.update(assistant_one_frame_issue=note["issue"], assistant_review_note=note["note"],
                   input_qualification=note["input_qualification"],
                   assistant_review_scope="one_frozen_prompt_frame_only",
                   one_frame_components=f"assets/{row['clip_id']}/one_frame_components.jpg")
        override = notes["difficulty_overrides"].get(row["clip_id"])
        row["assistant_input_difficulty"] = override["value"] if override else row["input_difficulty_proxy"]
        row["assistant_difficulty_note"] = override["reason"] if override else "沿用冻结的GT难度代理；结合本帧可见性记录"
    data["assistant_review_summary"] = {
        "scope": "one_frozen_prompt_frame_per_clip_not_video_pass_rate",
        "reviewed_clips": 70,
        "issue_counts": dict(Counter(r["assistant_one_frame_issue"] for r in data["clips"])),
        "input_qualification_counts": dict(Counter(r["input_qualification"] for r in data["clips"])),
        "input_difficulty_counts": dict(Counter(r["assistant_input_difficulty"] for r in data["clips"])),
        "temporal_review": "not_scored_from_stills",
        "human_verdict": None,
    }
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(args.annotations, root / "review/assistant_reviews.json")
    print(json.dumps(data["assistant_review_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
