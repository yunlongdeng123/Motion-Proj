"""导出 V5 object-only replay 的人工复核包并聚合 verdict。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..cache.dataset import ProjectionCacheDataset
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import utc_now
from ..runtime.fingerprint import file_fingerprint, git_state, sha256_json
from ..utils.io import write_video
from ..utils.viz import hstack_panels


VALID_VERDICTS = {"yes", "no", "uncertain"}
PROTOCOL_VERSION = "p2-v2-replay-v5-object-review-v1"


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    import cv2

    output = image.copy()
    cv2.putText(output, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def _uint8_rgb(frames: torch.Tensor) -> np.ndarray:
    value = frames.detach().float().cpu()
    if value.min() < -0.01:
        value = (value + 1.0) / 2.0
    return (value.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def build_panel(base_rgb: torch.Tensor, projected_rgb: torch.Tensor, object_mask: torch.Tensor) -> np.ndarray:
    """把 object-only target 的来源、修正、mask 和差异并列，static 明确标为禁用。"""
    base = _uint8_rgb(base_rgb)
    projected = _uint8_rgb(projected_rgb)
    mask = F.interpolate(object_mask.float(), size=base.shape[1:3], mode="nearest")
    mask_rgb = (mask.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    difference = (np.abs(projected.astype(np.int16) - base.astype(np.int16)).clip(0, 48) * (255 / 48)).astype(np.uint8)
    frames = []
    for index in range(base.shape[0]):
        rendered_mask = np.repeat(mask_rgb[index, 0, ..., None], 3, axis=-1)
        frames.append(hstack_panels(
            _label_panel(base[index], "Base rollout"),
            _label_panel(projected[index], "Projected (object-only)"),
            _label_panel(rendered_mask, "Object mask; static disabled"),
            _label_panel(difference[index], "|Projected - Base|"),
        ))
    return np.stack(frames)


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_template(run_dir: Path, cases: list[dict]) -> None:
    template = run_dir / "reviews.template.jsonl"
    if template.exists():
        return
    rows = [
        {
            "case_id": case["case_id"], "object_correction_reasonable": "uncertain",
            "reviewer": "human", "notes": "",
        }
        for case in cases
    ]
    atomic_write_text(str(template), "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_write_text(
        str(run_dir / "REVIEW_README.md"),
        "# P2-V2 V5 object-only 人工复核\n\n"
        "每个 panel 为 `[Base rollout | Projected object-only | Object mask（static 已禁用） | abs difference]`。\n"
        "判 `yes`：局部 object 修正与可见运动/支撑关系相符，且未引入明显撕裂或主体破坏；"
        "判 `no`：局部修正明显错位、漂移或破坏主体；判 `uncertain`：无法可靠判断。\n\n"
        "复制 `reviews.template.jsonl` 为 `reviews.jsonl`，填写全部 20 条后运行同一命令附加 `--aggregate-only`。"
        "门槛：decisive verdict 的 yes 比例不低于 70%；未通过不得训练。\n",
    )


def export_reviews(cache_dir: Path, run_dir: Path, num_cases: int, seed: int) -> dict:
    stage_path = cache_dir / "_stage" / "manifest.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("status") != "completed" or not (cache_dir / "_stage" / "COMPLETE").is_file():
        raise RuntimeError("只能从完成的 V5 replay stage 导出人工复核")
    dataset = ProjectionCacheDataset(str(cache_dir), expected_fingerprint=str(stage["fingerprint"]))
    if len(dataset) < num_cases:
        raise ValueError(f"有效 V5 sample 只有 {len(dataset)}，不足 {num_cases} 个 review case")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num_cases))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "panels").mkdir(exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)
    cases = []
    for case_index, sample_index in enumerate(indices):
        item = dataset[sample_index]
        meta = item["metadata"]
        case_id = f"replay-{case_index:02d}-{meta['sample_id']}"
        panel_path = run_dir / "panels" / f"{case_id}.mp4"
        write_video(build_panel(item["base_rgb"], item["projected_rgb"], item["object_mask"]), str(panel_path), fps=4)
        case = {
            "case_id": case_id, "case_index": case_index, "sample_id": meta["sample_id"],
            "generation_seed": meta["generation_seed"], "condition_id": meta["condition_id"],
            "panel_path": str(panel_path), "cache_fingerprint": stage["fingerprint"],
            "object_valid_fraction": meta["object_valid_fraction"],
            "object_energy_before": meta["energy_before_by_component"]["object"],
            "object_energy_after": meta["energy_after_by_component"]["object"],
            "static_component": "disabled",
            "generated_track_diagnostics": meta["projector_diagnostics"].get("generated_tracks", {}),
        }
        atomic_write_json(str(run_dir / "cases" / f"{case_id}.json"), case)
        cases.append(case)
    _write_template(run_dir, cases)
    manifest = {
        "protocol": PROTOCOL_VERSION, "status": "awaiting_reviews", "run_id": run_dir.name,
        "cache_dir": str(cache_dir), "cache_fingerprint": stage["fingerprint"], "seed": seed,
        "selected_indices": indices, "git": git_state(), "started_at": utc_now(),
    }
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    summary = {"status": "awaiting_reviews", "cases": len(cases), "required_reviews": num_cases,
               "minimum_reasonable_rate": 0.70, "cache_fingerprint": stage["fingerprint"]}
    atomic_write_json(str(run_dir / "summary.json"), summary)
    return summary


def aggregate_reviews(run_dir: Path) -> dict:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "cases").glob("*.json"))]
    reviews = {str(row.get("case_id")): row for row in _load_rows(run_dir / "reviews.jsonl")}
    valid = [reviews[case["case_id"]] for case in cases if str(reviews.get(case["case_id"], {}).get("object_correction_reasonable")) in VALID_VERDICTS]
    decisive = [row for row in valid if row["object_correction_reasonable"] != "uncertain"]
    yes = sum(row["object_correction_reasonable"] == "yes" for row in decisive)
    rate = yes / len(decisive) if decisive else None
    completed = len(valid) == len(cases)
    accepted = bool(completed and rate is not None and rate >= 0.70)
    summary = {
        "protocol": PROTOCOL_VERSION, "status": "completed" if completed else "awaiting_reviews",
        "cases": len(cases), "reviews_completed": len(valid), "decisive": len(decisive), "yes": yes,
        "reasonable_rate": rate, "minimum_reasonable_rate": 0.70,
        "decision": "promote" if accepted else ("blocked" if completed else "pending_review"),
        "review_fingerprint": file_fingerprint(str(run_dir / "reviews.jsonl")) if (run_dir / "reviews.jsonl").is_file() else None,
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": summary["status"], "ended_at": utc_now(), "decision": summary["decision"]})
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    if completed:
        atomic_write_text(str(run_dir / "COMPLETE"), sha256_json(summary) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    result = aggregate_reviews(Path(args.run_dir)) if args.aggregate_only else export_reviews(
        Path(args.cache_dir), Path(args.run_dir), args.num_cases, args.seed,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
