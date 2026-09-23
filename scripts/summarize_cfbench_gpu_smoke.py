"""汇总本轮实际输出和工程边界；不生成质量分数或人工 verdict。"""
import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run_root, args.evidence_root
    out.mkdir(parents=True, exist_ok=True)
    omni = read(run / "omnidreams/result.json")
    gaussian = read(run / "gaussiandwm-r2/result.json")
    hugs = {b: read(run / f"hugsim-r2-{b}/result.json") for b in ["factual", "counterfactual"]}
    assert omni["status"] == gaussian["status"] == "complete"
    assert all(item["status"] == "complete" for item in hugs.values())
    hug_states = {b: read(run / f"hugsim-r2-{b}/{b}/states.json") for b in hugs}
    assert len(hug_states["factual"]) == len(hug_states["counterfactual"]) == 9
    assert all(f["ego_pos"] == c["ego_pos"] and f["timestamp"] == c["timestamp"]
               for f, c in zip(hug_states["factual"], hug_states["counterfactual"]))
    image_rows = []
    for camera in hugs["factual"]["branches"]["factual"]["views"]:
        for frame in range(9):
            arrays = [np.asarray(Image.open(run / f"hugsim-r2-{b}/{b}/{frame:03d}-{camera}.png")) for b in hugs]
            diff = np.any(arrays[0] != arrays[1], axis=-1)
            image_rows.append({"camera": camera, "frame": frame, "changed_pixels": int(diff.sum()), "total_pixels": int(diff.size),
                               "mean_absolute_rgb_difference": float(np.abs(arrays[0].astype(float) - arrays[1].astype(float)).mean())})
    evidence = {
        "task_id": "WS-V75-DOWNSTREAM-GPU-SMOKE-01", "run_id": "20260923-r1",
        "resource": "single NVIDIA RTX 3090 24GB", "seed": 42,
        "complete_runtime_smoke_models": ["omnidreams", "gaussiandwm_qa_compat", "hugsim"],
        "fresh_paired_smokes": 2, "benchmark_completed_cases": 0,
        "six_dimension_scores": {name: None for name in ["adherence", "physics", "environment_preservation", "outcome", "trajectory_adherence", "object_background_preservation"]},
        "human_verdict": None, "no_composite_score": True, "run_root": str(run),
        "omnidreams": omni, "gaussiandwm": gaussian, "hugsim": hugs,
        "hugsim_pair_diagnostics": {"same_ego_states_and_timestamps": True, "actor_count_change": [1, 0], "image_differences": image_rows},
        "resim": {"status": "oom_stopped", "stage": "first_stage_VAE_encoding", "generated_videos": 0,
                  "requested_allocation_gib": 5.36, "free_gib_at_failure": 4.46, "automatic_retry_after_oom": False,
                  "log": str(run / "resim-factual-r2.log")},
        "street_gaussians": {"status": "blocked_missing_checkpoint", "new_render_calls": 0,
                            "prior_evidence_only": True, "missing_extensions": ["simple_knn", "diff_gaussian_rasterization"],
                            "restoration_boundary": "历史自训checkpoint已退役；本轮未训练、未用其他方法资产冒充"},
        "driveeditor": {"status": "excluded_by_current_request", "gpu_calls": 0},
        "failure_ledger_refs": [], "failure_ledger_delta": "none",
    }
    (out / "summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    for name in ["omnidreams", "gaussiandwm", "gaussiandwm-r2", "hugsim", "hugsim-r2-factual", "hugsim-r2-counterfactual"]:
        shutil.copy2(run / name / "result.json", out / f"{name}-result.json")
    for name in ["resim-factual", "resim-factual-r2", "gaussiandwm", "gaussiandwm-r2", "hugsim", "hugsim-r2-factual", "hugsim-r2-counterfactual", "omnidreams"]:
        shutil.copy2(run / f"{name}.log", out / f"{name}.log")
    shutil.copy2(run / "gaussiandwm-r2/infer/qa_predictions_gpu-smoke-20260923.json", out / "gaussiandwm-prediction.json")
    for name in ["input-contract.json", "dataset-preflight.json"]:
        if (run / "resim-r2" / name).exists():
            shutil.copy2(run / "resim-r2" / name, out / f"resim-{name}")
    # 仅为审阅排列实际推理帧，不代填人工结论。
    panel = Image.new("RGB", (1280, 820), "#17202b")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), "GPU runtime smoke only - NOT the 24-case benchmark", fill="white")
    for column, branch in enumerate(["factual", "counterfactual"]):
        with imageio.get_reader(run / f"omnidreams/{branch}.mp4") as reader:
            frame = reader.get_data(60)
        x = column * 640
        panel.paste(Image.fromarray(frame).resize((640, 352)), (x, 65))
        draw.text((x+12, 42), f"OmniDreams {branch}: ego speed " + ("1.0x" if branch == "factual" else "0.5x"), fill="white")
        img = Image.open(run / f"hugsim-r2-{branch}/{branch}/008-CAM_FRONT.png")
        panel.paste(img.resize((640, 352)), (x, 457))
        draw.text((x+12, 432), f"HUGSIM {branch}: actor count " + ("1" if branch == "factual" else "0"), fill="white")
    panel.save(out / "paired-smoke-preview.jpg", quality=92)
    print(json.dumps({"runtime_smoke_complete": 3, "paired_smokes": 2, "formal_bench_cases": 0, "evidence_root": str(out)}, indent=2))


if __name__ == "__main__":
    main()
