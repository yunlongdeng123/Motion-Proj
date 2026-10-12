"""Read paused-run TensorBoard scalars on CPU; never instantiate a model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(points, lo, hi):
    selected = [p for p in points if lo <= p[0] <= hi]
    if not selected:
        return {"n": 0}
    values = np.array([p[1] for p in selected], dtype=float)
    return {
        "n": len(selected), "first_step": selected[0][0], "last_step": selected[-1][0],
        "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "max": float(values.max()),
        "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    event = Path(args.event)
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
    accumulator.Reload()
    all_tags = accumulator.Tags()["scalars"]
    selected_tags = [tag for tag in all_tags if tag.startswith("train/")
                     and any(word in tag.lower() for word in ("loss", "psnr"))]
    rows = {tag: [[int(x.step), float(x.value), float(x.wall_time)]
                  for x in accumulator.Scalars(tag)] for tag in selected_tags}
    windows = {
        "19900_20000_pre_save": [19900, 20000],
        "20001_20100_post_save": [20001, 20100],
        "20900_21000_pre_reset": [20900, 21000],
        "21001_21100_post_reset": [21001, 21100],
        "23900_24000_pre_reset": [23900, 24000],
        "24001_24100_post_reset": [24001, 24100],
        "26900_27000_pre_reset": [26900, 27000],
        "27001_27100_post_reset": [27001, 27100],
        "29900_30000_pre_reset_and_save": [29900, 30000],
        "30001_30100_post_reset": [30001, 30100],
        "30101_30200_post_reset": [30101, 30200],
        "30201_30300_post_reset": [30201, 30300],
        "35750_35900_before_pause": [35750, 35900],
    }
    summary = {name: {tag: summarize(points, *limits) for tag, points in rows.items()}
               for name, limits in windows.items()}
    resets = {}
    for step in (12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000):
        resets[str(step)] = {
            tag: {"pre_100": summarize(points, step - 99, step),
                  "post_1_10": summarize(points, step + 1, step + 10),
                  "post_1_100": summarize(points, step + 1, step + 100),
                  "post_101_300": summarize(points, step + 101, step + 300),
                  "exact": [p[:2] for p in points if step - 3 <= p[0] <= step + 10]}
            for tag, points in rows.items()
        }
    weights = {"train/loss_paper_global_rgb": 1.0, "train/loss_paper_dynamic_rgb": 1.0,
               "train/loss_paper_sky": .05, "train/loss_paper_nps": .01,
               "train/loss_paper_dynamic_opacity": .05, "train/loss_paper_depth": .1}
    weighted = {}
    for name, values in summary.items():
        components = {tag: values[tag]["mean"] * weight for tag, weight in weights.items()
                      if tag in values and values[tag].get("n")}
        total = sum(components.values())
        weighted[name] = {"components": components, "sum_of_mean_components": total,
                          "fractions": {tag: value / total if total else 0.
                                        for tag, value in components.items()}}
    payload = {
        "scope": "Read-only CPU EventAccumulator on paused child event; no GPU/model/checkpoint",
        "event_path": str(event), "event_sha256": hashlib.sha256(event.read_bytes()).hexdigest(),
        "all_scalar_tags": all_tags, "selected_tags": selected_tags,
        "forward_timing": "train scalars at reset step describe its forward BEFORE reset; next step describes post-reset model. Paper eval at save step is AFTER reset/optimizer.",
        "windows": windows, "window_summaries": summary, "reset_summaries": resets,
        "weights_from_frozen_effective_config": weights, "weighted_components": weighted,
        "scalar_points_step_value_walltime": rows,
        "eval_scalars": {tag: [[int(x.step), float(x.value), float(x.wall_time)]
                               for x in accumulator.Scalars(tag)] for tag in all_tags
                         if "loss_viewpoint" in tag},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"selected_tags": selected_tags,
                      "step_span": {tag: [points[0][0], points[-1][0], len(points)]
                                    for tag, points in rows.items() if points}, "output": str(out)}))
    for name, values in summary.items():
        compact = {tag.removeprefix("train/"): round(v["mean"], 6)
                   for tag, v in values.items() if v.get("n")}
        print(name, json.dumps(compact))
    for tag, values in payload["eval_scalars"].items():
        print(tag, json.dumps([p[:2] for p in values]))


if __name__ == "__main__":
    main()
