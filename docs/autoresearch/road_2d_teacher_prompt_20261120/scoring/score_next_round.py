"""60组冻结提示词对照的独立CPU评分；不选seed、不读取生成中的GT。

必须先调用原推理入口 infer_next_round.freeze_gate 检查全部60个原始
预测及12个历史精确重放，才允许打开独立truth配置与真值图像。
颜色/HP/PSD实现逐字节复用上一轮冻结源码，主PSD只比较共同前三带。
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import metrics_pilot_frozen as frozen

IDS = frozen.IDS
SEEDS = [101, 202, 303, 404, 505]
PROMPTS = ["legacy", "material_match_v1"]
METRIC_SOURCE_SHA256 = "898c990128423df71705696d8cbe35066623518153f637232a2fce30a2c61ca1"
SCHEMA = "road_2d_teacher_prompt_round_postfreeze_CPU_score.v1"
DEFAULT_ENTRY = HERE.parent / "code/infer_next_round.py"
require, read, write, sha, pin = frozen.require, frozen.read, frozen.write, frozen.sha, frozen.pin
rgb, binary, roles = frozen.rgb, frozen.binary, frozen.roles


def load_entry(path):
    path = Path(path).resolve()
    require(path.is_file(), "frozen inference entry not found: " + str(path))
    name = "_road_teacher_prompt_round_inference"
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def complete_gate(predictions_path, manifest_path, entry_path=DEFAULT_ENTRY, allow_cpu_mock=False):
    """只检查预测/输入，不接收truth路径。原入口负责真实全60像素检查。"""
    require(sha(HERE / "metrics_pilot_frozen.py") == METRIC_SOURCE_SHA256, "frozen metric implementation changed")
    initial = read(predictions_path)
    mock = initial.get("CPU_mock") is True
    require(not mock or allow_cpu_mock, "CPU mock requires explicit --allow-cpu-mock; never GPU evidence")
    before = {"predictions": pin(predictions_path), "manifest": pin(manifest_path)}
    entry = load_entry(entry_path)
    gate = entry.freeze_gate(predictions_path, manifest_path)
    require(gate.get("passed") is True and gate.get("jobs") == 60 and gate.get("historical_replays") == 12,
            "canonical 60/12 freeze gate did not pass")
    require(before["predictions"] == pin(predictions_path) and before["manifest"] == pin(manifest_path),
            "frozen inputs changed while checking gate")
    predictions, plan = read(predictions_path), read(manifest_path)
    config = plan["configuration"]
    require(config["seeds"] == SEEDS and [p["id"] for p in config["prompts"]] == PROMPTS,
            "five-seed/two-prompt contract changed")
    expected = {(sid, prompt, seed) for sid in IDS for prompt in PROMPTS for seed in SEEDS}
    actual = {(j["case_id"], j["prompt_id"], j["seed"]) for j in predictions["jobs"]}
    require(len(predictions["jobs"]) == len(actual) == 60 and actual == expected,
            "missing/duplicate/extra case-prompt-seed combinations")
    require(all(j["arm"] == "full1152_T0" and j["view"] == "full1152" for j in predictions["jobs"]),
            "only adopted full1152_T0 is allowed")
    return predictions, plan, dict(gate, completed_jobs=60, CPU_mock=mock,
        canonical_inference_entry=pin(entry_path), GT_opened_before_gate=False,
        before_check_source_pins=before)


def inherited_valid_window(plan, sid, native_hw):
    """延续旧共同诊断窗口，使12个历史重放的指标分母可核对。"""
    source = read(plan["source_pins"]["baseline_manifest"]["path"])
    case = next(c for c in source["cases"] if c["id"] == sid)
    valid = np.ones(tuple(native_hw), bool)
    windows = []
    for name in ["full768", "crop768", "full1152"]:
        view = case["views"][name]
        require(view.get("available") is True, "frozen diagnostic view unavailable")
        _, _, x0, y0, x1, y1, *_ = frozen.view_geometry(view["plan"], native_hw)
        inside = np.zeros(tuple(native_hw), bool)
        inside[y0:y1, x0:x1] = True
        valid &= inside
        windows.append(dict(view=name, window_xyxy=[x0, y0, x1, y1]))
    return valid, windows


def add_common_psd(grid):
    ratios = grid["PSD"]["band_power_ratio"][:3]
    good = len(ratios) == 3 and all(v is not None for v in ratios)
    grid["PSD"]["common3_mean_abs_log2_band_power_ratio"] = (
        float(np.mean([abs(math.log2(max(value, 1e-12))) for value in ratios])) if good else None)
    grid["PSD"]["common3_status"] = "quantified" if good else "insufficient_fixed_support"
    grid["PSD"]["common3_band_edges_cycles_per_native_px"] = [list(b) for b in frozen.BANDS[:3]]
    return grid


def score(predictions_path, manifest_path, truth_config_path, out, *, entry_path=DEFAULT_ENTRY, allow_cpu_mock=False):
    out = Path(out)
    require(not out.exists(), "preserve existing score; choose a new --out")
    # No truth config/image is opened above or inside this gate.
    predictions, plan, gate = complete_gate(predictions_path, manifest_path, entry_path, allow_cpu_mock)
    require(Path(truth_config_path).resolve() not in [Path(predictions_path).resolve(), Path(manifest_path).resolve()],
            "truth configuration must be independent")
    truth = read(truth_config_path)
    require(truth.get("schema") == "road_2d_teacher_scoring_truth.v1", "independent truth schema mismatch")
    require(set(truth["cases"]) == set(IDS), "truth must explicitly name all six cases")
    result = dict(schema=SCHEMA, complete=False, CPU_only=True, threads=1,
        mode="CPU_MOCK_VALIDATION_ONLY" if gate["CPU_mock"] else "FROZEN_PREDICTION_CPU_SCORING",
        GT_opened_after_gate=True, gate=gate, truth_config=pin(truth_config_path),
        metric_implementation=pin(HERE / "metrics_pilot_frozen.py"),
        prompts=copy.deepcopy(plan["configuration"]["prompts"]), seeds=SEEDS, arm="full1152_T0",
        common_PSD_band_edges_cycles_per_native_px=[list(v) for v in frozen.BANDS[:3]],
        cases=[], jobs=[], automatic_winner_selected=False, best_of_N_selection_implemented=False,
        notes=[
            "All60 actual raw/final PNGs, inverse transforms, W/K pixels and12 historical replays passed infer_next_round.freeze_gate before truth was read.",
            "Metric code is byte-identical to the frozen48-pilot implementation. HP sigma2.5/5/10 is in native pixels; native and same_view are separate.",
            "The previous three-view common valid window is inherited only as a fixed diagnostic denominator; new full1152 generation remains full-FOV. No prediction RGB selects masks or tiles.",
            "Primary PSD uses the common first3 bands0.025–0.0625,0.0625–0.125,0.125–0.2 cycles/native pixel. All-available-band scalars are descriptive only; more energy is not better material.",
            "Four synthetic tasks reuse only two clean source frames. Five seeds characterize noise variability, not five independent scenes. No confidence interval or significance claim is made.",
            "Real cases score only legally known visible-road preservation of raw candidates, not hidden-road accuracy. W-exterior final copying is a separate scope guarantee.",
            "No prompt ranking, metric reweighting, best-of-N choice, exposure fitting, new prompt, training or3D renderer is implemented."])
    loaded = {}
    cases = {c["id"]: c for c in plan["cases"]}
    maskdir = out.parent / (out.stem + "_masks")
    require(not maskdir.exists(), "preserve existing mask evidence; choose a new --out")
    maskdir.mkdir(parents=True)
    for sid in IDS:
        case = cases[sid]
        observation, native_roles = rgb(case["observation"]["path"]), roles(case["roles_native"]["path"])
        valid, windows = inherited_valid_window(plan, sid, case["native_hw"])
        source = truth["cases"][sid]
        require(source["domain"] == case["domain"], "truth/generation case domain mismatch")
        if source["domain"] == "synthetic_visible_2d":
            for key in ["clean_rgb", "visible_road_certificate"]:
                require(sha(source[key]["path"]) == source[key]["sha256"], "independent truth digest mismatch")
            reference = rgb(source["clean_rgb"]["path"])
            certificate = binary(source["visible_road_certificate"]["path"])
            require(reference.shape == observation.shape and certificate.shape == observation.shape[:2], "truth grid mismatch")
            base = native_roles["target_core"] & native_roles["W_output"] & ~native_roles["K"] & certificate
            distance = distance_transform_edt(native_roles["W_output"])
            kind = "synthetic_certified_hidden_target"
        else:
            require(source["domain"] == "real_unlabelled" and "clean_rgb" not in source, "real hidden GT must not be invented")
            reference = observation
            base = native_roles["C_mat"] & ~native_roles["H_input"] & ~native_roles["K"]
            for key in ["shadow_possible", "shadow_uncertain", "shadow_boundary_guard", "foreground_uncertainty"]:
                if key in native_roles:
                    base &= ~native_roles[key]
            distance = distance_transform_edt(base)
            kind = "known_visible_road_preservation_only"
        mask = base & (distance > 12) & frozen.reference_structure_ok(reference) & (distance_transform_edt(valid) > 40)
        require(int(mask.sum()) >= 64, "insufficient fixed scoring support: " + sid)
        tiles = frozen.fixed_tiles(mask)
        path = maskdir / (sid + ".png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        result["cases"].append(dict(id=sid, domain=source["domain"], metric_role=kind,
            hidden_road_GT_available=source["domain"] == "synthetic_visible_2d",
            fixed_native_mask=pin(path), fixed_native_mask_pixels=int(mask.sum()),
            fixed_native_tiles=[list(tile) for tile in tiles], inherited_diagnostic_views=windows))
        loaded[sid] = (reference, mask, distance, tiles, kind)
    cache = {}
    for job in predictions["jobs"]:
        sid = job["case_id"]
        reference, mask, distance, tiles, kind = loaded[sid]
        view = cases[sid]["view"]
        view_plan = view["plan"]
        spacing = frozen.view_geometry(view_plan, cases[sid]["native_hw"])[6]
        offset = tuple(view_plan["window_xyxy"][:2])
        key = (sid, job["raw_model"]["sha256"], job["raw_native"]["sha256"])
        if key not in cache:
            model, native = rgb(job["raw_model"]["path"]), rgb(job["raw_native"]["path"])
            reference_view = frozen.sample_view(reference, view_plan)
            cache[key] = dict(
                native=add_common_psd(frozen.metrics(native, reference, mask, distance, tiles)),
                same_view=add_common_psd(frozen.metrics(model, reference_view, mask, distance, tiles,
                                                      spacing, offset, view_plan)))
        result["jobs"].append(dict(id=job["id"], case_id=sid, prompt_id=job["prompt_id"],
            arm=job["arm"], seed=job["seed"], view=job["view"], metric_role=kind,
            historical_replay_expected=job["historical_replay_expected"], **cache[key]))
    require(len(result["jobs"]) == 60, "scoring output count mismatch")
    require(pin(predictions_path) == gate["predictions"] and pin(manifest_path) == gate["manifest"],
            "frozen predictions/plan changed during scoring")
    result.update(complete=True, scored_jobs=60, unique_pixel_metric_computations=len(cache))
    write(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["predictions", "manifest", "truth-config", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inference-entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--allow-cpu-mock", action="store_true", help="仅CPU fixture；强制标记非GPU证据")
    args = parser.parse_args()
    result = score(args.predictions, args.manifest, args.truth_config, args.out,
                   entry_path=args.inference_entry, allow_cpu_mock=args.allow_cpu_mock)
    print(__import__("json").dumps(dict(complete=result["complete"], scored_jobs=60,
          mode=result["mode"], GT_opened_after_gate=result["GT_opened_after_gate"], out=str(args.out.resolve()))))


if __name__ == "__main__":
    main()
