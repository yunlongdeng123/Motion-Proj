"""只读检查6个冻结W与3种view的完整足迹；不读取生成RGB或隐藏真值。"""
import argparse
import json
from pathlib import Path

import numpy as np

from teacher_supervision_cpu import footprint_mask
from teacher_supervision_view_cpu import fixed_view_spec, mapped_resample, validate_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("保留已有证据，请选择新out")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        with np.load(args.native_assets / case["id"] / "B_roles_native.npz", allow_pickle=False) as z:
            w, k = z["W_output"].copy(), z["K"].copy()
        assert w.shape == k.shape == (1280, 1920) and w.dtype == k.dtype == bool
        assert not (w & k).any()
        yy, xx = np.where(w)
        for name, view in case["views"].items():
            assert view["available"]
            spec = fixed_view_spec(name)
            assert view["model_wh"] == spec["model_wh"] == view["plan"]["output_wh"]
            assert view["plan"]["window_xyxy"] == spec["native_window_xyxy"]
            window, _ = validate_view(spec["model_wh"], (1920, 1280), spec["native_window_xyxy"], spec["native_from_model"], view_name=name)
            x0, y0, x1, y1 = window
            native_valid = np.zeros_like(w)
            native_valid[y0:y1, x0:x1] = True
            outside = int((w & ~native_valid).sum())
            assert outside == 0
            source_valid = np.ones((spec["model_wh"][1], spec["model_wh"][0]), bool)
            outputs = []
            targets = [(480, 320), (768, 512)] + ([(1152, 768)] if name == "full1152" else [])
            for wh in targets:
                rgb_valid = mapped_resample(source_valid, wh, window, (0, 0, 1920, 1280), mask_mode="all")
                w_full = footprint_mask(w, wh, "all")
                unsupported = int((w_full & ~rgb_valid).sum())
                assert unsupported == 0
                outputs.append(dict(output_wh=list(wh), full_native_W_pixels=int(w_full.sum()),
                                    W_pixels_without_full_view_support=unsupported,
                                    teacher_RGB_full_valid_pixels=int(rgb_valid.sum())))
            rows.append(dict(case_id=case["id"], view=name, model_wh=spec["model_wh"], window=list(window),
                             native_from_model=spec["native_from_model"], W_native_pixels=int(w.sum()),
                             W_bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1],
                             native_W_outside_window=outside, outputs=outputs))
    assert len(manifest["cases"]) == 6 and len(rows) == 18
    result = dict(schema="road_2d_view_frozen_W_cpu_qualification.v1", complete=True, passed=True,
                  cases=6, view_checks=18, output_checks=sum(len(row["outputs"]) for row in rows),
                  teacher_RGB_opened=False, hidden_truth_opened=False, GPU_executed=False,
                  actual_raster_camera_convention_verified=False,
                  rows=rows, note="W仅为冻结2D来源域；该几何覆盖检查不产生3D行贡献或高频材质许可。")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "cases", "view_checks", "output_checks", "GPU_executed")}))


if __name__ == "__main__":
    main()
