"""固定点原始覆盖对照：真实RGB回放、集合和无模型外推不变量。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "2"

import argparse
import json
from pathlib import Path

import numpy as np

from build_shadow_evidence import DEFAULT_ROOT, RawClip, sample


def verify(output, root):
    contract = json.loads((output / "contract.json").read_text())
    assert not contract["source_sh_matching_performed"]
    assert not contract["native_motion_extrapolated"]
    assert len(contract["comparisons"]["local49_all"]) == 49
    assert len(contract["comparisons"]["local49_stride4"]) == 12
    assert contract["comparisons"]["full199_stride4"] == list(range(0, 199, 4))
    assert len(contract["comparisons"]["union87"]) == 87
    clip = RawClip(root)
    export = Path(contract["row_export"])
    rows = []
    for raw in contract["query_raws"]:
        report = json.loads((output / ("%03d.json" % raw)).read_text())
        with np.load(output / ("%03d.npz" % raw)) as f:
            a = {k: f[k] for k in f.files}
        with np.load(export / ("%03d.npz" % raw)) as q:
            assert np.array_equal(a["query_world_fixed"], q["world"])
            assert np.array_equal(a["query_sigma_m"], q["sigma_m"])
        assert not bool(a["source_sh_matching_performed"])
        assert not bool(a["native_motion_extrapolated"])
        assert not a["background_certified"].any()
        assert "background_protection_candidate" not in a
        assert not a["source_geometry_valid"][:, a["source_raws"] == raw].any()
        assert np.all(a["full199_stride4_source_count"] >= a["local49_stride4_source_count"])
        assert np.all(a["union87_source_count"] >= a["local49_all_source_count"])
        replayed = 0
        for si, sr in enumerate(a["source_raws"]):
            selected = np.flatnonzero(a["source_geometry_valid"][:, si])
            if not len(selected):
                continue
            gain = np.array(report["exposure_sources"][str(int(sr))]["gain_rgb_source_to_query"])
            original = sample(clip.get(int(sr))["rgb"], a["source_actual_surface_uv"][selected, si]) * gain
            assert np.allclose(original, a["source_original_rgb_exposure_matched"][selected, si], atol=2e-6)
            replayed += len(selected)
        for stage in contract["comparisons"]:
            positive = a[stage + "_raw_surface_witness"]
            assert np.all(a[stage + "_source_count"][positive] >= 2)
            assert np.all(a[stage + "_time_span_s"][positive] >= 0.499)
            assert np.all(a[stage + "_camera_baseline_m"][positive] >= 1)
            assert np.all(a[stage + "_view_angle_deg"][positive] >= 1)
            assert np.all(a[stage + "_rgb_median_deviation"][positive] <= 0.035001)
            assert np.all(a[stage + "_rgb_p90_deviation"][positive] <= 0.060001)
            assert int(positive.sum()) == report["stages"][stage]["stable_raw_surface_witness_rows"]
        rows.append({"raw": raw, "original_rgb_observations_replayed": replayed,
                     "query_world_exactly_preserved": True, "source_sh_matching_performed": False,
                     "native_motion_extrapolated": False, "all_invariants_passed": True})
    result = {"schema": "fixed-world-raw-coverage-audit-v1", "queries": rows,
              "all_invariants_passed": True, "meaning": "raw observation coverage only; no certified background, no car/U ground truth"}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = p.parse_args()
    verify(a.output, a.root)
