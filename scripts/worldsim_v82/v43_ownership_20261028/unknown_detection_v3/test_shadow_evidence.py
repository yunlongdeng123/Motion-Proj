"""真实输出的几何/来源回放验收，不把候选数或模型审阅当作精度。"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_shadow_evidence import (RawClip, collect_references,
                                   footprint_distance, project, sample,
                                   visibility)


def verify(output, root):
    contract = json.loads((output / "contract.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    clip = RawClip(root)
    sources = [clip.get(raw) for raw in contract["source_raws"]]
    rows = []
    for report in summary["queries"]:
        raw = report["raw"]
        frame = clip.get(raw)
        with np.load(output / ("%03d.npz" % raw)) as saved:
            a = {k: saved[k] for k in saved.files}
        assert np.array_equal(a["body"], frame["body"]), raw
        assert not a["road_counterevidence_certified"].any(), raw
        assert bool(a["source_shadow_state_unknown"]), raw
        candidate = a["shadow_candidate"]
        strict = a.get("shadow_candidate_strict", candidate)
        propagated = a.get("shadow_propagated", np.zeros_like(candidate))
        valid = a["shadow_evidence_valid"]
        assert not (candidate & ~valid).any(), raw
        assert not (candidate & a["body"]).any(), raw
        assert not (a["road_counterevidence_valid"] & ~valid).any(), raw
        assert not (strict & (a["motion_witness_support_count"] < 2)).any(), raw
        assert not (candidate & (a["source_count"] < 3)).any(), raw
        assert not (candidate & (a["source_span_raw"] < 5)).any(), raw
        assert not (a["source_raw"] == raw).any(), raw
        assert int(candidate.sum()) == report["shadow_candidate_pixels"], raw
        if "shadow_propagated" in a:
            with np.load(Path(report["strict_input"])) as old:
                assert np.array_equal(strict, old["shadow_candidate"]), raw
                assert np.array_equal(a["shadow_score_strict"], old["shadow_score"]), raw
            assert not (propagated & ~a["propagation_eligible"]).any(), raw
            assert not (propagated & a["road_counterevidence_valid"]).any(), raw
            assert not (propagated & a["body"]).any(), raw
            assert not (propagated & a["query_actual_foreground_depth_block"]).any(), raw
            assert np.all(a["propagation_seed_geodesic_m"][propagated] <= 1.50001), raw
            assert np.all(a["shadow_propagated_score"][propagated] <= 0.49), raw
            assert np.array_equal(candidate, strict | propagated), raw
            assert set(np.unique(a["proposal_shadow_evidence_state"])) <= {-1, 0, 1}, raw
            assert not (a["photo_supported_unknown"] & (a["strict_shadow_evidence_state"] == 0)).any(), raw
        cy, cx = np.where(candidate)
        by, bx = np.where(a["road_counterevidence_valid"])
        # 回放所有候选和至多20个弱反证，核对真实源图、深度与排除条件。
        if len(bx) > 20:
            take = np.linspace(0, len(bx) - 1, 20).astype(int)
            by, bx = by[take], bx[take]
        yy, xx = np.r_[cy, by], np.r_[cx, bx]
        points = a["world_point"][yy, xx]
        uv, _ = project(points, frame)
        assert np.allclose(uv, np.column_stack([xx, yy]), atol=1e-5), raw
        visible, _, detail = visibility(points, frame)
        is_strict = strict[yy, xx]
        assert visible[is_strict].all(), raw
        for gate in ["inside", "road", "actor_clear", "clear_mask"]:
            assert detail[gate].all(), (raw, gate)
        gains = {int(k): np.array(v["gain_rgb_source_to_query"])
                 for k, v in report["exposure_sources"].items() if v["valid"]}
        reference = collect_references(points, sources, gains, excluded_raw=raw)
        assert reference["valid"].all(), raw
        assert np.allclose(reference["rgb"], a["reference_rgb_exposure_matched"][yy, xx], atol=1e-6), raw
        assert np.array_equal(reference["count"], a["source_count"][yy, xx]), raw
        audits = []
        # JSON展示至多12个点，所有候选均已通过上面的来源回放。
        take = np.linspace(0, len(xx) - 1, min(12, len(xx))).astype(int) if len(xx) else []
        for j in take:
            y, x = int(yy[j]), int(xx[j])
            point = points[j:j + 1]
            observations = []
            for sr in a["source_raw"][y, x]:
                if sr < 0:
                    continue
                source = clip.get(int(sr))
                sv, suv, _ = visibility(point, source, baseline=True)
                assert bool(sv[0]), (raw, y, x, int(sr))
                original_rgb = sample(source["rgb"], suv)[0]
                observations.append({"raw": int(sr), "pixel_xy": suv[0].tolist(),
                                     "original_rgb_0_1": original_rgb.tolist(),
                                     "exposure_gain_to_query": gains[int(sr)].tolist(),
                                     "target_footprint_distance_m": float(footprint_distance(point[:, :2], source["target"])[0]),
                                     "shadow_state": "unknown"})
            audits.append({"query_pixel_xy": [x, y], "is_candidate": bool(candidate[y, x]),
                           "query_original_rgb_0_1": frame["rgb"][y, x].tolist(),
                           "world_point": point[0].tolist(), "sources": observations})
        rows.append({"raw": raw, "checked_candidate_pixels": len(cx),
                     "checked_weak_counterevidence_pixels": len(bx),
                     "all_invariants_passed": True, "source_observation_examples": audits})
        print("RAW_EVIDENCE_AUDIT", raw, "candidates", len(cx), "weak_bg", len(bx), flush=True)
    result = {"schema": "raw-shadow-evidence-audit-v1", "queries": rows,
              "all_invariants_passed": True, "human_GT": None,
              "meaning": "output/source replay and missingness audit; no accuracy/recall/human verdict claim"}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028"))
    args = parser.parse_args()
    verify(args.output, args.root)
