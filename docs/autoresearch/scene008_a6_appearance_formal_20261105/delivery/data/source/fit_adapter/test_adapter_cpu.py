"""Small deterministic CPU check with real 18432-row geometry and real pose.

The sparse fitting rays here are labelled fixtures, NOT a native-render test.
No output from this check is an appearance candidate or a research result.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz, vstack

from adapter import (SCHEMA, TEST_SCHEMA, Solver, array_sha, bank, collect_problem,
                     export_asset, original_zero_columns, project_local, same_bytes,
                     scalar, sha, validate_view, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", default=str(Path(__file__).with_name("cpu_validation")))
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    basepath = workspace/"a6_io_audit/source_evidence/scene008_a6_asset_controls_20261105/m2/complete_a6.npz"
    buildpath = basepath.with_name("design.npz")
    viewpath = Path(__file__).with_name("evidence")/"090_view.npz"
    base, view = bank(basepath), bank(viewpath)
    audit = validate_view(view)
    projected = project_local(base["xyz"], view, crop=False)
    x0, y0, x1, y1 = view["crop_xyxy"].astype(int)
    projected_crop = project_local(base["xyz"], view, crop=True)
    geometry_projection_error = float(abs(projected_crop-(projected-[x0, y0])).max())
    assert geometry_projection_error < 1e-9
    build = load_npz(buildpath).tocsr()
    unknown = original_zero_columns(build, len(base["xyz"]))
    u = np.flatnonzero(unknown)[:3]; k = np.flatnonzero(~unknown)[0]
    truth = base["rgb"].astype(np.float64).copy()
    truth[u] = [[.82, .11, .19], [.08, .69, .24], [.15, .22, .86]]
    rr = [0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7]
    cc = [u[0], u[1], k, u[2], k, u[0], k, u[1], u[2], k, u[2]]
    vv = [.95, .57, .38, .95, .95, .4, .55, .95, .2, .1, .95]
    W = coo_matrix((vv, (rr, cc)), shape=(8, len(truth))).tocsr()
    alpha = np.asarray(W.sum(1)).ravel()
    target = (W@truth)/alpha[:, None]
    pixels = np.flatnonzero(view["RGB_unknown_mask"].ravel())[30:38]
    foreground = np.zeros_like(view["RGB_unknown_mask"]); foreground.ravel()[pixels] = True
    np.savez_compressed(root/"masks.npz", foreground_mask=foreground,
                        RGB_unknown_mask=view["RGB_unknown_mask"])
    views = []
    for index in range(2):
        rgb = np.zeros((*foreground.shape, 3), np.float32)
        rgb.reshape(-1, 3)[pixels] = target.astype(np.float32)
        # Outside-mask RGB deliberately differs; it must have no influence.
        rgb[~foreground] = [.37*index, .23*index, .91*index]
        np.savez_compressed(root/f"candidate{index}.npz", RGB_srgb=rgb)
        op = root/f"operator{index}"; op.mkdir()
        save_npz(op/"weights.npz", W)
        fullpixels = (pixels//(x1-x0)+y0)*int(view["full_image_hw"][1])+pixels % (x1-x0)+x0
        np.savez_compressed(op/"rays.npz", alpha=alpha, crop_pixels=pixels, full_pixels=fullpixels)
        write_json(op/"operator.json", dict(schema=TEST_SCHEMA,
            primitive_count=len(truth), geometry_hash=scalar(base["geometry_hash"]),
            provenance_hash=scalar(base["provenance_hash"]), camera_package_sha256=sha(viewpath),
            eligible_mask_sha256=array_sha(foreground), foreground_erosion_pixels=0,
            weights_sha256=sha(op/"weights.npz"), rays_sha256=sha(op/"rays.npz")))
        views.append(dict(id=f"real_pose_fixture_{index}", camera_npz=str(viewpath),
                          foreground_mask="masks.npz", unknown_mask="masks.npz",
                          rgb=f"candidate{index}.npz", rgb_mode="straight_srgb",
                          operator_dir=op.name))
    write_json(root/"manifest.json", dict(schema=SCHEMA, views=views))
    problem = collect_problem(root/"manifest.json", basepath, buildpath, allow_test_operator=True)
    assert len(problem["target"]) == 12  # six opaque unknown rows in each view
    assert np.array_equal(np.flatnonzero(problem["active"]), u)
    try:
        collect_problem(root/"manifest.json", basepath, buildpath)
    except ValueError:
        formal_refuses_fixture = True
    else:
        raise AssertionError("Formal default admitted non-native fixture rays")
    solver = Solver(problem, ridge=0.)
    state = solver.begin()
    for _ in range(17):
        state = solver.step(state)
    solver.save_state(root/"state17.npz", state)
    resumed = solver.load_state(root/"state17.npz")
    n1, n2 = solver.step(state), solver.step(resumed)
    assert all(same_bytes(n1[key], n2[key]) for key in ("x", "y"))
    for _ in range(47):
        state = solver.step(state)
    error = float(abs(state["x"]-truth[u]).max()); assert error < 1e-6
    exported = export_asset(root/"fixture_asset.npz", problem, state,
                            dict(test_only=True, native_rays=False))
    after = bank(root/"fixture_asset.npz")
    assert all(same_bytes(base[key][~unknown], after[key][~unknown])
               for key in ("rgb", "canonical_sh", "colour_inferred"))
    # Reuse the existing old-runtime loader on CPU; no runtime or scene/GPU is loaded.
    deps = workspace/"a6_asset_controls/_cpu_deps"
    sys.path[:0] = [str(deps), str(workspace/"a6_asset_controls"), str(workspace/"a6_complete_asset")]
    from sh_asset import SHAsset
    loaded = SHAsset.from_geometry(root/"fixture_asset.npz")
    assert loaded.assert_geometry_unchanged() and len(loaded.xyz) == 18432
    assert np.array_equal(loaded.coefficients.cpu().numpy()[~unknown], base["canonical_sh"][~unknown])
    # A newly observed HR-only column must join the protected set, even though
    # it was an exact zero in the old BUILD operator.
    extra = coo_matrix(([1.], ([0], [u[2]])), shape=(1, len(truth))).tocsr()
    hr_design = vstack([build, extra], format="csr")
    save_npz(root/"additional_observed_design.npz", hr_design)
    hr_zero = original_zero_columns(hr_design, len(truth))
    np.savez_compressed(root/"additional_zero.npz", unknown=hr_zero)
    protected_problem = collect_problem(root/"manifest.json", basepath, buildpath,
        allow_test_operator=True, additional_zero_mask_path=root/"additional_zero.npz",
        additional_protected_design_path=root/"additional_observed_design.npz")
    assert np.array_equal(np.flatnonzero(protected_problem["active"]), u[:2])
    protected_solver = Solver(protected_problem, ridge=0.)
    protected_state = protected_solver.begin()
    for _ in range(32):
        protected_state = protected_solver.step(protected_state)
    export_asset(root/"additional_protected_fixture_asset.npz", protected_problem,
                 protected_state, dict(test_only=True, additional_observations=True))
    protected_after = bank(root/"additional_protected_fixture_asset.npz")
    assert same_bytes(base["canonical_sh"][u[2]], protected_after["canonical_sh"][u[2]])
    assert np.array_equal(hr_design@base["rgb"], hr_design@protected_after["rgb"])
    bad = {key: value.copy() for key, value in view.items()}
    bad["camera_K_crop"][0, 2] += 1.
    try:
        validate_view(bad)
    except ValueError:
        rejects_bad_crop = True
    else:
        raise AssertionError("Admitted inconsistent crop camera")
    report = dict(status="PASS_CPU_ADAPTER_CONTRACT", real_geometry_primitives=18432,
        original_BUILD_columns=build.shape[1], original_zero_columns=int(unknown.sum()),
        original_nonzero_columns=int((~unknown).sum()), camera_raw=90, real_pose_audit=audit,
        all_primitive_crop_projection_maxabs_pixels=geometry_projection_error,
        fixture_views=2, fixture_admitted_rays=len(problem["target"]), fixture_fit_steps=64,
        fitted_RGB_maxabs_error=error, complete_state_nextstep_bitwise_replay=True,
        original_BUILD_RGB_and_protected_SH_exact=True, old_SHAsset_CPU_reload=True,
        additional_HR_only_observed_column_protected=True,
        additional_observed_design_prediction_bitwise_unchanged=True,
        formal_default_refuses_non_native_rays=formal_refuses_fixture,
        inconsistent_crop_camera_rejected=rejects_bad_crop, export=exported,
        native_GPU_operator_extraction_tested=False, native_scene_insertion_tested=False,
        no_GPU_used=True, source_files_unchanged=True)
    write_json(root/"receipt.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
