from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARAMETERS = ("xyz", "scales", "rotation", "coefficients", "opacity")
RADIUS_M = 0.075
PATHS = {
    "old": "work/vehicle_road_joint_20261121/takeback/vehicle_road_joint_20261121_r1/assets/joint_fit/vehicle.npz",
    "trellis": "outputs/a20_sd15_sameview_20261122/a20_actor_candidate.npz",
    "fixed_OBB_diagnostic": "work/a20_native_gaussian_fusion_20261122/regions/O_source_full3sigma_OBB_diagnostic_v1.npz",
    "old_identity_map": "work/a20_native_gaussian_fusion_20261122/observed_source/observed_actor_identity_map.npz",
    "old_opaque_qualification": "work/a20_native_gaussian_fusion_20261122/source_qualification/O4131_opaque_qualification_v1.npz",
    "trellis_appearance_support": "work/a20_sd15_sameview_20261122/student_native_support_control_r1/appearance_support.npz",
    "native_scene_bindings": "work/a20_native_gaussian_fusion_20261122/observed_source/source_manifest.json",
}
EXPECTED_PARENT_SHA256 = {
    "old": "5ffe95bcf5c100d365a6aea7e0f5036cb629c64dfd52dcadf1d6a68e5a2b0bec",
    "trellis": "0dc65c4aa9fefbf73a1c2d482c0015915818c5bf0ac34d5d318e0784693da53a",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_record(key: str) -> dict:
    path = ROOT / PATHS[key]
    result = {"path": PATHS[key], "sha256": file_sha256(path), "bytes": path.stat().st_size}
    if key in EXPECTED_PARENT_SHA256:
        assert result["sha256"] == EXPECTED_PARENT_SHA256[key], (key, result)
    return result


def exact_bytes(actual: np.ndarray, expected: np.ndarray) -> bool:
    return actual.dtype == expected.dtype and actual.shape == expected.shape and np.array_equal(
        np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8)
    )


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    inputs = {key: source_record(key) for key in PATHS}
    old = np.load(ROOT / PATHS["old"], allow_pickle=False)
    trellis = np.load(ROOT / PATHS["trellis"], allow_pickle=False)
    footprint = np.load(ROOT / PATHS["fixed_OBB_diagnostic"], allow_pickle=False)
    identity = np.load(ROOT / PATHS["old_identity_map"], allow_pickle=False)
    qualification = np.load(ROOT / PATHS["old_opaque_qualification"], allow_pickle=False)
    support = np.load(ROOT / PATHS["trellis_appearance_support"], allow_pickle=False)
    bindings = json.loads((ROOT / PATHS["native_scene_bindings"]).read_text(encoding="utf-8"))

    old_n, trellis_n = len(old["xyz"]), len(trellis["xyz"])
    assert (old_n, trellis_n) == (25060, 410080)
    actor_rows = np.arange(old_n, dtype=np.int64)
    assert np.array_equal(footprint["actor_row_ids"], actor_rows)
    assert np.array_equal(identity["actor_row_ids"], actor_rows)
    assert np.array_equal(footprint["source_branch"], old["source_branch"])
    assert np.array_equal(footprint["source_row"], old["source_row"])
    assert np.array_equal(identity["saved_source_branch"], old["source_branch"])
    assert np.array_equal(identity["saved_source_raw"], old["source_raw"])
    assert str(footprint["source_O_sha256"].item()) == EXPECTED_PARENT_SHA256["old"]

    native = old["source_branch"] == 2
    legacy = old["source_branch"] == 3
    intersects = footprint["full_3sigma_ellipsoid_intersects_fixed_OBB"]
    mixed = identity["old_K_mixed_or_non_target"] & native
    assert np.all(native | legacy)
    assert np.array_equal(native, identity["native_origin"])
    assert np.array_equal(legacy, identity["old_engineering_prior"])
    keep = native & intersects & ~mixed
    kept_rows = actor_rows[keep]
    native_global = identity["native_global_row_ids"]
    assert len(np.unique(native_global[keep])) == len(kept_rows)
    assert np.all(native_global[keep] >= 0)
    assert len(kept_rows) == 8925

    high_geometry = np.zeros(old_n, dtype=bool)
    high_geometry[qualification["diagnostic_opaque_shell_association_actor_row_ids"]] = True
    assert np.all(keep[high_geometry]) and high_geometry.sum() == 17

    # Independent reasons are retained even when they overlap. Primary reasons
    # use legacy -> disjoint geometry -> remaining mixed K priority.
    reason_bits = np.zeros(old_n, dtype=np.uint8)
    reason_bits[legacy] |= np.uint8(1)
    reason_bits[native & ~intersects] |= np.uint8(2)
    reason_bits[mixed] |= np.uint8(4)
    assert np.array_equal(reason_bits == 0, keep)
    primary_reason = np.zeros(old_n, dtype=np.uint8)
    primary_reason[legacy] = 1
    primary_reason[native & ~intersects] = 2
    primary_reason[native & intersects & mixed] = 4
    cleanup = {
        "actor_row_ids": actor_rows,
        "native_global_row_ids": native_global,
        "original_source_row_ids": old["source_row"],
        "source_branch": old["source_branch"],
        "source_raw": old["source_raw"],
        "keep_for_joint_actor": keep,
        "exclusion_reason_bits": reason_bits,
        "exclusion_primary_reason_code": primary_reason,
        "kept_actor_row_ids": kept_rows,
        "kept_native_global_row_ids": native_global[keep],
        "excluded_actor_row_ids": actor_rows[~keep],
        "excluded_native_global_row_ids_including_legacy_minus1": native_global[~keep],
        "excluded_legacy_engineering_actor_row_ids": actor_rows[legacy],
        "excluded_exact_3sigma_disjoint_actor_row_ids": actor_rows[native & ~intersects],
        "excluded_K_mixed_or_non_target_actor_row_ids": actor_rows[mixed],
        "excluded_K_mixed_inside_OBB_actor_row_ids": actor_rows[native & intersects & mixed],
        "native_context_excluded_source_rows_must_remain_frozen_global_row_ids": native_global[native & ~keep],
        "native_context_K_other_must_remain_frozen_global_row_ids": native_global[mixed],
        "native_joint_copy_dedup_candidate_global_row_ids": native_global[keep],
    }
    np.savez_compressed(HERE / "old_source_cleanup.npz", **cleanup)

    direct_trellis = support["observed_appearance_support"].astype(bool, copy=False)
    assert direct_trellis.shape == (trellis_n,) and direct_trellis.sum() == 12865
    assert np.array_equal(np.flatnonzero(direct_trellis), support["observed_row_ids"])
    assert np.array_equal(direct_trellis, trellis["observed_protected"])
    assert np.array_equal(~direct_trellis, support["unknown_appearance_support"])
    assert np.array_equal(~direct_trellis, trellis["missing"])
    # The original full T array is untouched; this query only creates metadata.
    observed_tree = cKDTree(trellis["xyz"][direct_trellis].astype(np.float64))
    distance_trellis, nearest_support_index = observed_tree.query(
        trellis["xyz"].astype(np.float64), workers=1
    )
    distance_old, nearest_support_old_index = observed_tree.query(
        old["xyz"][keep].astype(np.float64), workers=1
    )
    support_row_ids = np.flatnonzero(direct_trellis)
    within_trellis = distance_trellis <= RADIUS_M
    u = np.clip(distance_trellis / RADIUS_M, 0.0, 1.0)
    continuous_trellis = (1.0 - (3.0 * u**2 - 2.0 * u**3)).astype(np.float32)
    continuous_trellis[direct_trellis] = np.float32(1.0)
    unknown_trellis = ~within_trellis
    old_observed = identity["visible_optimizer_rows"][keep]
    assert np.array_equal(old_observed, identity["initial_native_contribution"][keep] > 0.0)
    assert within_trellis.sum() == 24410 and unknown_trellis.sum() == 385670

    old_kept_n = len(kept_rows)
    total_n = old_kept_n + trellis_n
    false_old = np.zeros(old_kept_n, dtype=bool)
    false_trellis = np.zeros(trellis_n, dtype=bool)
    old_geometry = np.full(old_kept_n, 0.10, dtype=np.float32)
    old_geometry[high_geometry[keep]] = np.float32(1.0)
    old_appearance = np.full(old_kept_n, 0.60, dtype=np.float32)
    old_appearance[high_geometry[keep]] = np.float32(0.90)
    old_opacity_confidence = np.full(old_kept_n, 0.10, dtype=np.float32)
    old_opacity_confidence[high_geometry[keep]] = np.float32(0.80)
    payload = {key: np.concatenate((old[key][keep], trellis[key]), axis=0) for key in PARAMETERS}
    assert all(payload[key].dtype == np.float32 for key in PARAMETERS)
    payload.update(
        source_id=np.concatenate((np.zeros(old_kept_n, dtype=np.int8), np.ones(trellis_n, dtype=np.int8))),
        source_row_id=np.concatenate((kept_rows, np.arange(trellis_n, dtype=np.int64))),
        native_global_row_id=np.concatenate((native_global[keep], np.full(trellis_n, -1, dtype=np.int64))),
        original_source_row_id=np.concatenate((old["source_row"][keep], np.full(trellis_n, -1, dtype=np.int64))),
        original_source_branch=np.concatenate((old["source_branch"][keep], np.full(trellis_n, -1, dtype=np.int8))),
        source_raw=np.concatenate((old["source_raw"][keep], np.full(trellis_n, -1, dtype=np.int16))),
        geometry_confidence=np.concatenate((old_geometry, np.full(trellis_n, 0.50, dtype=np.float32))),
        appearance_confidence=np.concatenate((old_appearance, np.float32(0.50) + np.float32(0.40) * continuous_trellis)),
        opacity_confidence=np.concatenate((old_opacity_confidence, np.full(trellis_n, 0.50, dtype=np.float32))),
        is_observed=np.concatenate((old_observed, direct_trellis)),
        has_native_source_evidence=np.concatenate((np.ones(old_kept_n, dtype=bool), false_trellis)),
        geometry_is_generated_prior=np.concatenate((false_old, np.ones(trellis_n, dtype=bool))),
        opaque_geometry_high_confidence=np.concatenate((high_geometry[keep], false_trellis)),
        observed_surface=np.concatenate((old_observed, within_trellis)),
        observed_surface_confidence=np.concatenate((old_observed.astype(np.float32), continuous_trellis)),
        distance_to_T_direct_observed_surface_m=np.concatenate((distance_old, distance_trellis)).astype(np.float32),
        nearest_T_direct_observed_source_row_id=np.concatenate((support_row_ids[nearest_support_old_index], support_row_ids[nearest_support_index])).astype(np.int64),
        teacher_unknown=np.concatenate((false_old, unknown_trellis)),
        teacher_prior_weight=np.concatenate((np.zeros(old_kept_n, dtype=np.float32), np.float32(1.0) - continuous_trellis)),
        teacher_reference_row_id=np.concatenate((np.full(old_kept_n, -1, dtype=np.int64), np.arange(trellis_n, dtype=np.int64))),
        high_frequency_permission=np.ones(total_n, dtype=np.float32),
        real_V_supervision_weight=np.ones(total_n, dtype=np.float32),
    )
    joint_path = HERE / "joint_actor_init.npz"
    np.savez_compressed(joint_path, **payload)

    # CPU acceptance reloads the written archive and compares parent values as
    # bytes. This catches an unintended activation, normalization, or alpha edit.
    checks: dict = {}
    per_parameter: dict = {}
    output = np.load(joint_path, allow_pickle=False)
    for key in PARAMETERS:
        value = output[key]
        old_exact = exact_bytes(value[:old_kept_n], old[key][keep])
        trellis_exact = exact_bytes(value[old_kept_n:], trellis[key])
        assert old_exact and trellis_exact
        assert value.dtype == np.float32 and np.isfinite(value).all()
        per_parameter[key] = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "old_retained_bitwise_exact": old_exact,
            "all_T_bitwise_exact": trellis_exact,
            "old_partition_array_sha256": array_sha256(value[:old_kept_n]),
            "T_partition_array_sha256": array_sha256(value[old_kept_n:]),
        }
    assert np.all(output["scales"] > 0)
    assert np.all((output["opacity"] >= 0) & (output["opacity"] <= 1))
    assert not np.any(output["coefficients"][:, 1:, :])
    assert np.all(output["high_frequency_permission"] == 1.0)
    assert np.all(output["real_V_supervision_weight"] == 1.0)
    assert np.array_equal(output["source_row_id"][old_kept_n:], np.arange(trellis_n))
    assert np.array_equal(output["native_global_row_id"][:old_kept_n], native_global[keep])
    assert not np.any(output["teacher_unknown"][:old_kept_n])
    assert np.all(output["teacher_prior_weight"][old_kept_n:][unknown_trellis] == 1.0)
    assert np.all(output["teacher_prior_weight"][old_kept_n:][direct_trellis] == 0.0)
    input_unchanged = {key: file_sha256(ROOT / PATHS[key]) == record["sha256"] for key, record in inputs.items()}
    assert all(input_unchanged.values())
    verification = {
        "schema": "a20.joint_initialization.cpu_verification.v1", "passed": True,
        "joint_rows": total_n, "old_retained_rows": old_kept_n, "T_retained_rows": trellis_n,
        "all_T_rows_once_in_source_order": True,
        "old_retained_rows_once_in_source_order": True,
        "parameters": per_parameter,
        "input_files_unchanged": input_unchanged,
        "old_disallowed_rows_in_joint": 0,
        "old_opaque_high_confidence_rows_retained": int(high_geometry[keep].sum()),
        "native_mapping_unique_valid_rows": len(np.unique(native_global[keep])),
        "SH_active_degree": 0,
        "activated_opacity_range": [float(output["opacity"].min()), float(output["opacity"].max())],
        "opacity_exactly_one_rows_preserved": int((output["opacity"] == 1.0).sum()),
        "quaternion_norm_range_untouched": [float(np.linalg.norm(output["rotation"], axis=1).min()), float(np.linalg.norm(output["rotation"], axis=1).max())],
        "T_direct_observed_rows": int(direct_trellis.sum()),
        "T_observed_surface_075m_rows": int(within_trellis.sum()),
        "T_unknown_rows": int(unknown_trellis.sum()),
        "T_unknown_full_prior_weight_rows": int(np.sum(output["teacher_prior_weight"][old_kept_n:][unknown_trellis] == 1.0)),
        "real_V_supervision_all_rows_enabled": True,
        "hidden_high_frequency_permission_all_rows_enabled": True,
        "GPU_used": False, "SSH_used": False,
        "joint_actor_init_sha256": file_sha256(joint_path),
    }
    write_json(HERE / "cpu_exact_source_verification.json", verification)

    fields = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in payload.items()}
    counts = {
        "old_input": old_n, "old_native_branch2": int(native.sum()),
        "old_legacy_branch3_excluded": int(legacy.sum()),
        "old_native_exact_3sigma_intersects_fixed_OBB": int((native & intersects).sum()),
        "old_native_exact_3sigma_disjoint_excluded": int((native & ~intersects).sum()),
        "old_K_mixed_or_non_target_excluded_all_including_overlap": int(mixed.sum()),
        "old_K_mixed_or_non_target_intersects_OBB_excluded": int((native & intersects & mixed).sum()),
        "old_kept": old_kept_n, "old_excluded_union": int((~keep).sum()),
        "old_opaque_high_geometry_confidence_kept": int(high_geometry[keep].sum()),
        "old_low_geometry_confidence_kept": int((keep & ~high_geometry).sum()),
        "old_prior_visible_source_evidence": int(old_observed.sum()),
        "old_kept_without_prior_visible_source_evidence": int((~old_observed).sum()),
        "T_kept": trellis_n, "T_direct_observed": int(direct_trellis.sum()),
        "T_observed_surface_075m": int(within_trellis.sum()),
        "T_unknown": int(unknown_trellis.sum()), "joint_total": total_n,
    }
    policy = {
        "old_contribution_identity_lock": False,
        "delete_T_to_preserve_old_contribution": False,
        "trainable_actor_parameters_for_both_sources": list(PARAMETERS),
        "unchanged_source_values_at_initialization_only": True,
        "frozen_context": ["road", "cameras", "actor_trajectory", "other_vehicles", "background_scene"],
        "real_V_supervision": "All genuine V pixels remain eligible; row metadata never masks real V photometric supervision.",
        "high_frequency_permission": "All rows equal 1, including hidden T. No observation mask disables high-frequency permission.",
        "teacher_reference": "Before optimization, clone all T parameters from joint rows [8925,419005) into immutable teacher buffers. These bytes equal the parent T file. Never replace the teacher by current student parameters.",
        "teacher_multiview_preservation": "Use frozen T multi-view teacher targets and per-row teacher_prior_weight; full weight outside the 0.075m support neighbourhood, smooth transition within it.",
        "source_cleaning_scope": "Exclusions remove duplicated rows from the actor initialization only. They do not delete corresponding native scene background or K_other rows.",
        "native_scene_dedup": "The 8925 kept native_global_row_ids identify duplicate source copies in a combined renderer. Only those IDs are emitted as dedup candidates; excluded native context rows, especially all 1831 K-mixed rows, remain frozen context.",
    }
    manifest = {
        "schema": "a20.clean_joint_actor_initialization.v1",
        "state": "CPU_EXACT_SOURCE_VERIFIED_READY_FOR_JOINT_RECONSTRUCTION",
        "artifact": {"path": "joint_actor_init.npz", "sha256": file_sha256(joint_path), "bytes": joint_path.stat().st_size},
        "parents": inputs,
        "coordinate_frame": {"unit": "metre", "x": "forward", "y": "left", "z": "up", "same_actor_frame_both_sources": True},
        "parameter_conventions": {"scales": "activated standard deviations in metres", "rotation": "wxyz; original source quaternion values preserved", "opacity": "activated [0,1], including exact 1.0", "coefficients": "[N,16,3], SH0 active; higher slots exactly zero"},
        "row_order": [{"source_id": 0, "source": "old", "start": 0, "stop": old_kept_n}, {"source_id": 1, "source": "trellis", "start": old_kept_n, "stop": total_n}],
        "counts": counts,
        "fields": fields,
        "field_semantics": {
            "source_row_id": "Row number in the corresponding immutable old/T parent NPZ, not original packed scene source_row.",
            "native_global_row_id": "Exact original native scene global row ID from identity map; -1 for generated T.",
            "original_source_row_id": "Original old.source_row packed source identity; -1 for generated T.",
            "geometry_confidence": "Prior confidence, not a hard freeze: old low .10, old 17 qualified opaque 1.0, all generated T .50.",
            "appearance_confidence": "Prior confidence: old .60/.90 for qualified17; T .50+.40*observed_surface_confidence.",
            "opacity_confidence": "Old ordinary .10, qualified17 .80; T .50. Initial opacity itself is unchanged.",
            "is_observed": "Direct source evidence only: old prior visible_optimizer_rows/positive initial contribution (5473); T direct observed appearance support (12865). It is not a current visibility claim or training exclusion mask.",
            "has_native_source_evidence": "Real-scene provenance: true for all retained old rows; does not certify old geometry.",
            "observed_surface": "Old direct source visibility evidence; T nearest 3D direct appearance support distance <= .075m.",
            "observed_surface_confidence": "T: 1-smoothstep(clamp(distance/.075,0,1)); old: direct source visibility evidence cast to float32.",
            "distance_to_T_direct_observed_surface_m": "3D Euclidean actor-space centre distance; metadata only. It neither changes source parameters nor restricts real V supervision.",
            "teacher_unknown": "T rows outside the .075m observed-surface neighbourhood only. Old rows are never labeled generated teacher unknown.",
            "teacher_prior_weight": "T: 1-observed_surface_confidence; old: 0. Unknown T rows have full weight 1.",
            "teacher_reference_row_id": "Immutable parent T row identity, -1 for old.",
            "high_frequency_permission": "Float32 ones for every row, including teacher_unknown.",
            "real_V_supervision_weight": "Float32 ones for every row; actual per-view V and rendering determine losses.",
        },
        "observed_surface_neighbourhood": {"radius_m": RADIUS_M, "method": "CPU cKDTree exact nearest centre, workers=1", "weight": "1-(3u^2-2u^3), u=clip(d/.075,0,1)", "T_source_resampled": False},
        "cleanup": {"path": "old_source_cleanup.npz", "sha256": file_sha256(HERE / "old_source_cleanup.npz"), "keep_rule": "source_branch == 2 AND exact full 3sigma ellipsoid intersects fixed OBB AND NOT old_K_mixed_or_non_target", "reason_bits": {"1": "legacy engineering lower branch3", "2": "branch2 exact 3sigma ellipsoid disjoint fixed OBB", "4": "old mixed K_other or known non-target"}, "primary_reason_counts": {"1": int((primary_reason == 1).sum()), "2": int((primary_reason == 2).sum()), "4": int((primary_reason == 4).sum())}, "surface_shell_or_mu_inside_filter": False, "old_interior_uncertain_gaussians_retained": True},
        "native_scene_bindings": bindings["input_native_scene_bindings"],
        "policy": policy,
        "verification": {"path": "cpu_exact_source_verification.json", "passed": True},
        "generator": {"path": Path(__file__).name, "sha256": file_sha256(Path(__file__))},
        "execution": {"CPU_only": True, "GPU_used": False, "SSH_used": False, "source_files_modified": False, "initial_parameters_modified": False, "noise_added": False, "opacity_rescaled": False, "T_resampled": False},
    }
    write_json(HERE / "manifest.json", manifest)
    write_json(HERE / "interface_proposal.json", {"schema": manifest["schema"], "counts": counts, "fields": fields, "policy": policy, "status": "IMPLEMENTED_AND_VERIFIED"})
    print(json.dumps({"artifact": joint_path.as_posix(), "sha256": manifest["artifact"]["sha256"], "counts": counts, "CPU_exact_source_verification": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
