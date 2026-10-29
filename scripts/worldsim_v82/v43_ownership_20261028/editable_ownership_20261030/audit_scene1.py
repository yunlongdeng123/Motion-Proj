"""场景一可编辑归属的只读输入审计；不训练、不调用 CUDA、不打开 SSH。

统计已保存的观测与行选择，输出复核候选；不输出最终语义/实例真值。
运行：python work/v43_editable_ownership_20261030/audit_scene1.py
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

CTX = "14705303724557273004_3105_000_3125_000"
RAWS = list(range(77, 126))
ANCHORS = [98, 113, 114]
NS, NA = 918850, 6241
SCHEMA = "v43-editable-ownership-cpu-audit-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def assignment(path):
    source = Path(path).read_text(encoding="utf-8-sig")
    start = source.find("={", source.find(";") + 1) if "||{};" in source else source.find("=")
    require(start >= 0, "Missing data assignment: " + str(path))
    return json.JSONDecoder().raw_decode(source[start + 1:])[0]


def unpack(packed):
    raw = zlib.decompress(base64.b64decode(packed["data"]))
    return {key: np.frombuffer(raw, dtype=desc["dtype"], count=int(np.prod(desc["shape"])),
                               offset=desc["offset"]).reshape(desc["shape"])
            for key, desc in packed["arrays"].items()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sorted_ids(ids, maximum, label):
    require(ids.ndim == 1 and np.all(np.diff(ids) > 0), label + ": IDs are not unique and sorted")
    require(not len(ids) or (ids[0] >= 0 and ids[-1] < maximum), label + ": ID outside population")


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(levels=[0, .1, .5, .9, 1], values=np.quantile(values, [0, .1, .5, .9, 1]).tolist()) if len(values) else None


def index_join(sorted_source_ids, query_ids):
    """精确持久 ID 连接；未出现在源文件中的行保留 -1。"""
    pos = np.searchsorted(sorted_source_ids, query_ids)
    safe = np.minimum(pos, max(len(sorted_source_ids) - 1, 0))
    valid = (pos < len(sorted_source_ids)) & (sorted_source_ids[safe] == query_ids)
    return np.where(valid, pos, -1)


def native_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=native_scalar) + "\n", encoding="utf8")


def run(root, output):
    root, output = root.resolve(), output.resolve()
    scene = root / "outputs/GS_Scene_Diagnostics_20261027" / CTX
    work = root / "work"
    detection = work / "v43_unknown_detection_20261028"
    r2_root = work / "v43_r2_regression_20261029/r2_delivery/r2_inherited49"
    contract_path = work / "v43_unknown_white_20261029/render_inputs/input_contract.json"
    contract = read(contract_path)
    evidence_path = scene / "r2_ownership_data.js"
    ownership = assignment(evidence_path)
    index = assignment(scene / "gpu_semantics/boundary_index.js")
    full_cloud = assignment(scene / "cloud_full_manifest.js")
    metric = read(work / "v43_unknown_white_20261029/render_inputs/source_assets/v43_actor_10000_metrics.json")
    require(contract["context"] == ownership["context"] == CTX, "Context mismatch")
    require(contract["branch_sizes"] == ownership["branch_sizes"] == dict(static=NS, actor=NA), "Branch population mismatch")
    require(ownership["raws"] == RAWS and sorted(map(int, index["frames"])) == RAWS, "Require exact all49 observations")
    require(index["branch_sizes"]["actor10k"] == [NS, NA] and index["radii"] == [3, 5, 8], "Semantic namespace mismatch")
    require(ownership["static_namespace"] == contract["static_namespace"], "Static checkpoint namespace mismatch")
    require(sha(evidence_path) == contract["source_assets"]["ownership"]["sha256"], "Ownership evidence differs from frozen input contract")
    require(metric["checkpoint_sha256"] == contract["actor_checkpoint_sha256"], "Actor checkpoint receipt mismatch")
    require(full_cloud["static"]["total"] == NS and full_cloud["static"]["checkpoint"] == contract["static_checkpoint_ref"] + "/chkpnt30000.pth", "Full static cloud checkpoint mismatch")

    def relative(path):
        return Path(path).relative_to(root).as_posix()

    source_map = {
        "semantic_all49": dict(path=relative(scene / "gpu_semantics/boundary_index.js"),
            frame_path="gpu_semantics/boundary_{raw:03d}.js", branch="actor10k",
            fields=["ids", "class_id", "mass", "purity", "target_fraction", "vehicle_fraction", "boundary_fraction"],
            shape="rows x fields; IDs < 918850 are static, IDs >= 918850 map to actor_id = ID - 918850",
            selection="historical exporter saved total native alpha*T mass > .001",
            semantic_gate="class_id already contains original total mass > .01 and semantic purity > .65; do not reapply gates to rounded float16",
            precision="mass/fractions are saved float16; class_id is uint8; no full 20-class histogram or pixel-to-row weights saved here"),
        "r2_all49": dict(path=relative(r2_root), fields=["existing_removed_ids", "new_unknown_ids", "final_removed_ids", "row_namespace", "checkpoint_ref"],
            role="historical selection membership only; no semantic/instance ground truth; nonmembership is not background proof"),
        "native_three_anchor": dict(path=relative(detection / "native_target_contact_smoke_r1/evidence"),
            fields=["ids", "static_role", "body_mass", "bg_mass", "shadow_mass", "total_mass", "target_fraction", "shadow_fraction", "bg_depth_rms", "reason_bits"],
            transmittance="actor-omitted frozen static alpha*T; do not mix numerically with full Actor10K semantic transmittance",
            saved_evidence_predicate="role > 0 OR bg_mass > .02; every row passing the present bg_mass > .05 dual-domain diagnostic must therefore be saved",
            precision="contribution scalars are float32; role and threshold results are historical proposals"),
        "background_three_anchor": dict(path=relative(detection / "row_background_evidence_r1"),
            fields=["ids", "background_protection_candidate", "background_geometry_witness", "mixed_color_mismatch", "source_count", "source_raws", "source_geometry_valid", "source_sh_match", "reason_bits"],
            role="independent visible LiDAR surface + stable original RGB/SH witness; all certified background flags false"),
        "surface_three_anchor": dict(path=relative(detection / "row_surface_coverage_r1"),
            fields=["ids", "query_world_fixed", "query_sigma_m", "query_native_opacity", "local49_all_raw_surface_witness", "union87_raw_surface_witness"],
            role="fixed query-world surface coverage; no source Gaussian/SH extrapolation outside49"),
        "actor_pose_all49": dict(path=relative(contract_path), field="groups.v43_actor_10000.frames[raw].original_actor_pose_world",
            precision="float64 original actor-local to global-world rigid pose"),
        "native_shadow_all49": dict(path=relative(detection / "shadow_evidence_r3"),
            fields=["shadow_candidate_strict", "shadow_candidate_proposal", "photo_supported_unknown", "world_point", "source_raw", "reference_rgb_exposure_matched"],
            role="road receiver photo/effect evidence; never car semantic labels"),
        "full_geometry_local": dict(path=relative(scene / "cloud_full_manifest.js"),
            fields=["xyz", "scales", "rotation", "opacity", "t", "sigma_t", "velocity", "ids"],
            coordinates="first-vehicle-local meters; xyz/rotation/scales are float32; native SHM/time-opacity still applies",
            role="all static rows available for CPU covariance inspection; displayed uint8 RGB is DC-only and cannot replace native SH"),
    }

    keys = ["visible", "semantic_known", "vehicle", "road", "vehicle_away_contour", "nonvehicle", "r2_member", "r2_existing", "r2_new"]
    count = {key: np.zeros(NS + NA, np.uint8) for key in keys}
    class_bits = np.zeros(NS + NA, np.uint32)
    mass_sum = np.zeros(NS + NA, np.float64)
    actor_class = np.full((49, NA), 255, np.uint8)  # 255 专门表示未观察，0 表示已观察但语义弃权。
    actor_core = np.full((49, NA), 255, np.uint8)
    anchor_semantics = {}
    per_frame = []
    input_checks = []
    for frame_index, raw in enumerate(RAWS):
        sem_path = scene / index["frames"][str(raw)]
        r2_path = r2_root / f"{raw:03d}/evidence.npz"
        expected = ownership["frames"][str(raw)]["source_sha256"]
        require(sha(sem_path) == expected["boundary"] and sha(r2_path) == expected["r2_evidence"], f"raw{raw}: frozen source binding differs")
        sem = unpack(assignment(sem_path)["actor10k"])
        ids, classes = sem["ids"], sem["class_id"]
        sorted_ids(ids, NS + NA, f"raw{raw} semantics")
        require(sem["boundary_fraction"].shape == (len(ids), 3), "Boundary shape mismatch")
        vehicle = (classes >= 14) & (classes <= 19)
        road = classes == 1
        core = vehicle & (sem["boundary_fraction"][:, 1].astype(np.float64) < .2)
        nonvehicle = (classes > 0) & ~vehicle
        for key, mask in [("visible", np.ones(len(ids), bool)), ("semantic_known", classes > 0),
                          ("vehicle", vehicle), ("road", road), ("vehicle_away_contour", core), ("nonvehicle", nonvehicle)]:
            count[key][ids[mask]] += 1
        class_bits[ids] |= np.where(classes > 0, np.left_shift(np.uint32(1), classes.astype(np.uint32)), 0)
        mass_sum[ids] += sem["mass"].astype(np.float64)
        actor_mask = ids >= NS
        actor_ids = ids[actor_mask] - NS
        actor_class[frame_index, actor_ids] = classes[actor_mask]
        actor_core[frame_index, actor_ids] = core[actor_mask].astype(np.uint8)
        saved_actor = ownership["frames"][str(raw)]["actor"]
        require(np.array_equal(np.flatnonzero(np.asarray(saved_actor["present"])), actor_ids), "Actor visible identity differs")
        require(np.array_equal(np.asarray(saved_actor["default_role"])[actor_ids], core[actor_mask]), "Actor core formula differs")
        r2 = load_npz(r2_path)
        rec = read(r2_root / f"{raw:03d}/receipt.json")
        require(str(r2["row_namespace"]) == rec["row_namespace"] == contract["static_namespace"], "R2 static identity differs")
        require(rec["context"] == CTX and rec["raw"] == raw and rec["camera_index"] == raw - 77, "R2 frame identity differs")
        require(rec["checkpoint_ref"] == contract["actor_checkpoint_ref"] and rec["static_checkpoint_ref"] == contract["static_checkpoint_ref"], "R2 checkpoint binding differs")
        for field, key in [("final_removed_ids", "r2_member"), ("existing_removed_ids", "r2_existing"), ("new_unknown_ids", "r2_new")]:
            sorted_ids(r2[field], NS, f"raw{raw} {field}")
            require(np.array_equal(r2[field], np.asarray(ownership["frames"][str(raw)]["static"][field])), "R2 membership source mismatch")
            count[key][r2[field]] += 1
        require(np.array_equal(r2["final_removed_ids"], np.union1d(r2["existing_removed_ids"], r2["new_unknown_ids"])), "R2 union differs")
        selected = np.isin(ids, r2["final_removed_ids"]) & (ids < NS)
        visible_static = ids < NS
        per_frame.append(dict(raw=raw, static_visible_rows=int(visible_static.sum()),
            static_missing_visible_evidence_rows=NS-int(visible_static.sum()),
            static_road_predicted_rows=int((road & visible_static).sum()),
            static_vehicle_predicted_rows=int((vehicle & visible_static).sum()),
            static_semantic_abstain_visible_rows=int(((classes == 0) & visible_static).sum()),
            actor_visible_rows=len(actor_ids), actor_missing_rows=NA-len(actor_ids),
            actor_core_rows=int(core[actor_mask].sum()), actor_observed_nonvehicle_rows=int(nonvehicle[actor_mask].sum()),
            r2_selected_rows=len(r2["final_removed_ids"]), r2_selected_visible_rows=int(selected.sum()),
            r2_selected_missing_visible_evidence_rows=len(r2["final_removed_ids"])-int(selected.sum()),
            r2_selected_road_predicted_rows=int((selected & road).sum()),
            r2_selected_vehicle_predicted_rows=int((selected & vehicle).sum()),
            r2_selected_semantic_abstain_rows=int((selected & (classes == 0)).sum()),
            r2_selected_other_known_class_rows=int((selected & nonvehicle & ~road).sum())))
        if raw in ANCHORS:
            anchor_semantics[raw] = sem
        input_checks.append(dict(raw=raw, boundary_source_digest_matches=True, r2_source_digest_matches=True,
                                 row_namespace_and_checkpoint_paths_match=True))

    summaries = {}
    for name, start, stop in [("static", 0, NS), ("actor", NS, NS + NA)]:
        c = {key: value[start:stop] for key, value in count.items()}
        summaries[name] = dict(population=stop-start,
            ever_observed_rows=int((c["visible"] > 0).sum()), never_observed_rows=int((c["visible"] == 0).sum()),
            ever_vehicle_prediction_rows=int((c["vehicle"] > 0).sum()), ever_road_prediction_rows=int((c["road"] > 0).sum()),
            both_road_and_vehicle_prediction_rows=int(((c["road"] > 0) & (c["vehicle"] > 0)).sum()),
            ever_vehicle_away_contour_rows=int((c["vehicle_away_contour"] > 0).sum()),
            vehicle_away_contour_all49_rows=int((c["vehicle_away_contour"] == 49).sum()),
            vehicle_away_contour_in_all_observed_frames_rows=int(((c["visible"] > 0) & (c["vehicle_away_contour"] == c["visible"])).sum()),
            never_vehicle_away_contour_rows=int((c["vehicle_away_contour"] == 0).sum()),
            away_contour_and_nonvehicle_prediction_rows=int(((c["vehicle_away_contour"] > 0) & (c["nonvehicle"] > 0)).sum()),
            r2_ever_selected_rows=int((c["r2_member"] > 0).sum()), r2_selected_all49_rows=int((c["r2_member"] == 49).sum()),
            r2_selected_once_rows=int((c["r2_member"] == 1).sum()),
            r2_ever_selected_and_ever_road_prediction_rows=int(((c["r2_member"] > 0) & (c["road"] > 0)).sum()),
            visible_frames_quantiles=quantiles(c["visible"][c["visible"] > 0]))

    native_contract = read(detection / "native_target_contact_smoke_r1/contract.json")
    native_summary = read(detection / "native_target_contact_smoke_r1/summary.json")
    bg_contract = read(detection / "row_background_evidence_r1/contract.json")
    coverage_contract = read(detection / "row_surface_coverage_r1/contract.json")
    require(native_contract["context"] == native_summary["context"] == CTX and native_contract["source_checkpoint_frozen"], "Native source lineage mismatch")
    require(native_summary["source_checkpoint_unchanged"] and native_summary["training_updates"] == 0, "Native asset is not frozen")
    require(bg_contract["row_namespace"] == "original_static_rows" and bg_contract["root"] == str(Path(contract["actor_checkpoint_ref"]).parents[3]).replace("\\", "/"), "Background exporter source differs")
    require(coverage_contract["row_export"] == bg_contract["row_export"], "Background/surface persistent ID export differs")
    # 三份历史导出均通过 probe_detection.load_native 绑定同一 frozen static30K/Actor10K。
    # 本地不保存原二进制checkpoint；不能把代码路径一致写成这次重新校验checkpoint字节。
    loader_source = (detection / "probe_detection.py").read_text(encoding="utf-8-sig")
    require("formal10k/checkpoint010000.pth" in loader_source and "n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000)" in loader_source,
            "Frozen export loader binding differs")
    require(native_contract["body_probe"].endswith("body_probe_r1") and bg_contract["row_export"].endswith("unknown_detection_v3/source_rows_r1"), "Unexpected evidence lineage")

    candidates = []
    anchors = []
    route = {}
    worlds, poses, row_ids = [], [], None

    def identity(population, row):
        return dict(population=population, row_id=int(row),
            row_namespace=contract["static_namespace"] if population == "static" else contract["actor_namespace"],
            checkpoint_ref=contract["static_checkpoint_ref"] + "/chkpnt30000.pth" if population == "static" else contract["actor_checkpoint_ref"])

    def review(reason, population, row, raw=None, **values):
        item = dict(candidate_id=f"{reason}:{raw if raw is not None else 'all49'}:{population}:{int(row)}",
                    reason=reason, **identity(population, row), **values)
        if raw is not None:
            item["raw"] = raw
        candidates.append(item)

    for raw in ANCHORS:
        e = load_npz(detection / f"native_target_contact_smoke_r1/evidence/{raw:03d}.npz")
        b = load_npz(detection / f"row_background_evidence_r1/{raw:03d}.npz")
        g = load_npz(detection / f"row_surface_coverage_r1/{raw:03d}.npz")
        r = load_npz(r2_root / f"{raw:03d}/evidence.npz")
        sorted_ids(e["ids"], NS, "Native IDs")
        sorted_ids(b["ids"], NS, "Background IDs")
        require(np.array_equal(b["ids"], g["ids"]), "Background/surface row order differs")
        require(not b["background_certified"].any() and not g["background_certified"].any(), "Unexpected certified background")
        if row_ids is None:
            row_ids = b["ids"]
        require(np.array_equal(row_ids, b["ids"]), "Cross-query static IDs differ")
        worlds.append(g["query_world_fixed"])
        poses.append(np.asarray(contract["groups"]["v43_actor_10000"]["frames"][str(raw)]["original_actor_pose_world"], np.float64))
        require(np.isfinite(worlds[-1]).all(), "Missing original row center")
        require(np.allclose(poses[-1][:3, :3].T @ poses[-1][:3, :3], np.eye(3), atol=1e-8)
                and np.allclose(poses[-1][3], [0, 0, 0, 1]), "Actor pose is not a rigid local-to-world transform")
        total = np.maximum(e["total_mass"], np.float32(1e-8))
        bg_fraction = e["bg_mass"] / total
        body_fraction = e["target_fraction"]
        dual = (body_fraction > .05) & (e["bg_mass"] > .05)
        source_pos = index_join(row_ids, e["ids"])
        e_in_r2 = np.isin(e["ids"], r["final_removed_ids"])
        sem = anchor_semantics[raw]
        sem_pos = index_join(sem["ids"], e["ids"])
        sem_present = sem_pos >= 0
        sem_class = np.full(len(e["ids"]), 255, np.uint8)
        sem_class[sem_present] = sem["class_id"][sem_pos[sem_present]]
        native_pos = index_join(e["ids"], row_ids)
        native_present = native_pos >= 0
        native_roles = np.full(len(row_ids), 255, np.uint8)
        native_roles[native_present] = e["static_role"][native_pos[native_present]]
        in_r2 = np.isin(row_ids, r["final_removed_ids"])
        joined = {}
        for field in ["background_protection_candidate", "background_geometry_witness", "mixed_color_mismatch"]:
            mask = b[field]
            joined[field] = dict(rows=int(mask.sum()), current_r2_members=int((mask & in_r2).sum()),
                native_role4_rows=int((mask & (native_roles == 4)).sum()),
                native_evidence_missing_rows=int((mask & ~native_present).sum()))
        for field in ["local49_all_raw_surface_witness", "union87_raw_surface_witness"]:
            joined[field] = dict(rows=int(g[field].sum()), current_r2_members=int((g[field] & in_r2).sum()))
        anchors.append(dict(raw=raw, independent_row_export_population=len(row_ids),
            native_saved_evidence_population=len(e["ids"]), native_role4_proposal_rows=int((e["static_role"] == 4).sum()),
            native_rows_missing_independent_row_export=int((source_pos < 0).sum()),
            two_domain_contribution_candidate_rows=int(dual.sum()),
            shadow_and_background_contribution_candidate_rows=int(((e["shadow_mass"] > .01) & (e["bg_mass"] > .05)).sum()),
            joins=joined, two_domain_body_fraction_quantiles=quantiles(body_fraction[dual]),
            two_domain_background_fraction_quantiles=quantiles(bg_fraction[dual])))
        route[f"raw{raw}_native_ids"] = e["ids"]
        route[f"raw{raw}_two_domain_candidate"] = dual
        route[f"raw{raw}_target_fraction"] = body_fraction
        route[f"raw{raw}_background_fraction"] = bg_fraction
        route[f"raw{raw}_shadow_fraction"] = e["shadow_fraction"]
        route[f"raw{raw}_native_semantic_class_or255missing"] = sem_class
        route[f"raw{raw}_background_protection_candidate"] = b["background_protection_candidate"]
        route[f"raw{raw}_color_mismatch_with_surface_witness"] = b["mixed_color_mismatch"]
        route[f"raw{raw}_r2_member_in_row_export"] = in_r2
        # 按两个域较小贡献比例排序，只挑少量复核；不把分数解释成归属概率。
        dual_index = np.flatnonzero(dual)
        order = np.lexsort((e["ids"][dual_index], -np.minimum(body_fraction[dual], bg_fraction[dual])))[:3]
        for j in dual_index[order]:
            bi = source_pos[j]
            review("same_frame_target_background_contribution", "static", e["ids"][j], raw,
                target_fraction=float(body_fraction[j]), background_fraction=float(bg_fraction[j]),
                body_mass=float(e["body_mass"][j]), background_mass=float(e["bg_mass"][j]),
                total_mass=float(e["total_mass"][j]), native_role=int(e["static_role"][j]),
                semantic_class_full_model=int(sem_class[j]),
                independent_background_geometry_witness=bool(b["background_geometry_witness"][bi]) if bi >= 0 else None,
                independent_row_export_present=bool(bi >= 0),
                r2_member=bool(e_in_r2[j]), limitation="两屏幕域贡献，背景证据仍弱；不是三维真混合或安全可删标签")
        for bi in np.flatnonzero(b["background_protection_candidate"]):
            review("independent_background_protection_candidate", "static", row_ids[bi], raw,
                r2_member=bool(in_r2[bi]), source_count=int(b["source_count"][bi]),
                matched_source_count=int(b["source_sh_match_count"][bi]),
                source_pair=b["independent_matched_source_pair"][bi].tolist(),
                certified=False, limitation="已有正背景候选仍无认证无阴影真值；不得自动改成车标签")
        bi = np.flatnonzero(b["mixed_color_mismatch"])
        order = np.lexsort((row_ids[bi], -b["source_count"][bi].astype(np.int16)))[:2]
        for j in bi[order]:
            review("background_surface_witness_color_mismatch", "static", row_ids[j], raw,
                r2_member=bool(in_r2[j]), source_count=int(b["source_count"][j]),
                native_role_or255missing=int(native_roles[j]),
                query_sh_rgb_mean_abs_error=float(b["query_sh_rgb_mean_abs_error"][j]),
                limitation="真实背景几何见证与SH颜色冲突；不是同帧两域空间贡献的同义词，也不能自动判car")
        shadow = (e["shadow_mass"] > .01) & (e["bg_mass"] > .05) & (sem_class == 1)
        si = np.flatnonzero(shadow)
        order = np.lexsort((e["ids"][si], -np.minimum(e["shadow_fraction"][si], bg_fraction[si])))[:2]
        for j in si[order]:
            review("road_receiver_effect_contribution", "static", e["ids"][j], raw,
                semantic_class=1, shadow_fraction=float(e["shadow_fraction"][j]),
                background_fraction=float(bg_fraction[j]), strict_shadow_reason=bool(e["reason_bits"][j] & 4),
                limitation="photo/contact/strict效果贡献候选；道路语义与目标car effect分开，效果关联不是物理因果证明")

    world = np.stack(worlds)
    canonical = np.stack([(w - p[:3, 3]) @ p[:3, :3] for w, p in zip(worlds, poses)])
    anchor113 = ANCHORS.index(113)
    world_drift = np.max(np.linalg.norm(world - world[anchor113], axis=2), axis=0)
    canonical_drift = np.max(np.linalg.norm(canonical - canonical[anchor113], axis=2), axis=0)
    route.update(row_export_ids=row_ids, three_query_raws=np.asarray(ANCHORS, np.int16),
                 static_parent_world_max_drift_m=world_drift.astype(np.float32),
                 static_parent_target_canonical_max_drift_m=canonical_drift.astype(np.float32))
    canonical_summary = dict(population=len(row_ids), raws=ANCHORS, reference_raw=113,
        world_max_displacement_m=quantiles(world_drift), target_canonical_max_displacement_m=quantiles(canonical_drift),
        target_canonical_more_stable_rows=int((canonical_drift < world_drift).sum()),
        target_canonical_within_025m_rows=int((canonical_drift <= .25).sum()),
        world_within_025m_rows=int((world_drift <= .25).sum()),
        meaning="只比较旧static父中心轨迹，不判定贡献表面归属；旧中心不随车不能证明其目标外观支撑不存在",
        consequence="后续canonical约束应施加于有真实深度/表面支持的子高斯，不把父中心直接硬绑到actor")
    for item in list(candidates):
        if item["reason"] != "same_frame_target_background_contribution":
            continue
        j = int(np.searchsorted(row_ids, item["row_id"]))
        if j >= len(row_ids) or int(row_ids[j]) != item["row_id"]:
            item["parent_center_geometry_missing"] = True
            continue
        item["parent_world_max_drift_m"] = float(world_drift[j])
        item["parent_target_canonical_max_drift_m"] = float(canonical_drift[j])
    # 稳定观测锚点仍只是复核正例提案，没有补造独立真值。
    stable_actor = np.flatnonzero(count["vehicle_away_contour"][NS:] == 49)
    order = np.lexsort((stable_actor, -mass_sum[NS:][stable_actor]))[:3]
    for row in stable_actor[order]:
        review("actor_core_observed_all49", "actor", row, core_frames=49, observed_frames=49,
            limitation="现有Actor10K资产内稳定语义/轮廓支持；不是新学到的实例GT")
    conflict = np.flatnonzero((count["road"][:NS] > 0) & (count["vehicle"][:NS] > 0) & (count["r2_member"][:NS] > 0))
    score = np.minimum(count["road"][conflict], count["vehicle"][conflict])
    order = np.lexsort((conflict, -score.astype(np.int16)))[:3]
    for row in conflict[order]:
        review("cross_view_road_vehicle_prediction_conflict", "static", row,
            observed_frames=int(count["visible"][row]), road_prediction_frames=int(count["road"][row]),
            vehicle_prediction_frames=int(count["vehicle"][row]), r2_member_frames=int(count["r2_member"][row]),
            limitation="2D预测跨帧冲突；可能由可见性、表面共享、语义误差或错误几何导致，不能单独定混合机制")

    partition = np.zeros(NS + NA, np.uint8)
    partition[count["visible"] > 0] = 1
    partition[(count["road"] > 0) & (count["vehicle"] == 0)] = 2
    partition[(count["vehicle"] > 0) & (count["road"] == 0)] = 3
    partition[(count["road"] > 0) & (count["vehicle"] > 0)] = 4
    partition[(count["semantic_known"] > 0) & (count["road"] == 0) & (count["vehicle"] == 0)] = 5
    route.update({f"{name}_frames": values for name, values in count.items()})
    route.update(semantic_class_seen_bits=class_bits, observed_support_partition=partition,
                 actor_class_by_raw_or255missing=actor_class, actor_core_by_raw_or255missing=actor_core,
                 raws=np.asarray(RAWS, np.int16), branch_sizes=np.asarray([NS, NA], np.int32))

    limitations = [
        "未观察的行/帧记missing；已观察但class_id=0记语义弃权；两者不作为car或noncar负标签。",
        "R2 membership记录历史干预选择，不能作训练标签、GT、独立召回或精确率分母。",
        "semantic full Actor10K与native_contact actor-omitted的alpha*T不同；本审计只按持久ID连接证据，不把质量数值直接相减。",
        "同帧目标/背景双域贡献、跨帧语义冲突、背景几何+颜色冲突是三种不同观察，均不证明三维可分性。",
        "背景/阴影候选均无完整human GT；source_shadow_state_unknown=true，certified_background=0。",
        "本地只校验既有证据绑定和导出器checkpoint路径；没有重开远端checkpoint字节，也没有重新生成贡献。",
        "三锚点是既有场景训练时刻；不是未见测试集。float16语义比例只能作已有证据的统计，不作新阈值精确重放。",
    ]
    metadata = dict(schema=SCHEMA, context=CTX, static_checkpoint_ref=contract["static_checkpoint_ref"],
        static_namespace=contract["static_namespace"], actor_checkpoint_ref=contract["actor_checkpoint_ref"],
        actor_checkpoint_sha256_receipt=contract["actor_checkpoint_sha256"], actor_namespace=contract["actor_namespace"],
        branch_sizes=dict(static=NS, actor=NA), source_fields=source_map,
        denominators=dict(all49_frames=49, static_row_population=NS, actor_row_population=NA,
            three_anchor_row_export_population=len(row_ids), missing_policy="Unobserved is missing, never negative",
            membership_policy="All49 historical selection events, independent of whether full-model semantic evidence was visible"),
        parameters=dict(core_vehicle_classes=list(range(14, 20)), semantic_road_class=1, boundary_radius_px=5,
            boundary_fraction_strict_less_than=.2, original_semantic_acceptance_not_reapplied=True,
            two_domain_target_fraction_strict_gt=.05, two_domain_background_absolute_mass_strict_gt=.05,
            two_domain_ranking="descending min(target_fraction, background_fraction); row ID resolves ties",
            parent_drift_reference_raw=113, parent_drift_threshold_diagnostic_m=.25,
            inherited_native_classifier=native_contract["configuration"]),
        provenance_checks=dict(all49_frozen_source_bindings_and_checkpoint_paths=input_checks,
            native_background_surface_exporters_share_loader="probe_detection.load_native",
            original_binary_checkpoints_reopened_locally=False,
            cross_checkpoint_or_cross_namespace_join_performed=False,
            missing_ids_filled_with_negative_labels=False),
        all49_summary=summaries, per_frame=per_frame, anchors=anchors,
        parent_center_canonical_diagnostic=canonical_summary,
        route_npz=dict(path=relative(output / "row_evidence_partitions.npz"),
            combined_count_array_layout="[0:918850] static, [918850:925091] Actor10K; array index is persistent branch row ID plus offset",
            actor_class_core_missing_sentinel=255,
            support_partition_codes={"0": "never observed", "1": "observed; no accepted semantic class",
                "2": "road observed, vehicle not observed", "3": "vehicle observed, road not observed",
                "4": "both road and vehicle observed", "5": "other accepted semantics only"},
            support_partition_is_final_owner=False,
            anchor_native_arrays="native_ids ordering; independent background arrays use row_export_ids ordering; missing semantic class=255",
            geometry_arrays="row_export_ids ordering, meters"),
        code_entries=dict(native_load="work/v43_unknown_detection_20261028/probe_detection.py:load_native/kwargs/attribution",
            exact_contribution="auxiliary-feature adjoint; sum-row mass closure checked by original GPU exporters",
            semantic_extractor="work/v43_gpu_20261028/attribute_boundary_semantics.py",
            editable_actor="work/v43_gpu_20261028/train_actor.py:Actor.kwargs/project/render",
            original_decomposition="work/v43_gpu_20261028/train_target_unknown.py:Decomposition is whole-row gate/DC plus new background, not a two-child spatial split"),
        next_gpu_evidence_needed=[
            "Same frozen static30K+Actor10K, three anchors: recompute per-row semantic/instance/depth/effect support in a shared declared transmittance context.",
            "Keep both supported car and road children; use actual LiDAR/visible surface points for depth-conditioned moments and target canonical coordinates, retaining missing/unknown.",
            "Compare baseline, old whole-row proposal, paired children retained, and car-child-only edit; report identity RGB/alpha/depth preservation and road-region damage.",
            "Road-receiver shadow effect gets a separate target-linked effect field; no car semantic relabeling of road rows.",
        ], limitations=limitations, execution=dict(device="CPU", CUDA_calls=0, SSH_calls=0,
            training_updates=0, source_files_modified=False), quality_accepted=False, human_verdict=None)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "row_evidence_partitions.npz", **route)
    review_document = dict(schema="v43-editable-ownership-review-candidates-v1", context=CTX,
        candidate_count=len(candidates), selection="deterministic evidence-rich examples; not a representative accuracy sample",
        candidates=candidates, limitations=limitations, human_verdict=None)
    dump(output / "review_candidates.json", review_document)
    dump(output / "cpu_audit.json", metadata)
    # 核对写出的机器接口；不把文件存在冒充GPU或科研质量通过。
    reread = read(output / "cpu_audit.json")
    with np.load(output / "row_evidence_partitions.npz", allow_pickle=False) as arrays:
        require(np.array_equal(arrays["actor_class_by_raw_or255missing"], actor_class), "Actor missing state serialization differs")
        require(np.array_equal(arrays["r2_member_frames"], count["r2_member"]), "Membership serialization differs")
        require(reread["all49_summary"] == summaries, "JSON summary serialization differs")
    return dict(output=str(output), all49_summary=summaries,
                anchor_two_domain_rows={str(a["raw"]): a["two_domain_contribution_candidate_rows"] for a in anchors},
                candidate_count=len(candidates), parent_center_canonical_diagnostic=canonical_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "outputs/V43_Ownership_20261027/gpu_v43/editable_ownership_20261030"
    print(json.dumps(run(args.root, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
