"""独立sceneSpec/GN11准入合同。导入和默认dry-run仅用stdlib，不导入Torch/CUDA。"""
import ast
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import zipfile

NODES = (10000, 20000, 30000)
PAIR_IGNORED = {"hybrid_arm", "model_path", "paper_sgcm_enabled", "hybrid_nps_enabled"}
DATA_IGNORED = {"source_path", "model_path", "start_frame", "end_frame"}
REMOTE_BASE = Path("/root/autodl-tmp/drivingeditor_repro")
READINESS_STATUS = "PASS_CPU_PREPARATION_TRAINING_GATED"


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def same_path(a, b):
    return Path(a).resolve() == Path(b).resolve()


def below(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or parent not in path.parents:
        raise ValueError("目标必须是指定目录内的独立子路径: " + str(path))
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scene_spec(value):
    require(isinstance(value, dict), "sceneSpec必须是JSON对象")
    key = value.get("scene_id", value.get("scene_name"))
    match = re.fullmatch(r"scene(\d{4})_(\d{3})_(\d{3})", str(key))
    require(match is not None, "未知/未识别scene不能作为其他场景替代输入")
    prefix, first, last = match.groups()
    first, last = int(first), int(last)
    require(last - first == 48, "独立协议要求真实49连续source frames")
    ids = value.get("frame_ids")
    require(ids == [prefix + ("%03d" % i) for i in range(first, last + 1)],
            "49 frame IDs必须与真实scene/window一致")
    require(value.get("frame_count", len(ids)) == 49, "frame_count不是49")
    raw_range = value.get("raw_frame_range", value.get("source_frame_window"))
    require(raw_range == [first, last], "raw frame范围与scene/IDs不一致")
    camera = value.get("camera_directory")
    if camera is None and value.get("camera") == "FRONT":
        camera = "image_0"
    require(camera == "image_0", "当前继承实现仅确认FRONT image_0；其他camera需另核验")
    stamps = value.get("model_timestamp_values", value.get("model_timestamps"))
    require(isinstance(stamps, list) and len(stamps) == 49
            and all(finite_number(t) for t in stamps), "需要49个有限model timestamps")
    require(all(abs(t - (-.48 + .02 * i)) < 1e-7 for i, t in enumerate(stamps)),
            "timestamp合同必须为49点[-.48,.48]/.02，不冒称真实秒")
    anchor = value.get("fixed_pose_anchor", value.get("anchor_pose_frame"))
    index = value.get("fixed_pose_camera_index", value.get("anchor_pose_index", 10))
    require(type(index) is int and index == 10 and anchor == ids[index],
            "anchor必须是预选的实际local-index10，不能使用scene17 fallback")
    require(value.get("original_WH", value.get("expected_source_wh")) == [1920, 1280],
            "真实source尺寸不是继承的W1920 H1280")
    require(value.get("render_WH", value.get("expected_render_wh")) == [480, 320],
            "实际render尺寸必须W480 H320")
    require(value.get("resolution_scale", 4) == 4, "NPS sum合同需要scale4")
    source = value.get("source_path")
    require(isinstance(source, str) and
            (Path(source).is_absolute() or PurePosixPath(source).is_absolute()), "source_path需绝对路径")
    priors = value.get("actor_PLY_paths", value.get("actor_ply_paths"))
    require(isinstance(priors, list) and priors and len(set(priors)) == len(priors),
            "需要明确真实actor PLY路径")
    expected_priors = [str(Path(source) / ("points3d_dynamic_%d.ply" % i)) for i in range(len(priors))]
    require(all(same_path(a, b) for a, b in zip(priors, expected_priors)),
            "actor PLY必须是该scene内连续编号的真实先验")
    return {"scene_id": key, "source_path": source, "frame_ids": ids,
            "camera_directory": camera, "frame_ids_with_camera": [camera + "/" + f for f in ids],
            "raw_frame_range": [first, last], "model_timestamps": [float(t) for t in stamps],
            "original_wh": [1920, 1280], "render_wh": [480, 320],
            "anchor_frame": anchor, "anchor_index": index,
            "anchor_timestamp": float(stamps[index]), "actor_ply_paths": priors}


def readiness_contract(value, spec):
    require(value.get("status") == READINESS_STATUS, "场景实际CPU准备未通过")
    require(value.get("gpu_used") is False and value.get("cuda_initialized") is False
            and value.get("training_started") is False, "CPU准备receipt须明确无GPU/训练")
    require(value.get("training_ready") is False,
            "该receipt只证明CPU准备；不能把实际GPU/guard验证补写成通过")
    require(same_path(value.get("scene_copy", ""), spec["source_path"]), "readiness绑定了不同scene copy")
    require(value.get("train_cameras") == 49 and value.get("test_cameras") == 0,
            "CPU reader未确认49 train/0 independent test")
    require(value.get("fixed_pose_anchor") == spec["anchor_frame"], "readiness anchor不匹配")
    rows = value.get("canonical_full_mask_validation", [])
    require(len(rows) == 49 and [r.get("frame_id") for r in rows] == spec["frame_ids"],
            "CPU canonical mask frame identities不完整")
    require(all(r.get("actual_canonical_mask_loader_exact") is True for r in rows),
            "actual canonical mask loader尚未逐帧通过")
    views = value.get("SGCM_view_inputs", {})
    require(views.get("count") == 49 and set(views.get("ids", [])) == set(spec["frame_ids"]),
            "actual SGCM view inputs不完整")
    require(value.get("training_guard_enabled_in_candidate_config") is True
            and value.get("cpu_geometry_guard_enabled") is False, "CPU geometry与真实training guard角色不清")
    geometry = value.get("SGCM_geometry", {})
    require(geometry.get("status") == "validated" and
            finite_number(geometry.get("meter_to_model_scale")) and geometry["meter_to_model_scale"] > 0,
            "本scene SGCM metric geometry尚未通过")
    fit = geometry.get("ground_fit", {})
    require(finite_number(fit.get("rmse_m")) and fit["rmse_m"] <= .1, "实际ground fit无有效RMSE")
    transform = value.get("transform_matrix")
    require(isinstance(transform, list) and len(transform) == 4 and
            all(isinstance(r, list) and len(r) == 4 and all(finite_number(v) for v in r) for r in transform),
            "实际loader4x4 transform缺失")
    return {"status": "SCENE_CPU_PREPARED_ACTUAL_GPU_BINDING_PENDING",
            "scene_id": spec["scene_id"], "ground_rmse_m": fit["rmse_m"],
            "guard_runtime_verified": False, "restoration_verified": False}


def validate_gn11_config(config):
    require(config.get("hybrid_method") is True and config.get("paper_method") is True,
            "必须保留2024 hybrid目标与完整checkpoint设施")
    require(config.get("hybrid_arm") == "GN11" and
            all(config.get(k) is True for k in ("hybrid_nps_enabled", "paper_sgcm_enabled", "sgcm_guard_enabled")),
            "入口仅允许corrected GN11，不能按名字触发其他臂")
    require(type(config.get("iterations")) is int and config["iterations"] == 30000
            and config.get("position_lr_max_steps") == 30000, "预算/LR horizon必须各自30k")
    require(config.get("start_checkpoint") is None and type(config.get("seed")) is int
            and config["seed"] == 0, "仅fromscratch seed0")
    require(config.get("eval") is False and config.get("resolution") == -1
            and config.get("resolution_scales") == [4], "只允许49全观测/res-1/scale4")
    require(config.get("including_dynamic") is True and config.get("paper_eval_train_each_checkpoint") is True,
            "完整动态分支及49节点评价必须启用")
    require(config.get("data_device") == "cuda", "训练候选必须保留CUDA配置，不能使用CPU geometry临时配置")
    for name in ("checkpoint_iterations", "save_iterations", "test_iterations"):
        require(config.get(name) == list(NODES), name + "必须为10k/20k/30k")
    weights = {"hybrid_lambda_dynamic_rgb": 5.0, "hybrid_lambda_sky_rgb": 5.0,
               "hybrid_lambda_static_rgb": 1.0, "hybrid_lambda_dynamic_opacity": 100.0,
               "hybrid_time_reg_eps": 1e-8, "paper_lambda_nps": .01, "paper_lambda_sky": .05,
               "paper_lambda_depth": .1, "lambda_t_reg": .1, "lambda_v_reg": .01,
               "lambda_inv_depth": .1, "lambda_v_smooth": .1, "lambda_opacity_entropy": .05,
               "lambda_dssim": .2, "lidar_decay": 1.0}
    for name, value in weights.items():
        require(type(config.get(name)) is type(value) and config[name] == value,
                "共同目标值/类型被改变: " + name)
    require(config.get("hybrid_opacity_reset_policy") == "densification_only", "reset共同修复未保留")
    require(config.get("hybrid_gradient_nodes") == [1, 1000, 10000, 15000, 20000, 30000],
            "NPS梯度诊断节点改变")
    return config


def build_config(reference, spec, campaign):
    validate_gn11_config(reference)
    config = dict(reference)
    output = below(Path(campaign) / "runs" / (spec["scene_id"] + "_hybrid_30000_seed0_GN11"), campaign)
    config.update(source_path=spec["source_path"], model_path=str(output),
                  start_frame=spec["raw_frame_range"][0], end_frame=spec["raw_frame_range"][1],
                  followup_scene_spec=spec, followup_static_anchor_frame=spec["anchor_frame"])
    validate_gn11_config(config)
    return config


def png_wh(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
            "PNG header无效: " + str(path))
    return list(struct.unpack(">II", header[16:24]))


def input_inventory(spec):
    source = Path(spec["source_path"])
    ids = spec["frame_ids"]
    for directory, suffix in (
        ("image/" + spec["camera_directory"], ".png"), ("sky/sky_0", ".png"),
        ("dynamic_masks/dynamic_mask_0", ".png"), ("dynamic_image/dynamic_image_0", ".png"),
        ("calib", ".txt"), ("pose", ".txt"), ("velodyne", ".bin")):
        folder = source / directory
        require(sorted(p.stem for p in folder.glob("*" + suffix)) == ids,
                "输入49 frame集合不匹配: " + str(folder))
        for name in ids:
            path = folder / (name + suffix)
            require(path.is_file() and path.stat().st_size > 0, "输入文件缺失/为空: " + str(path))
            if suffix == ".png":
                require(png_wh(path) == spec["original_wh"], "source RGB/mask尺寸错误: " + str(path))
    priors = [source / "points3d.ply"] + [Path(p) for p in spec["actor_ply_paths"]]
    require(sorted(str(p.resolve()) for p in source.glob("points3d_dynamic_*.ply")) ==
            sorted(str(Path(p).resolve()) for p in spec["actor_ply_paths"]), "出现未知actor先验")
    for path in priors:
        require(path.is_file() and path.stat().st_size > 0, "真实PLY缺失: " + str(path))
    require((source / "sgcm_transform_matrix.txt").is_file() or (source / "transform_matrix.txt").is_file(),
            "实际CPU loader transform文件缺失")
    return {"status": "INPUT_FILESETS_AND_HEADERS_PASS", "frames": 49, "original_wh": spec["original_wh"],
            "deep_pixels_revalidated": False, "actual_cpu_mask_reader_evidence": "readiness_file"}


def event_contract(path):
    event = read_json(path)
    require(event.get("status") == "complete" and event.get("iteration") == 30000, "gate只接受完整30k event")
    frames = event.get("frames", [])
    require(len(frames) == 49 and len({r.get("frame_id") for r in frames}) == 49, "gate event缺49唯一frame")
    require(all(r.get("width") == 480 and r.get("height") == 320 and
                finite_number(r.get("timestamp")) for r in frames), "gate frame尺寸/timestamp无效")
    require([r["timestamp"] for r in frames] == sorted(r["timestamp"] for r in frames), "gate frames非时间序")
    require(set(event.get("checkpoints", {})) == {"static", "dynamic", "environment", "training_state"},
            "gate完整模型/Adam/RNG sidecar合同未满足")
    require(event.get("parameters_evaluated") == "after_optimizer_and_sgcm_same_as_checkpoint",
            "gate不是保存后同参数评价")
    return event


def strict_dict_equal(left, right, ignored):
    a = {k: v for k, v in left.items() if k not in ignored}
    b = {k: v for k, v in right.items() if k not in ignored}
    # JSON序列保留true与1、整数与浮点的区别；无需新增checksum。
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def gate_contract(gate_path, spec, reference, verify_steps=False):
    require(gate_path is not None, "--run必须提供明确gate passed JSON")
    gate = read_json(gate_path)
    require(gate.get("schema_version") == 1 and gate.get("status") == "passed", "gate内容未passed，文件名不授权")
    require(gate.get("candidate_arm") == "GN11" and gate.get("reference_arm") == "C00"
            and gate.get("candidate_checkpoint") == gate.get("reference_checkpoint") == 30000,
            "gate不是GN11/C00对应30k比较")
    require(gate.get("paired_contract_pass") is True and gate.get("visual_review_complete") is True,
            "paired合同与坏帧/静态时间视觉复核未通过")
    require(gate.get("severe_numeric_regression") is False
            and gate.get("severe_visual_or_static_regression") is False, "严重回退尚未排除")
    require(spec["scene_id"] in gate.get("allowed_scene_ids", []), "该真实scene未纳入通过的gate范围")
    evidence = gate.get("paired_steps_evidence", {})
    require(evidence.get("status") == "passed" and evidence.get("rows") == 30000,
            "需要root已有30000步配对采样证据")
    events = gate.get("events", {})
    c00, gn11 = event_contract(events.get("C00", "")), event_contract(events.get("GN11", ""))
    require(c00["config"].get("hybrid_arm") == "C00" and
            c00["config"].get("paper_sgcm_enabled") is False and c00["config"].get("hybrid_nps_enabled") is False,
            "gate reference config不是C00")
    validate_gn11_config(gn11["config"])
    require(strict_dict_equal(c00["config"], gn11["config"], PAIR_IGNORED), "两30k event共同配置不一致")
    require(strict_dict_equal(reference, gn11["config"], DATA_IGNORED),
            "候选GN11配置与通过gate的目标/LR/seed/资源合同不一致")
    signature = lambda e: [(r["frame_id"], r["timestamp"], r["width"], r["height"]) for r in e["frames"]]
    require(signature(c00) == signature(gn11), "两30k event frame IDs/timestamps不配对")
    matrix = Path(events["GN11"]).resolve().parents[3] / "matrix_status.json"
    require(read_json(matrix).get("status") == "all_four_arms_complete", "现有四臂尚未全部终态，不能启动followup")
    drop = gate.get("max_psnr_drop_db", 1.0)
    require(finite_number(drop) and drop == 1.0, "gate必须使用用户已确认的1.0dB阈值")
    means = {}
    for label, event in (("C00", c00), ("GN11", gn11)):
        values = {"full": [], "actor": []}
        for frame in event["frames"]:
            metric = read_json(frame["hybrid_metrics"])
            require(metric.get("frame_id") == frame["frame_id"], "gate每帧指标身份不一致")
            for region in values:
                value = metric["regions"][region]["composite_psnr_db"]
                require(finite_number(value), "gate PSNR不是finite")
                values[region].append(value)
        means[label] = {k: math.fsum(v) / 49 for k, v in values.items()}
    require(all(means["GN11"][k] - means["C00"][k] >= -drop for k in ("full", "actor")),
            "实际30k数字触发严重回退，拒绝followup")
    if verify_steps:
        steps = gate.get("paired_steps_files", {})
        with Path(steps["C00"]).open(encoding="utf-8") as a, Path(steps["GN11"]).open(encoding="utf-8") as b:
            for iteration in range(1, 30001):
                ra, rb = json.loads(a.readline()), json.loads(b.readline())
                require(ra["iteration"] == rb["iteration"] == iteration and
                        all(ra[k] == rb[k] for k in ("frame_id", "time_shift", "learning_rates")),
                        "实际paired_steps不一致: " + str(iteration))
            require(not a.readline() and not b.readline(), "paired_steps超过30000")
    return {"status": "GATE_PASSED_RECHECKED", "gate_path": str(Path(gate_path).resolve()),
            "means": means, "frame_contract": "49 identical IDs/timestamps/W480H320",
            "restoration_verified": False, "gn11_event": gn11, "paired_steps_reloaded": verify_steps}


def disk_forecast(config, reference_event=None, actual_counts=None, remaining_nodes=3):
    """可审查的元数据预测，不是最大体积证明；不依赖固定12/48GiB门槛。"""
    degree = int(config["sh_degree"])
    scalars = 3 + 3 * (degree + 1) ** 2 + 1 + 3 + 4 + 1 + 1 + 3
    # 参数FP32+Adam两槽+4个point统计；SGCM静态谱系int64与spent bool另9B。
    dynamic_per_point = scalars * 12 + 16
    static_per_point = dynamic_per_point + 9
    if actual_counts is None:
        static = max(int(config["num_pts"]), int(config["densify_until_num_points"]))
        dynamic = max(1, int(config["dynamic_densify_until_num_points"]))
        static, dynamic = math.ceil(static * 1.1), math.ceil(dynamic * 1.1)
        count_basis = "configured caps with10%generic-topology overshoot forecast; not strict maximum"
    else:
        static, dynamic = map(int, actual_counts)
        count_basis = "actual current Gaussian counts before save"
    environment = 226494608
    measured = 0
    if reference_event:
        components = reference_event["checkpoints"]
        environment = int(components["environment"]["bytes"])
        measured = sum(int(v["bytes"]) for v in components.values())
    stat_bytes, dyn_bytes = static * static_per_point, dynamic * dynamic_per_point
    bundle = max(measured, stat_bytes + dyn_bytes + environment + 1048576)
    pixels = 480 * 320
    # NPZ未压缩上界：4RGB+3alpha/depth FP32+2bool；static-time4FP32。
    fp32 = 49 * pixels * (62 + 16)
    png_and_grids = 49 * pixels * 3 * (4 + 4)
    viewer = 49 * pixels * 3 * 6
    exports = (fp32 + png_and_grids + viewer) * remaining_nodes
    logs = 30000 * 4096
    atomic_component = max(stat_bytes, dyn_bytes, environment)
    subtotal = bundle * remaining_nodes + exports + logs + atomic_component
    reserve = math.ceil(subtotal * .1)
    return {"required_free_bytes": subtotal + reserve, "forecast_bundle_bytes": bundle,
            "remaining_nodes": remaining_nodes, "checkpoint_bytes": bundle * remaining_nodes,
            "evaluation_and_viewer_bytes": exports, "logs_bytes": logs,
            "atomic_component_temporary_bytes": atomic_component, "forecast_headroom_bytes": reserve,
            "point_counts_basis": count_basis, "static_points_forecast": static,
            "dynamic_points_forecast": dynamic, "static_bytes_per_point": static_per_point,
            "dynamic_bytes_per_point": dynamic_per_point,
            "boundary": "forecast only; recheck before start/save; no deletion or machine change"}


def require_disk_space(path, forecast):
    disk = shutil.disk_usage(path)
    require(disk.free >= forecast["required_free_bytes"],
            "磁盘不足：free=%d required=%d；保留旧资产，不启动/写checkpoint" %
            (disk.free, forecast["required_free_bytes"]))
    return dict(forecast, actual_free_bytes=disk.free, status="DISK_FORECAST_PASS")


def completion_contract(output, config):
    """完成检查只读小JSON和stat/header，不加载模型或NPZ像素数组。"""
    spec = config["followup_scene_spec"]
    for step in NODES:
        event = read_json(Path(output) / "checkpoint_events" / ("iteration_%06d.json" % step))
        require(event.get("status") == "complete" and event.get("iteration") == step, "complete event缺失")
        require(set(event["checkpoints"]) == {"static", "dynamic", "environment", "training_state"},
                "4component bundle缺失")
        for row in event["checkpoints"].values():
            path = Path(row["path"])
            require(path.is_file() and path.stat().st_size == row["bytes"] > 0, "checkpoint stat不匹配")
        frames = event["frames"]
        require([r["frame_id"] for r in frames] == spec["frame_ids_with_camera"], "完成frame IDs不匹配")
        require(all(abs(r["timestamp"] - t) < 1e-7 for r, t in zip(frames, spec["model_timestamps"])),
                "完成timestamp不匹配")
        for row in frames:
            require(set(row["channels"]) == {"gt", "composite", "static", "dynamic"}, "四PNG通道不全")
            for path in row["channels"].values():
                require(png_wh(path) == [480, 320], "完成PNG dimensions不匹配")
            metric = read_json(row["hybrid_metrics"])
            path = Path(metric["arrays"])
            require(metric["frame_id"] == row["frame_id"] and path.is_file() and path.stat().st_size > 0,
                    "rawFP32 NPZ或对应JSON缺失")
            raw = npz_headers(path)
            require(set(raw) == {"gt.npy", "composite.npy", "static.npy", "dynamic.npy", "alpha_d.npy",
                                "alpha_s.npy", "depth_accum_model.npy", "actor_mask.npy", "sky_mask.npy"},
                    "rawNPZ schema缺失")
            for name, header in raw.items():
                channels = 3 if name in ("gt.npy", "composite.npy", "static.npy", "dynamic.npy") else 1
                require(header["shape"] == [channels, 320, 480]
                        and header["dtype"] == ("|b1" if name in ("actor_mask.npy", "sky_mask.npy") else "<f4"),
                        "rawNPZ dtype/shape合同不符")
        summary = read_json(Path(output) / ("static_time_%d" % step) / "summary.json")
        require(summary["anchor_pose_frame"] == spec["anchor_frame"]
                and summary["anchor_camera_directory"] == spec["camera_directory"], "完成anchor/camera不匹配")
        rows = summary["rows"]
        require([r["source_timestamp_frame"] for r in rows] == spec["frame_ids"] and
                all(abs(r["timestamp_model"] - t) < 1e-7 for r, t in zip(rows, spec["model_timestamps"])),
                "静态时间49timestamp不完整")
        require(all((Path(output) / ("static_time_%d" % step) / ("%03d.npz" % i)).is_file()
                    for i in range(49)), "静态时间NPZ不完整")
        for i in range(49):
            headers = npz_headers(Path(output) / ("static_time_%d" % step) / ("%03d.npz" % i))
            require(headers == {"static_rgb.npy": {"dtype": "<f4", "shape": [3, 320, 480]},
                                "static_alpha.npy": {"dtype": "<f4", "shape": [1, 320, 480]}},
                    "静态时间NPZ FP32 header合同不符")
    return {"status": "FOLLOWUP_COMPLETE_CONTRACT_PASS", "scene_id": spec["scene_id"],
            "nodes": list(NODES), "frames_each": 49, "restoration_verified": False}


def npz_headers(path):
    """只解压小型NPY header，不加载FP32像素数组。"""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                require(stream.read(6) == b"\x93NUMPY", "NPY magic错误")
                version = tuple(stream.read(2))
                require(version in ((1, 0), (2, 0), (3, 0)), "NPY version不支持")
                width = 2 if version == (1, 0) else 4
                size = int.from_bytes(stream.read(width), "little")
                require(0 < size <= 100000, "NPY header size不合理")
                header = ast.literal_eval(stream.read(size).decode("utf-8" if version == (3, 0) else "latin1").strip())
                result[name] = {"dtype": header["descr"], "shape": list(header["shape"])}
    return result
