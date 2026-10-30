"""Frozen M1 conditional static selection plus learned dynamic native edit.
Post-hoc scalar thresholds define diagnostic policies, never ideal ownership or GT.
"""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code' / 'reconstruction'))
from native_worker import SOURCE, assert_method, bind_scene_contract
from generic_source_audit import diagnose, configure

HELPER = Path("/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/engine/utils/v4_road_render.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_id(value):
    text = str(value)
    return "%06d" % int(text) if text.isdigit() else text


def save_rgb(path, array):
    image = np.round(np.clip(np.moveaxis(array, 0, -1), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def region_error(prediction, reference, mask):
    count = int(mask.sum())
    result = dict(pixels=count)
    if count:
        for name, image in (("raw", prediction), ("clamped", np.clip(prediction, 0, 1))):
            delta = image[:, mask].astype(np.float64) - reference[:, mask].astype(np.float64)
            mse = float(np.square(delta).mean())
            result[name + "_mae"] = float(np.abs(delta).mean())
            result[name + "_mse"] = mse
            result[name + "_psnr_db"] = -10 * math.log10(max(mse, 1e-12))
    return result


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255


def reference_records(root):
    manifest = root / "reference_manifest.json"
    if not manifest.is_file():
        return {}, None
    data = read(manifest)
    if data.get("schema") != "v43_sim_reference_v1":
        raise ValueError("独立参考schema与当前仿真接口不符")
    if (data.get("edits", {}).get("shift", {}).get("translation_world_m") != [0, 1.5, 0]
            or data.get("edits", {}).get("rotate", {}).get("yaw_delta_deg") != 30.0):
        raise ValueError("独立参考编辑参数与当前分支编辑控制不符")
    records = {}
    for row in data["references"]:
        if type(row["vehicle_shadow"]) is not bool:
            raise ValueError("独立参考vehicle_shadow必须是明确布尔值")
        key = (normalized_id(row["frame_id"]), row["edit"], row["vehicle_shadow"])
        if key in records:
            raise ValueError("独立参考出现重复frame/edit/shadow键")
        records[key] = row
    return records, data


def score_one(reference_root, records, frame, edit, prediction):
    key = (frame, edit, True)
    if key not in records:
        return dict(status="REFERENCE_NOT_YET_AVAILABLE", frame=frame, edit=edit)
    item = records[key]
    target = rgb(reference_root / item["rgb"])
    if target.shape != prediction.shape:
        raise ValueError("独立参考和真实重建图像尺寸不同")
    with np.load(reference_root / item["geometry"], allow_pickle=False) as data:
        semantic = data["semantic_id"].copy()
    with np.load(reference_root / "original" / (frame + ".npz"), allow_pickle=False) as data:
        original_semantic = data["semantic_id"].copy()
    if semantic.shape != target.shape[1:] or original_semantic.shape != semantic.shape:
        raise ValueError("独立first-hit表面和RGB参考尺寸不一致")
    if not set(np.unique(semantic)).issubset({0, 1, 2, 3, 10}):
        raise ValueError("独立参考semantic ID超出已声明映射")
    road = np.isin(semantic, [1, 2])
    original_road = np.isin(original_semantic, [1, 2])
    body = semantic == 10
    original_body = original_semantic == 10

    def shadow_region(operation, receiver):
        on, off = records.get((frame, operation, True)), records.get((frame, operation, False))
        if on is None or off is None:
            return None
        difference = np.max(np.abs(rgb(reference_root / on["rgb"]) - rgb(reference_root / off["rgb"])), axis=0)
        return receiver & (difference > 2 / 255)

    shadow = shadow_region(edit, road)
    old_shadow = shadow_region("none", original_road)
    masks = dict(full=np.ones_like(road), body_reference=body, original_body_footprint=original_body,
        road_reference=road, newly_exposed_road=road & original_body, environment_or_no_hit=semantic == 0,
        roadside=semantic == 3)
    if shadow is not None:
        masks["shadow_road_current_effect_contrast_proxy"] = shadow
        masks["road_outside_current_effect_contrast_proxy"] = road & ~shadow
    if old_shadow is not None:
        masks["shadow_road_original_effect_contrast_proxy"] = old_shadow & road
    return dict(status="SCORED_INDEPENDENT_REFERENCE", frame=frame, edit=edit,
        reference=str((reference_root / item["rgb"]).resolve()),
        regions={name: region_error(prediction, target, mask) for name, mask in masks.items()},
        body_and_road="independent first-hit semantic surfaces; not Gaussian labels",
        shadow_effect_contrast_proxy="road receiver pixel contrast of independent vehicle-shadow on/off RGB, max channel >2/255; finite Monte Carlo rendering at32 samples has not been shown converged; proxy can include sampling variation, excludes body self-shadow; not a GT shadow mask and never used for training",
        reference_only_used_for_evaluation=True)


def score_rendered(rendered, output, reference_root):
    manifest = read(rendered / "render_manifest.json")
    records, _ = reference_records(reference_root)
    rows = []
    for row in manifest["frames"]:
        with np.load(row["fp32"], allow_pickle=False) as data:
            prediction = data["final"].copy()
        rows.append(score_one(reference_root, records, row["frame"], row["edit"], prediction))
    result = dict(status="INDEPENDENT_REFERENCE_AUDIT", rendered=str(rendered), rows=rows,
        scored=sum(row["status"] == "SCORED_INDEPENDENT_REFERENCE" for row in rows),
        missing_reference_count=sum(row["status"] != "SCORED_INDEPENDENT_REFERENCE" for row in rows),
        routing="conditional static selection plus learned dynamic native edit; geometry policies and alphaT recall diagnostics are not owner GT",
        privileged_reference_used_for_training=False, quality_accepted=False, human_verdict=None)
    write(output / "reference_metrics.json", result)
    return result


def render(args):
    import torch
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(args.source))
    from scene import Scene
    from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
    from scene.envlight import EnvLight
    from gaussian_renderer import prepare_rasterization, render as actual_native_render
    from utils.general_utils import seed_everything
    if Path(sys.modules["scene"].__file__).resolve() != args.source / "scene" / "__init__.py":
        raise RuntimeError("未导入已核对的当前DrivingEditor Scene")
    if not torch.cuda.is_available():
        raise RuntimeError("原生编辑入口需要真实 CUDA 光栅，不使用 toy renderer")
    event = read(args.event)
    if event["status"] != "complete":
        raise ValueError("只能读取完整 checkpoint event")
    config = event["config"]
    assert_method(config)
    iteration = event["iteration"]
    scene_root = Path(config["source_path"])
    prepared = read(scene_root / "preparation_manifest.json")
    paths = [Path(row["path"]) for row in event["checkpoints"].values()]
    before_files = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    loader_output = args.output / "scene_loader"
    loader_output.mkdir()
    pipe = OmegaConf.create(dict(config, model_path=str(loader_output), start_checkpoint=None))
    if pipe.compute_cov3D_python or pipe.convert_SHs_python or pipe.white_background:
        raise ValueError("此入口使用当前已核对的原生 quaternion/SH/black-background 配置")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    seed_everything(config["seed"])
    static, dynamic = GaussianModel(pipe), Dynamic_GaussianModel(pipe)
    # 真实 Scene 只负责加载相机和世界归一化；推理直接恢复训练结果，不再创建随机/种子模型。
    initialize_static, initialize_dynamic = static.create_from_pcd, dynamic.initialize
    static.create_from_pcd = lambda *_: None
    dynamic.initialize = lambda *_: None
    binding_proxy = type("BindingProxy", (), {"Scene": Scene})
    bind_scene_contract(binding_proxy, dict(config, model_path=str(loader_output)))
    scene = binding_proxy.Scene(pipe, static, dynamic, shuffle=False)
    static.create_from_pcd, dynamic.initialize = initialize_static, initialize_dynamic
    environment = EnvLight(resolution=pipe.env_map_res).cuda()
    components = {"static": static, "dynamic": dynamic, "environment": environment}
    for name, model in components.items():
        payload, saved_step = torch.load(event["checkpoints"][name]["path"], map_location="cpu", mmap=True, weights_only=False)
        if saved_step != iteration:
            raise ValueError("模型 checkpoint step 与 event 不符")
        payload = list(payload)
        if name == "environment":
            payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
        else:
            for index in range(1, 11):
                payload[index] = payload[index].detach().cuda()
            model.time_duration = tuple(config["time_duration"])
        model.restore(tuple(payload), training_args=None)
        if getattr(model, "optimizer", None) is not None:
            raise RuntimeError("评价入口不创建 optimizer 或执行更新")
        for value in vars(model).values():
            if torch.is_tensor(value):
                value.requires_grad_(False)
        del payload
        gc.collect()
    continuation = torch.load(event["checkpoints"]["training_state"]["path"], map_location="cpu", weights_only=False)
    for name, attributes in continuation["model_runtime"].items():
        for key, value in attributes.items():
            setattr(components[name], key, value)
    scene.scale_index = continuation["scale_index"]
    spec = importlib.util.spec_from_file_location("v43_native_raster_helper", args.raster_helper)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    recorded_cameras = {row["img_name"]: row for row in read(Path(event["model_path"]) / "cameras.json")}
    if set(cameras) != set(recorded_cameras):
        raise ValueError("推理重载相机身份与原训练cameras.json不符")
    maximum_extrinsic_error, maximum_intrinsic_error = 0., 0.
    for name, camera in cameras.items():
        saved = recorded_cameras[name]
        saved_c2w = np.eye(4)
        saved_c2w[:3, :3], saved_c2w[:3, 3] = saved["rotation"], saved["position"]
        maximum_extrinsic_error = max(maximum_extrinsic_error,
            float(np.max(np.abs(camera.c2w.detach().cpu().numpy() - saved_c2w))))
        downsample_x, downsample_y = saved["width"] / camera.image_width, saved["height"] / camera.image_height
        maximum_intrinsic_error = max(maximum_intrinsic_error,
            abs(float(camera.fx) - saved["fx"] / downsample_x), abs(float(camera.fy) - saved["fy"] / downsample_y),
            abs(float(camera.cx) - saved["cx"] / downsample_x), abs(float(camera.cy) - saved["cy"] / downsample_y))
    if maximum_extrinsic_error > 1e-5 or maximum_intrinsic_error > 1e-5:
        raise ValueError("推理重载的外参或K与真实训练相机不符")
    write(args.output / "training_camera_restore_verification.json", dict(all_passed=True,
        cameras=len(cameras), maximum_extrinsic_abs_error=maximum_extrinsic_error,
        maximum_intrinsic_abs_error=maximum_intrinsic_error, reference=str(Path(event["model_path"]) / "cameras.json")))
    if args.keyframes == "auto":
        _, reference_manifest = reference_records(args.reference_root) if args.reference_root else ({}, None)
        if reference_manifest and reference_manifest["keyframes"]:
            keys = [normalized_id(value) for value in reference_manifest["keyframes"]]
        else:
            names = prepared["followup_scene_spec"]["frame_ids"]
            keys = [names[index] for index in sorted({0, len(names) // 2, len(names) - 1})]
    else:
        keys = [normalized_id(value) for value in args.keyframes.split(",")]
    if not set(keys).issubset(cameras):
        raise ValueError("评价关键帧不属于真实训练相机集合")
    track_rows = read(scene_root / "ideal_tracks.json")["tracks"]
    tracks = {normalized_id(row["frame_id"]): row for row in track_rows}
    transform = np.loadtxt(scene_root / "transform_matrix.txt")
    linear = transform[:3, :3]
    inverse_linear = np.linalg.inv(linear)
    scale = float(np.linalg.norm(linear[:, 0]))
    if not np.allclose(linear.T @ linear, np.eye(3) * scale * scale, atol=1e-8):
        raise ValueError("原生 world→model 不是有效 similarity")
    model_rotation = linear @ Rotation.from_euler("z", 30, degrees=True).as_matrix() @ inverse_linear
    base_background = torch.zeros(3, device="cuda")

    def versions():
        return {(name, key): (value._version, tuple(value.shape)) for name, model in components.items()
                for key, value in vars(model).items() if torch.is_tensor(value)}

    before_versions = versions()
    frames, identity_max, identity_frames = [], 0., []
    edits = args.edits.split(",")
    if not set(edits).issubset({"none", "delete", "shift", "rotate"}):
        raise ValueError("未知编辑")
    summaries = []
    with torch.no_grad():
        for frame in keys:
            camera = cameras[frame]
            static_kwargs = prepare_rasterization(camera, static, pipe)
            static_kwargs["means2D"] = torch.zeros_like(static_kwargs["means3D"])
            dynamic_kwargs = prepare_rasterization(camera, dynamic, pipe, dynamic=True)
            dynamic_kwargs["means2D"] = torch.zeros_like(dynamic_kwargs["means3D"])
            selected, summary = diagnose(static_kwargs, dynamic_kwargs, camera, pipe, static.active_sh_degree, helpers, environment, transform, args.reference_root, frame, args.output)
            summaries.append(summary)
            selected_rows = {k: (None if v is None else v[selected]) for k,v in static_kwargs.items()}
            kept_rows = {k: (None if v is None else v[~selected]) for k,v in static_kwargs.items()}
            edit_dynamic = helpers.merge_rows(selected_rows, dynamic_kwargs)
            env = environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            for edit in edits:
                kwargs = static_kwargs if edit == "none" else kept_rows
                if edit != "delete":
                    moving = dict(dynamic_kwargs if edit == "none" else edit_dynamic)
                    if edit == "shift":
                        delta = torch.tensor(linear @ np.array([0., 1.5, 0.]), dtype=torch.float32, device="cuda")
                        moving["means3D"] = moving["means3D"] + delta
                    elif edit == "rotate":
                        track = tracks[frame]
                        center_world = np.asarray(track["object_to_world"], dtype=float)[:3, 3]
                        center = torch.tensor(linear @ center_world + transform[:3, 3], dtype=torch.float32, device="cuda")
                        rotation = torch.tensor(model_rotation, dtype=torch.float32, device="cuda")
                        moving["means3D"] = (moving["means3D"] - center) @ rotation.T + center
                        q = torch.tensor(np.roll(Rotation.from_matrix(model_rotation).as_quat(), 1), dtype=torch.float32, device="cuda")
                        a, b = q, moving["rotations"]
                        aw, ax, ay, az = a.unbind(-1)
                        bw, bx, by, bz = b.unbind(-1)
                        rotation_q = torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                                                 aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)
                        moving["rotations"] = rotation_q / rotation_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    if edit in ("shift", "rotate"):
                        depth = torch.cat((moving["means3D"], torch.ones_like(moving["means3D"][:, :1])), 1) @ camera.world_view_transform[:, 2:3]
                        moving["features"] = torch.cat((depth, torch.ones_like(depth)), 1)
                    kwargs = helpers.merge_rows(kwargs, moving)
                package = helpers.raster(camera, pipe, static.active_sh_degree, kwargs, base_background)
                color, alpha = package["rgb"], package["alpha"]
                fill = (1 - alpha) * env
                final = color + fill
                white = color + (1 - alpha)
                if edit == "none":
                    native = actual_native_render(camera, static, dynamic, pipe, base_background, env_map=environment)["render"]
                    error = float((native - final).abs().max())
                    identity_max = max(identity_max, error)
                    identity_frames.append(frame)
                    if error > 2e-6:
                        raise RuntimeError("自定义合成未复现真实原生 none 渲染: " + str(error))
                arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in
                          (("C", color), ("alpha", alpha), ("E", env), ("fill", fill), ("final", final), ("white", white), ("depth_sum_model", package["depth_sum"]))}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise FloatingPointError("原生编辑 C/alpha/E 输出非有限")
                folder = args.output / edit
                folder.mkdir(exist_ok=True)
                fp32 = folder / (frame + ".npz")
                np.savez_compressed(fp32, **arrays)
                for name in ("C", "E", "fill", "final", "white"):
                    save_rgb(folder / (frame + "_" + name + ".png"), arrays[name])
                Image.fromarray(np.round(arrays["alpha"][0].clip(0, 1) * 255).astype("uint8")).save(folder / (frame + "_alpha.png"))
                frames.append(dict(frame=frame, edit=edit, fp32=str(fp32.resolve()),
                    environment=dict(min=float(arrays["E"].min()), max=float(arrays["E"].max()),
                        channel_fraction_outside_0_1=float(((arrays["E"] < 0) | (arrays["E"] > 1)).mean())),
                    environment_fill=dict(min=float(arrays["fill"].min()), max=float(arrays["fill"].max()),
                        channel_fraction_outside_0_1=float(((arrays["fill"] < 0) | (arrays["fill"] > 1)).mean())),
                    compositor_checks=dict(final_vs_C_plus_fill_max_abs=float(np.max(np.abs(arrays["final"] - (arrays["C"] + arrays["fill"])))),
                        fill_vs_one_minus_alpha_times_E_max_abs=float(np.max(np.abs(arrays["fill"] - (1-arrays["alpha"])*arrays["E"])))),
                    alpha=dict(min=float(arrays["alpha"].min()), max=float(arrays["alpha"].max()),
                        fraction_outside_0_1=float(((arrays["alpha"] < 0) | (arrays["alpha"] > 1)).mean())),
                    final_channel_fraction_outside_0_1=float(((arrays["final"] < 0) | (arrays["final"] > 1)).mean())))
    after_files = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    if before_files != after_files or before_versions != versions():
        raise RuntimeError("评价期间 checkpoint 文件或模型张量发生修改")
    write(args.output / "responsibility_summary.json", dict(policy=args.policy, frames=summaries, unique_gaussian_gt=False, training_updates=0))
    manifest = dict(status="NATIVE_BRANCH_ROUTING_EDIT_RENDERED", checkpoint_event=str(args.event), iteration=iteration,
        frames=frames, keyframes=keys, edits=edits, same_camera_for_all_edits=True,
        compositor="native premultiplied C; fill=(1-alpha)*E; final=C+fill; white=C+(1-alpha)",
        required_fp32_arrays=["C", "alpha", "E", "fill", "final", "white", "depth_sum_model"],
        native_none_identity_max_abs=identity_max if identity_frames else None,
        native_none_identity_checked_frames=identity_frames,
        original_checkpoint_unchanged=True, training_updates=0,
        routing="truth-assisted conditional static selection plus entire learned dynamic branch; frozen representation, no unique Gaussian GT or mathematical upper bound",
        operation=dict(shift_world_m=[0, 1.5, 0], rotate_world_z_deg=30, rotation_center="legal ideal-track object root; same XY as body center",
                       rotate_SH_frame="inherited world-space SH; no relighting, no object-frame SH superiority claim"),
        privileged_reference_used_for_training=False, quality_accepted=False, human_verdict=None)
    write(args.output / "render_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config",type=Path,required=True)
    parser.add_argument("--shared-gauge",type=Path)
    parser.add_argument("--policy", choices=["strict","balanced","recall"], required=True)
    parser.add_argument("--event", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--raster-helper", type=Path, default=HELPER)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--keyframes", default="auto")
    parser.add_argument("--edits", default="none,delete,shift,rotate")
    parser.add_argument("--score-existing", type=Path, help="仅CPU读取已保存FP32并评价，不导入CUDA或改写渲染")
    args = parser.parse_args()
    if args.shared_gauge:
        sys.path.insert(0,str(args.shared_gauge.parent))
        from shared_gauge import install
        install(args.shared_gauge,SOURCE)
    configure(read(args.audit_config))
    for key in ("event", "output", "source", "raster_helper", "reference_root", "score_existing"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        raise FileExistsError("保留旧评价结果，使用新的输出目录")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    if args.score_existing:
        if not args.reference_root:
            parser.error("--score-existing 需要 --reference-root")
        result = score_rendered(args.score_existing, args.output, args.reference_root)
    else:
        if not args.event:
            parser.error("需要 --event 或 --score-existing")
        result = render(args)
        if args.reference_root:
            score_rendered(args.output, args.output, args.reference_root)
    write(args.output / "execution.json", dict(wall_seconds=time.monotonic() - started,
        mode="CPU reference score" if args.score_existing else "real native CUDA edit render", training_updates=0))
    print(json.dumps(dict(status=result["status"], output=str(args.output), training_updates=0)))


if __name__ == "__main__":
    main()
