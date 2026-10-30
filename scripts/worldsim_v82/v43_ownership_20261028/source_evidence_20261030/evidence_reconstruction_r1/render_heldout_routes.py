"""一个独立留出相机的原生none渲染；查询标定可用，GT仅在渲染后评分。

保持evaluate_edits.py稳定，复用其文件/图像/区域评分函数、相同原生
model.restore(training_args=None)和现有prepare_rasterization/raster helpers。
单个留出视角只验证接口和一次新视角重建，不声称实景或多视角泛化。
"""
import argparse
import gc
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reconstruction'))
sys.path.insert(0, '/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/code/reconstruction')
from evaluate_edits import HELPER, normalized_id, read, region_error, rgb, save_rgb, write
from native_worker import SOURCE, assert_method
from shared_gauge import install
install(Path(__file__).resolve().with_name("baseline_gauge.json"), SOURCE)


def rigid_model_camera(world_c2w, transform):
    """world→model含统一scale；相机旋转必须去掉scale，平移仍按完整T变换。"""
    world_c2w = np.asarray(world_c2w, dtype=np.float64).reshape(4, 4)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    linear = transform[:3, :3]
    scale = float(np.linalg.norm(linear[:, 0]))
    if (not np.isfinite(world_c2w).all() or not np.isfinite(transform).all() or scale <= 0
            or not np.allclose(linear.T @ linear, np.eye(3) * scale * scale, atol=1e-8)
            or not np.allclose(world_c2w[:3, :3].T @ world_c2w[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(world_c2w[:3, :3]) < .999):
        raise ValueError("需要刚体OpenCV相机和有限各向同性world→model transform")
    result = np.eye(4)
    result[:3, :3] = linear @ world_c2w[:3, :3] / scale
    result[:3, 3] = linear @ world_c2w[:3, 3] + transform[:3, 3]
    return result, scale


def prepare_query(event, reference_root, view_id):
    config = event["config"]
    assert_method(config)
    if event["status"] != "complete":
        raise ValueError("只读取完整原生checkpoint event")
    manifest = read(reference_root / "reference_manifest.json")
    if manifest.get("schema") != "v43_sim_reference_v1":
        raise ValueError("留出参考schema与当前仿真不符")
    selected = [row for row in manifest.get("heldout", []) if row["view_id"] == view_id]
    if len(selected) != 1:
        raise ValueError("需要恰好一个明确声明的留出query camera")
    heldout = selected[0]
    if heldout.get("train_input") is not False:
        raise ValueError("留出相机必须显式标注train_input=false")
    frame = normalized_id(heldout["frame_id"])
    scene = Path(config["source_path"])
    prepared = read(scene / "preparation_manifest.json")
    spec = config["followup_scene_spec"]
    if frame not in spec["frame_ids"]:
        raise ValueError("留出相机的原时间帧不在登记的训练时间轴内")
    timestamp = float(spec["model_timestamps"][spec["frame_ids"].index(frame)])
    observations = read(Path(prepared["observations"]) / "manifest.json")
    source_rows = [row for row in observations["frames"] if normalized_id(row["id"]) == frame]
    if len(source_rows) != 1:
        raise ValueError("无法唯一核对留出相机对应的原始观测标定")
    source = source_rows[0]
    world_c2w = np.asarray(heldout["camera_to_world"], dtype=np.float64)
    if not np.allclose(world_c2w @ np.asarray(heldout["world_to_camera"], dtype=np.float64), np.eye(4), atol=1e-6):
        raise ValueError("留出camera_to_world/world_to_camera不互逆")
    original_world_c2w = np.asarray(source["camera_to_world"], dtype=np.float64)
    delta = world_c2w[:3, 3] - original_world_c2w[:3, 3]
    declared_delta = np.asarray(heldout["world_camera_translation_delta_m"], dtype=np.float64)
    if not np.allclose(declared_delta, [0, 1, 0], atol=1e-8) or not np.allclose(delta, declared_delta, atol=1e-5):
        raise ValueError("本入口仅接受已声明的原中帧worldY+1m留出相机")
    intrinsic = np.asarray(heldout.get("intrinsics", heldout.get("K")), dtype=np.float64).reshape(3, 3)
    width, height = int(heldout["width"]), int(heldout["height"])
    if (width < 1 or height < 1 or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("留出相机尺寸或K无效")
    if (width != int(source["width"]) or height != int(source["height"])
            or not np.allclose(intrinsic, np.asarray(source["intrinsics"]), atol=1e-6)):
        raise ValueError("该留出协议保持原中帧的尺寸和K，只改变相机位姿")
    transform = np.loadtxt(scene / "transform_matrix.txt")
    model_c2w, scale = rigid_model_camera(world_c2w, transform)
    original_model_c2w, _ = rigid_model_camera(original_world_c2w, transform)
    recorded = [row for row in read(Path(event["model_path"]) / "cameras.json") if row["img_name"] == frame]
    if len(recorded) != 1:
        raise ValueError("原训练cameras.json没有唯一对应时间帧")
    original_saved_c2w = np.eye(4)
    original_saved_c2w[:3, :3], original_saved_c2w[:3, 3] = recorded[0]["rotation"], recorded[0]["position"]
    original_error = float(np.max(np.abs(original_saved_c2w - original_model_c2w)))
    if original_error > 1e-5:
        raise ValueError("world→model映射未能复现原训练相机；不能继续渲染留出query")
    evidence = dict(frame=frame, view_id=view_id, train_input=False, training_updates=0,
        K=intrinsic.tolist(), width=width, height=height,
        camera_to_world=world_c2w.tolist(), camera_to_model=model_c2w.tolist(),
        world_to_model=transform.tolist(), meter_to_model_scale=scale,
        camera_world_translation_delta_m=delta.tolist(), original_camera_to_world=original_world_c2w.tolist(),
        camera_rotation_changed_after_reaiming=not np.allclose(world_c2w[:3, :3], original_world_c2w[:3, :3], atol=1e-7),
        model_timestamp=timestamp, simulator_timestamp_s=heldout.get("timestamp_s"),
        original_training_camera_restore_max_abs=original_error,
        camera_convention="OpenCV x-right y-down z-forward; corner K; rays through pixel centers +0.5; rigid rotation removes normalization scale",
        query_input_fields=["camera_to_world", "intrinsics", "width", "height", "frame_id"],
        ground_truth_rgb_or_geometry_used_for_query=False,
        scope="one heldout camera, known observed time; no real-world or multi-view generalization claim")
    return evidence, heldout


def score_reference(reference_root, heldout, prediction, output):
    # 只有前向渲染完成后才调用；GT图像不成为Camera.image或raster输入。
    target = rgb(reference_root / heldout["rgb"])
    if target.shape != prediction.shape:
        raise ValueError("留出参考RGB与原生输出尺寸不一致")
    with np.load(reference_root / heldout["geometry"], allow_pickle=False) as geometry:
        semantic = geometry["semantic_id"].copy()
    if semantic.shape != target.shape[1:] or not set(np.unique(semantic)).issubset({0, 1, 2, 3, 10}):
        raise ValueError("独立留出first-hit分区格式不符")
    masks = dict(full=np.ones_like(semantic, dtype=bool), body_reference=semantic == 10,
                 road_reference=np.isin(semantic, [1, 2]), roadside=semantic == 3,
                 environment_or_no_hit=semantic == 0)
    shutil.copy2(reference_root / heldout["rgb"], output / "reference_rgb.png")
    result = dict(status="SCORED_ONE_INDEPENDENT_HELDOUT_VIEW", heldout_views=1,
        regions={name: region_error(prediction, target, mask) for name, mask in masks.items()},
        reference_rgb=str((reference_root / heldout["rgb"]).resolve()),
        saved_reference_rgb=str((output / "reference_rgb.png").resolve()),
        body_and_road="independent first-hit surface regions, not Gaussian GT ownership",
        shadow_effect_contrast_proxy_available=False,
        shadow_scope="this heldout camera has one shadow-on reference only; no shadow mask or on/off contrast is inferred",
        ground_truth_read_after_forward_render=True, ground_truth_used_for_training=False,
        quality_accepted=False, human_verdict=None,
        scope="one known-time heldout-camera diagnostic; no generalization claim")
    write(output / "reference_metrics.json", result)
    return result


def render(args, event, query):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(args.source))
    from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
    from scene.envlight import EnvLight
    from scene.cameras import Camera
    from gaussian_renderer import prepare_rasterization, render as actual_native_render
    from utils.general_utils import seed_everything
    if Path(sys.modules["scene"].__file__).resolve() != args.source / "scene" / "__init__.py":
        raise RuntimeError("未加载当前真实DrivingEditor模型与Camera")
    if not torch.cuda.is_available():
        raise RuntimeError("真实留出渲染需要CUDA，不执行CPU替代渲染")
    config = event["config"]
    pipe = OmegaConf.create(config)
    if pipe.compute_cov3D_python or pipe.convert_SHs_python or pipe.white_background:
        raise ValueError("留出入口保留已核对的原生quaternion/SH/black-background配置")
    paths = [Path(row["path"]) for row in event["checkpoints"].values()]
    before_files = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    for row in event["checkpoints"].values():
        if Path(row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("完整checkpoint event文件长度不符")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    seed_everything(config["seed"])
    models = dict(static=GaussianModel(pipe), dynamic=Dynamic_GaussianModel(pipe),
                  environment=EnvLight(resolution=pipe.env_map_res).cuda())
    for name, model in models.items():
        payload, step = torch.load(event["checkpoints"][name]["path"], map_location="cpu", mmap=True, weights_only=False)
        if step != event["iteration"]:
            raise ValueError("模型checkpoint与event步数不符")
        payload = list(payload)
        if name == "environment":
            payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
        else:
            for index in range(1, 11):
                payload[index] = payload[index].detach().cuda()
            model.time_duration = tuple(config["time_duration"])
        # 与evaluate_edits一致：原生restore，training_args=None不创建optimizer。
        model.restore(tuple(payload), training_args=None)
        if getattr(model, "optimizer", None) is not None:
            raise RuntimeError("留出评价不得创建optimizer或执行更新")
        for value in vars(model).values():
            if torch.is_tensor(value):
                value.requires_grad_(False)
        del payload
        gc.collect()
    continuation = torch.load(event["checkpoints"]["training_state"]["path"], map_location="cpu", weights_only=False)
    for name, attributes in continuation["model_runtime"].items():
        for key, value in attributes.items():
            setattr(models[name], key, value)
    model_w2c = np.linalg.inv(np.asarray(query["camera_to_model"]))
    intrinsic = np.asarray(query["K"])
    camera = Camera(colmap_id=0, R=model_w2c[:3, :3].T.copy(), T=model_w2c[:3, 3].copy(),
        fx=float(intrinsic[0, 0]), fy=float(intrinsic[1, 1]), cx=float(intrinsic[0, 2]), cy=float(intrinsic[1, 2]),
        image=torch.zeros(3, query["height"], query["width"]), image_name=query["view_id"] + "_" + query["frame"],
        resolution=(query["width"], query["height"]), timestamp=query["model_timestamp"])
    camera_error = float(np.max(np.abs(camera.c2w.detach().cpu().numpy() - np.asarray(query["camera_to_model"]))))
    if camera_error > 1e-5:
        raise ValueError("真实Camera构造没有复现留出query外参")
    spec = importlib.util.spec_from_file_location("v43_heldout_native_raster", args.raster_helper)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    def versions():
        return {(name, key): (value._version, tuple(value.shape)) for name, model in models.items()
                for key, value in vars(model).items() if torch.is_tensor(value)}

    before_versions = versions()
    with torch.no_grad():
        values = []
        for name in ("static", "dynamic"):
            kwargs = prepare_rasterization(camera, models[name], pipe, dynamic=name == "dynamic")
            kwargs["means2D"] = torch.zeros_like(kwargs["means3D"])
            values.append(kwargs)
        from scipy.spatial.transform import Rotation
        static_rows,dynamic_rows=values
        selected=torch.tensor(np.load(args.selection)["selected"],device="cuda",dtype=torch.bool)
        if args.baseline: selected=torch.zeros(len(static_rows["means3D"]), device="cuda", dtype=torch.bool)
        actor={k:(None if v is None else v[selected]) for k,v in static_rows.items()}
        actor=helpers.merge_rows(actor,dynamic_rows)
        kept={k:(None if v is None else v[~selected]) for k,v in static_rows.items()}
        if args.edit=="none":
            values=[static_rows,dynamic_rows]
        elif args.edit=="delete":
            values=[kept,{k:(None if v is None else v[:0]) for k,v in actor.items()}]
        else:
            transform=np.asarray(query["world_to_model"]); linear=transform[:3,:3]
            if args.edit=="shift": actor["means3D"]=actor["means3D"]+torch.tensor(linear@np.array([0.,1.5,0.]),device="cuda",dtype=torch.float32)
            else:
                tracks=read(Path(config["source_path"])/"ideal_tracks.json")["tracks"]
                track=next(x for x in tracks if normalized_id(x["frame_id"])==query["frame"])
                center_world=np.asarray(track["object_to_world"])[:3,3]
                center=torch.tensor(linear@center_world+transform[:3,3],device="cuda",dtype=torch.float32)
                rotation_np=linear@Rotation.from_euler("z",30,degrees=True).as_matrix()@np.linalg.inv(linear)
                rotation=torch.tensor(rotation_np,device="cuda",dtype=torch.float32)
                actor["means3D"]=(actor["means3D"]-center)@rotation.T+center
                q=torch.tensor(np.roll(Rotation.from_matrix(rotation_np).as_quat(),1),device="cuda",dtype=torch.float32)
                aw,ax,ay,az=q.unbind(-1);bw,bx,by,bz=actor["rotations"].unbind(-1)
                rq=torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)
                actor["rotations"]=rq/rq.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            depth=torch.cat((actor["means3D"],torch.ones_like(actor["means3D"][:,:1])),1)@camera.world_view_transform[:,2:3]
            actor["features"]=torch.cat((depth,torch.ones_like(depth)),1)
            values=[kept,actor]
        background = torch.zeros(3, device="cuda")
        package = helpers.raster(camera, pipe, models["static"].active_sh_degree,
                                 helpers.merge_rows(*values), background)
        color, alpha = package["rgb"], package["alpha"]
        environment = models["environment"](camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        fill = (1 - alpha) * environment
        final, white = color + fill, color + (1 - alpha)
        native = actual_native_render(camera, models["static"], models["dynamic"], pipe, background,
                                      env_map=models["environment"])["render"]
        identity_error = float((native - final).abs().max())
        if args.edit=="none" and identity_error > 2e-6:
            raise RuntimeError("留出none未复现真实原生渲染，保持原2e-6阈值: " + str(identity_error))
        arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in
                  (("C", color), ("alpha", alpha), ("E", environment), ("fill", fill),
                   ("final", final), ("white", white), ("depth_sum_model", package["depth_sum"]))}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("留出C/alpha/E/fill输出非有限")
    fp32 = args.output / ("heldout_"+args.edit+".npz")
    np.savez_compressed(fp32, **arrays)
    for name in ("C", "E", "fill", "final", "white"):
        save_rgb(args.output / (name + ".png"), arrays[name])
    from PIL import Image
    Image.fromarray(np.round(arrays["alpha"][0].clip(0, 1) * 255).astype("uint8")).save(args.output / "alpha.png")
    if before_versions != versions() or before_files != {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}:
        raise RuntimeError("留出评价修改了checkpoint文件或模型tensor")
    result = dict(status="NATIVE_SINGLE_HELDOUT_VIEW_RENDERED", checkpoint_event=str(args.event), iteration=event["iteration"],
        heldout_views=1, edit=args.edit, baseline_dynamic_only=args.baseline, frame=query["frame"], view_id=query["view_id"], fp32=str(fp32.resolve()),
        query_camera=str(args.output / "query_camera.json"), actual_camera_reconstruction_max_abs=camera_error,
        native_none_identity_max_abs=identity_error if args.edit == "none" else None, selection="all static kept; original learned dynamic only" if args.baseline else str(args.selection), native_none_comparison_not_applicable=args.edit != "none", original_checkpoint_unchanged=True, training_updates=0,
        compositor="native premultiplied C; fill=(1-alpha)*E; final=C+fill; white=C+(1-alpha)",
        required_fp32_arrays=list(arrays), camera_image_input="all-zero placeholder; no reference RGB",
        ground_truth_rgb_or_geometry_used_for_forward=False,
        environment=dict(min=float(arrays["E"].min()), max=float(arrays["E"].max()),
            channel_fraction_outside_0_1=float(((arrays["E"] < 0) | (arrays["E"] > 1)).mean())),
        environment_fill=dict(min=float(arrays["fill"].min()), max=float(arrays["fill"].max()),
            channel_fraction_outside_0_1=float(((arrays["fill"] < 0) | (arrays["fill"] > 1)).mean())),
        alpha=dict(min=float(arrays["alpha"].min()), max=float(arrays["alpha"].max()),
            fraction_outside_0_1=float(((arrays["alpha"] < 0) | (arrays["alpha"] > 1)).mean())),
        compositor_checks=dict(final_vs_C_plus_fill_max_abs=float(np.max(np.abs(arrays["final"] - (arrays["C"] + arrays["fill"])))),
            fill_vs_one_minus_alpha_times_E_max_abs=float(np.max(np.abs(arrays["fill"] - (1-arrays["alpha"])*arrays["E"])))),
        scope="one known-time heldout-camera diagnostic; not real-world or multi-view generalization",
        quality_accepted=False, human_verdict=None)
    write(args.output / "render_manifest.json", result)
    return arrays["final"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit",choices=["none","delete","shift","rotate"],default="delete")
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--raster-helper", type=Path, default=HELPER)
    parser.add_argument("--view-id", default="lateral_y_plus_1m")
    parser.add_argument("--query-only", action="store_true", help="仅CPU验证并保存query标定，不读参考RGB/geometry，不导入Torch")
    args = parser.parse_args()
    for key in ("event", "reference_root", "output", "source", "raster_helper"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        raise FileExistsError("不覆盖旧留出结果，请使用新目录")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    event = read(args.event)
    query, heldout = prepare_query(event, args.reference_root, args.view_id)
    write(args.output / "query_camera.json", query)
    if args.query_only:
        status = "HELDOUT_QUERY_CAMERA_VALIDATED_CPU_ONLY"
    else:
        prediction = render(args, event, query)
        status = "NATIVE_FIXED_SOURCE_SELECTION_HELDOUT_DELETE_RENDERED_NO_EDIT_REFERENCE"
        write(args.output / "reference_metrics.json", dict(status="NOT_SCORED",reason="Available heldout RGB is none only; no independent delete reference; do not compare deletion to none",heldout_geometry_used_for_selection=False))
    write(args.output / "execution.json", dict(status=status, wall_seconds=time.monotonic() - started,
        query_only=args.query_only, training_updates=0, heldout_views=1, generalization_claim=False))
    print(status + " " + str(args.output))


if __name__ == "__main__":
    main()
