"""阶段2：冻结覆盖后，通过真实道路光栅梯度修正连续世界图集的局部颜色。"""
from pathlib import Path
import argparse
import json
import os
import random
import time
import hashlib
import shutil
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "2")
import numpy as np
import torch
import torch.nn.functional as F
from .road_asset import WorldRoadMaterialAtlas, srgb_to_linear, linear_to_srgb


FRAMES = [86, 110, 134, 179, 198]


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def torch_srgb(linear):
    linear = linear.clamp(0, 1)
    return torch.where(linear <= .0031308, linear * 12.92, 1.055 * linear.clamp_min(1e-9).pow(1 / 2.4) - .055)


class LocalWorldTextureCorrection(torch.nn.Module):
    """只训练世界图集delta；known与保留回退节点在参数化中严格为零。"""
    def __init__(self, atlas, eligible, device="cuda", maximum_linear_delta=.08):
        super().__init__()
        self.maximum_linear_delta = float(maximum_linear_delta)
        self.register_buffer("base_linear", torch.as_tensor(srgb_to_linear(atlas.rgb_srgb).transpose(2, 0, 1)[None].copy(), device=device))
        self.register_buffer("eligible", torch.as_tensor(eligible[atlas.grid_world_row][None, None], device=device))
        self.delta_raw = torch.nn.Parameter(torch.zeros_like(self.base_linear))

    def delta(self):
        return self.maximum_linear_delta * self.delta_raw.tanh() * self.eligible

    def sample(self, uv):
        height, width = self.base_linear.shape[-2:]
        scale = uv.new_tensor([width - 1, height - 1])
        grid = (uv / scale * 2 - 1)[None, None]
        field = (self.base_linear + self.delta()).clamp(0, 1)
        # 宽图集归一化坐标用float64，避免暗/亮邻接节点的float32半像素舍入。
        # 双线性数学不变；进入原生颜色shader前仍回到其float32约定。
        sampled = F.grid_sample(field.to(uv.dtype), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, 0].T.to(self.base_linear.dtype)
        return torch_srgb(sampled)


def frozen_contract(manifest, asset, supervision):
    return dict(schema="module2_texture_training_contract.v1", phase=2, formal_steps=1200, seed=101,
        manifest=str(Path(manifest).resolve()), bootstrap_asset=str(Path(asset).resolve()), supervision=str(Path(supervision).resolve()),
        frames=FRAMES, sampler="uniform shuffled epochs over five fixed frames; Random(101)",
        trainable="bounded local linear RGB atlas delta at requested active supported nonobserved nonguarded texels only",
        frozen=["road xyz/z/normals/scale/opacity/active", "observed texture", "persistent retained fallback", "all native/static/vehicle/environment/camera parameters"],
        optimizer=dict(name="Adam", lr=.02, betas=[.9, .999], eps=1e-8), maximum_linear_delta=.08,
        coordinate_precision="float64 normalized UV and bilinear interpolation; float32 linear texels/delta/shader",
        loss=dict(photo="confidence weighted L1(sRGB road-only RGB/alpha, immutable pretrain atlas ray projection)",
            valid="fixed legal_color & alpha>.8; all frozen evaluation pixels still reported separately",
            magnitude=.02, spatial_tv=.01), reference_role="self-consistency appearance prior, not hidden road truth",
        new_diffusion_updates=0, new_vehicle_updates=0, new_shadow_updates=0)


def differences(left, right, path=""):
    if torch.is_tensor(left):
        return [] if torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu()) else [path]
    if isinstance(left, np.ndarray):
        return [] if isinstance(right, np.ndarray) and np.array_equal(left, right) else [path]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [path]
        return [item for key in left for item in differences(left[key], right[key], path + "/" + str(key))]
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return [path]
        return [item for i, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, path + "/" + str(i))]
    return [] if left == right else [path]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def setup(manifest_path, proof_path, asset_dir, supervision_dir):
    from .persistent_road_handoff import load_persistent_handoff
    torch.set_num_threads(2)
    scene, manifest, loader_receipt = load_persistent_handoff(manifest_path, proof_path)
    for model in [scene.static, scene.dynamic, scene.environment]:
        for parameter in model.parameters() if hasattr(model, "parameters") else []:
            parameter.requires_grad_(False)
    base = load(manifest.road)
    if Path(manifest.road).resolve() != (Path(asset_dir)/"road.npz").resolve():
        raise ValueError("阶段2场景road必须就是给定已接受阶段1资产的road.npz")
    provenance = load(Path(asset_dir) / "provenance.npz")
    atlas = WorldRoadMaterialAtlas.load(Path(asset_dir) / "atlas.npz")
    frozen=json.loads((Path(supervision_dir)/"freeze.json").read_text())
    if frozen.get("schema")!="immutable_road_training_supervision.v1":
        raise ValueError("需要显式的训练前监督freeze.json")
    for item in frozen["source_files"]+frozen["target_files"]:
        if file_digest(item["path"])!=item["sha256"]:
            raise ValueError("训练前固定source/target内容已改变: "+item["path"])
    reference_atlas_path=next(item["path"] for item in frozen["source_files"] if Path(item["path"]).name=="atlas.npz")
    reference=load(reference_atlas_path)
    current=load(Path(asset_dir)/"atlas.npz")
    if set(current)!=set(reference) or not all(np.array_equal(current[key],reference[key],equal_nan=True) for key in current):
        raise ValueError("阶段1接续资产必须保留与监督对应的原固定F；覆盖可变而base atlas不可变")
    guard = provenance.get("guarded_world_rows", np.zeros(len(base["active"]), bool))
    eligible = provenance["requested"] & provenance["surface_supported"] & ~provenance["unknown"] & ~provenance["known"] & ~guard & base["active"]
    eligible &= provenance["material_updated"]
    model = LocalWorldTextureCorrection(atlas, eligible)
    active = base["active"]
    uv = torch.as_tensor(provenance["atlas_uv"][active].copy(), device="cuda", dtype=torch.float64)
    row_eligible = torch.as_tensor(eligible[active], device="cuda")
    base_rgb = torch.as_tensor(base["reference_srgb"][active].copy(), device="cuda", dtype=torch.float32)
    provider = next(provider for provider in scene._providers if provider.kind == "road")
    targets = {}
    for key in FRAMES:
        data = load(Path(supervision_dir) / f"{key:03d}.npz")
        targets[key] = {name: torch.as_tensor(data[name].copy(), device="cuda") for name in
            ["target_rgb_srgb", "legal_color", "confidence", "K", "frozen_evaluation_domain", "full_target"]}
    return dict(scene=scene, manifest=manifest, loader=loader_receipt, base=base, provenance=provenance,
                atlas=atlas, eligible=eligible, model=model, provider=provider, uv=uv,
                row_eligible=row_eligible, base_rgb=base_rgb, targets=targets, asset_dir=Path(asset_dir).resolve())


def render(state, key):
    scene = state["scene"]
    camera = scene.camera_provider(key)
    rows = state["provider"].rows(key, camera, scene.pipe)
    sampled = state["model"].sample(state["uv"])
    colors = torch.where(state["row_eligible"][:, None], sampled, state["base_rgb"])
    rows = dict(rows, colors_precomp=colors)
    return scene._bindings().raster(camera, scene.pipe, 0, rows, scene._background)


def loss(state, key):
    out = render(state, key)
    target = state["targets"][key]
    alpha = out["alpha"][0]
    valid = target["legal_color"] & (alpha > .8) & ~target["K"]
    confidence = target["confidence"] * valid
    rgb = out["rgb"] / out["alpha"].clamp_min(.1)
    photo = ((rgb - target["target_rgb_srgb"]).abs() * confidence[None]).sum() / (3 * confidence.sum().clamp_min(1))
    delta = state["model"].delta()
    magnitude = delta.square().sum() / (3 * state["model"].eligible.sum().clamp_min(1))
    tv = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean() + (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    total = photo + .02 * magnitude + .01 * tv
    metrics = dict(photo=float(photo.detach()), magnitude=float(magnitude.detach()), tv=float(tv.detach()),
                   valid_pixels=int(valid.sum()), frozen_evaluation_pixels=int(target["frozen_evaluation_domain"].sum()),
                   full_target_pixels=int(target["full_target"].sum()),
                   frozen_domain_alpha_mean=float(alpha[target["frozen_evaluation_domain"]].mean()))
    return total, metrics


def evaluate(state):
    with torch.no_grad():
        rows = []
        for key in FRAMES:
            value, metrics = loss(state, key)
            rows.append(dict(raw=key, loss=float(value), **metrics))
    return dict(rows=rows, mean_loss=float(np.mean([row["loss"] for row in rows])))


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(101); torch.cuda.manual_seed_all(101); np.random.seed(101); random.seed(101)
    state = setup(args.manifest, args.proof, args.asset, args.supervision)
    contract = frozen_contract(args.manifest, args.asset, args.supervision)
    contract["input_binding"] = {str(Path(path).resolve()): file_digest(path) for path in
        [args.manifest, args.proof, Path(args.asset) / "atlas.npz", Path(args.asset) / "provenance.npz",
         Path(args.supervision) / "receipt.json",Path(args.supervision)/"freeze.json",Path(__file__).resolve(),
         *[Path(args.supervision) / f"{raw:03d}.npz" for raw in FRAMES]]}
    write(output / "frozen_contract.json", contract)
    model = state["model"]
    optimizer = torch.optim.Adam([model.delta_raw], lr=.02)
    sampler = random.Random(101)
    queue = []
    start = 0
    def snapshot(step):
        return dict(schema="module2_texture_state.v1", step=step, delta_raw=model.delta_raw.detach().clone(),
            optimizer=optimizer.state_dict(), sampler=sampler.getstate(), queue=queue.copy(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), contract=contract,
            formal_run=args.mode == "formal")
    restored = None
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.mode == "formal" and not restored.get("formal_run", False):
            raise ValueError("正式1200步不能从smoke参数继续")
        if restored["contract"] != contract:
            raise ValueError("恢复的固定输入/优化合同不一致")
        if (not isinstance(restored["step"], int) or not 0 <= restored["step"] <= args.steps
                or restored["delta_raw"].shape != model.delta_raw.shape
                or not bool(torch.isfinite(restored["delta_raw"]).all())):
            raise ValueError("恢复步数、颜色参数形状或数值无效")
        model.delta_raw.data.copy_(restored["delta_raw"])
        optimizer.load_state_dict(restored["optimizer"])
        sampler.setstate(restored["sampler"]); queue = restored["queue"].copy(); start = restored["step"]
        torch.set_rng_state(restored["torch_rng"]); torch.cuda.set_rng_state_all(restored["cuda_rng"])
        np.random.set_state(restored["numpy_rng"]); random.setstate(restored["python_rng"])
        mismatch = differences(restored, snapshot(start))
        if mismatch:
            raise RuntimeError("完整状态未精确恢复: " + str(mismatch))
        write(output / "restore.json", dict(full_state_restored=True, start_step=start,
            parameter_exact=True, optimizer_restored=True, RNG_and_sampler_restored=True, differences=mismatch))
    else:
        torch.save(snapshot(0), output / "checkpoint000000.pt")
    write(output / "evaluation_before.json", evaluate(state))
    # 梯度路径探针使用真实目标与真实光栅，不计优化步；不修改参数。
    optimizer.zero_grad(set_to_none=True)
    probe_loss, _ = loss(state, FRAMES[0]); probe_loss.backward()
    gradient = model.delta_raw.grad
    eligible_grid = model.eligible.expand_as(gradient)
    finite = bool(torch.isfinite(gradient).all())
    maximum = float(gradient.abs().max())
    outside = float(gradient.masked_select(~eligible_grid).abs().max())
    if not finite or maximum == 0 or outside != 0:
        raise RuntimeError("真实世界颜色光栅梯度资格失败")
    write(output / "gradient_probe.json", dict(real_raster_backward=True, finite=finite,
        eligible_gradient_max=maximum, frozen_texel_gradient_max=outside,
        eligible_texels=int(state["eligible"].sum()), observed_texels_trainable=0,
        trainable_parameter="local atlas delta_raw only", color_path="linear atlas + bounded delta -> grid_sample -> sRGB -> native road raster"))
    optimizer.zero_grad(set_to_none=True)
    records = []
    begin = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        if not queue:
            queue = FRAMES.copy(); sampler.shuffle(queue)
        key = queue.pop()
        optimizer.zero_grad(set_to_none=True)
        value, metrics = loss(state, key)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("阶段2 loss非有限")
        value.backward()
        if not bool(torch.isfinite(model.delta_raw.grad).all()):
            raise FloatingPointError("阶段2 gradient非有限")
        optimizer.step()
        record = dict(step=step, raw=key, loss=float(value.detach()), elapsed_seconds=time.monotonic() - begin, **metrics)
        records.append(record)
        if step == start + 1 or step % 25 == 0:
            print(json.dumps(record), flush=True)
            write(output / "progress.json", record)
        if step % 100 == 0 or step == args.steps:
            torch.save(snapshot(step), output / f"checkpoint{step:06d}.pt")
    state["scene"].assert_persistent_unchanged()
    write(output / "evaluation_after.json", evaluate(state))
    write(output / "training.json", records)
    export(state, output / "asset")
    write(output / "completed.json", dict(completed=True, steps=args.steps, actual_updates=args.steps - start,
        checkpoint=str(output / f"checkpoint{args.steps:06d}.pt"), phase=2,
        formal=args.mode == "formal", budget=1200, frozen_coverage_and_source_scene_unchanged=True,
        seconds=time.monotonic() - begin, new_diffusion_or_vehicle_updates=0))


def export(state, output):
    output.mkdir(parents=True, exist_ok=False)
    atlas = state["atlas"]
    delta = state["model"].delta().detach().cpu().numpy()[0].transpose(1, 2, 0)
    color = linear_to_srgb(srgb_to_linear(atlas.rgb_srgb) + delta).astype(np.float32)
    color[~state["eligible"][atlas.grid_world_row]] = atlas.rgb_srgb[~state["eligible"][atlas.grid_world_row]]
    corrected = WorldRoadMaterialAtlas(color, atlas.confidence, atlas.source, atlas.legal,
        atlas.surface_world_xyz, atlas.world_from_uv, atlas.grid_world_row, atlas.source_raw,
        atlas.source_world_row, atlas.source_distance_m)
    query = corrected.query(state["provenance"]["world_xyz"])
    road = {key: value.copy() for key, value in state["base"].items()}
    road["reference_srgb"][state["eligible"]] = query["rgb_srgb"][state["eligible"]]
    if not all(np.array_equal(road[key], state["base"][key]) for key in road if key != "reference_srgb"):
        raise RuntimeError("颜色导出改变了固定覆盖")
    provenance = {key: value.copy() for key, value in state["provenance"].items()}
    provenance["texture_correction_eligible"] = state["eligible"]
    provenance["base_row_rgb"] = provenance["row_rgb"].copy()
    provenance["row_rgb"] = query["rgb_srgb"]
    provenance["effective_row_rgb"] = road["reference_srgb"]
    provenance["effective_active"] = road["active"].copy()
    provenance["effective_sigma_m"] = road["gaussian_sigma_m"].copy()
    provenance["effective_model_xyz"] = road["model_xyz"].copy()
    provenance["effective_road_opacity"] = road["road_opacity"].copy()
    provenance["texture_delta_linear"] = np.zeros_like(road["reference_srgb"])
    flat_ids = atlas.grid_world_row.ravel()
    provenance["texture_delta_linear"][flat_ids] = delta.reshape(-1, 3)
    atlas.save(output / "base_atlas.npz")
    corrected.save(output / "atlas.npz")
    np.savez_compressed(output / "texture_correction.npz", delta_linear=delta,
        eligible=state["eligible"][atlas.grid_world_row], world_from_uv=atlas.world_from_uv,
        surface_world_xyz=atlas.surface_world_xyz, grid_world_row=atlas.grid_world_row)
    np.savez_compressed(output / "road.npz", **road)
    np.savez_compressed(output / "provenance.npz", **provenance)
    for name in ["measurement_anchors.npz", "markings.npz", "marking_structure.json", "coverage_request_diff.npz"]:
        path=state["asset_dir"]/name
        if path.exists():
            shutil.copy2(path,output/name)
    # 下一阶段将消费已烘焙的RGB表。真实光栅比较可发现GPU插值/CPU烘焙接线差。
    baked_colors=torch.as_tensor(road["reference_srgb"][road["active"]].copy(),device="cuda",dtype=torch.float32)
    checks=[]
    with torch.no_grad():
        shader_colors=torch.where(state["row_eligible"][:,None],state["model"].sample(state["uv"]),state["base_rgb"])
        row_error=float((shader_colors-baked_colors).abs().max())
        for key in FRAMES:
            expected=render(state,key)["rgb"]
            camera=state["scene"].camera_provider(key)
            packet=state["provider"].rows(key,camera,state["scene"].pipe)
            actual=state["scene"]._bindings().raster(camera,state["scene"].pipe,0,
                dict(packet,colors_precomp=baked_colors),state["scene"]._background)["rgb"]
            error=float((expected-actual).abs().max())
            checks.append(dict(raw=key,max_RGB_abs_difference=error,
                RGB8_max_difference=int((torch.round(expected*255)-torch.round(actual*255)).abs().max())))
    write(output/"bake_validation.json",dict(real_raster_checked=True,row_RGB_max_abs_difference=row_error,
        rows=checks,tolerance_float_RGB=1e-4,bitexact_claim=False))
    if row_error>1e-4 or max(row["max_RGB_abs_difference"] for row in checks)>1e-4:
        raise RuntimeError("已导出世界图集到道路shader的重载接线误差过大")
    write(output / "receipt.json", dict(schema="world_road_texture_correction.v1", phase=2,
        eligible_texels=int(state["eligible"].sum()), geometry_scale_opacity_bitexact=True,
        observed_and_guarded_RGB_bitexact=True, reference_role="immutable pretraining world-atlas self consistency",
        local_delta_bound_linear=.08, observed_correction_count=0, fixed_world_material=True,
        base_atlas="base_atlas.npz", effective_atlas="atlas.npz", correction="texture_correction.npz",
        correction_grid="same metric atlas nodes; linear-light delta; bilinear after addition and clamp; no latent resizing",
        source_phase1_road=str(state["manifest"].road), source_bootstrap_asset=str(state["asset_dir"]),
        correction_semantics="underlying observed/retrieved/generated source kept; learned delta separately recorded"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "proof", "asset", "supervision", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=["smoke", "restore", "replay", "formal"], default="smoke")
    args = parser.parse_args()
    if args.mode == "formal" and args.steps != 1200:
        raise ValueError("正式阶段2预算固定1200步；smoke及重放单列")
    train(args)


if __name__ == "__main__":
    main()
