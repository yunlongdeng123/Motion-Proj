"""二维道路教师诊断：默认只做 CPU 准备与预检，GPU 必须显式开启。

不接受真值文件。生成只读取已冻结的合法观察、输入掩码和提示词；
合成真值评价由独立评分程序在预测冻结之后执行。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
for _name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from PIL import Image

ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1")
REPOSITORY = Path("/root/autodl-tmp/motion_proj_v82_gn11_v42")
LEGACY = Path("/root/autodl-tmp/drivingeditor_repro/campaign/road_mask_decoupling_20261116_r1")
IDS = ["visible_scene081_raw000_v0", "visible_scene081_raw000_v1",
       "visible_scene081_raw050_v0", "visible_scene081_raw050_v1",
       "scene065_raw086", "scene065_raw110"]
SEEDS = [101, 202]
ARMS = ["full768_T0", "full768_base", "crop768_T0", "full1152_T0"]
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
SCHEMA = "road_2d_teacher_pilot.v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check_pin(record):
    assert sha(record["path"]) == record["sha256"], record["path"]


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).copy()


def binary(path):
    value = np.asarray(Image.open(path).convert("L"))
    assert np.isin(value, [0, 255]).all(), str(path)
    return value > 127


def roles(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def save_mask(path, value):
    Image.fromarray(np.asarray(value, np.uint8) * 255).save(path)


def fresh(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("保留已有结果，请指定新目录: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def inverse_view(model_rgb, native_hw, window):
    """严格复用历史的双线性逆变换；裁剪区外显式标无效。"""
    h, w = native_hw
    x0, y0, x1, y1 = window
    assert all(isinstance(value, int) for value in window)
    value = np.full((h, w, 3), 127, np.uint8)
    value[y0:y1, x0:x1] = np.asarray(Image.fromarray(model_rgb).resize(
        (x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
    valid = np.zeros((h, w), bool)
    valid[y0:y1, x0:x1] = True
    return value, valid


def known_metrics(predicted, reference, visible):
    """只评合法可见道路；固定原生网格 sigma=2.5px。"""
    from scipy.ndimage import gaussian_filter
    assert visible.any()
    p = predicted.astype(np.float32) / 255
    r = reference.astype(np.float32) / 255
    difference = (p - r)[visible]
    yweights = np.array([.2126, .7152, .0722], np.float32)
    py, ry = p @ yweights, r @ yweights
    ph = py - gaussian_filter(py, 2.5)
    rh = ry - gaussian_filter(ry, 2.5)
    return dict(pixels=int(visible.sum()), rgb_mae=float(np.abs(difference).mean()),
                signed_rgb_bias=difference.mean(0).tolist(),
                signed_luminance_bias=float((difference @ yweights).mean()),
                highpass_sigma_native_px=2.5,
                highpass_rms_reference=float(np.sqrt(np.mean(rh[visible] ** 2))),
                highpass_rms_prediction=float(np.sqrt(np.mean(ph[visible] ** 2))),
                highpass_rms_ratio=float(np.sqrt(np.mean(ph[visible] ** 2)) /
                                         max(np.sqrt(np.mean(rh[visible] ** 2)), 1e-12)))


def prepare(args):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    out = fresh(args.out)
    legacy = Path(args.legacy).resolve()
    old = read(legacy / "prepared/prepared.json")
    predictions = read(legacy / "predictions/predictions.json")
    assert old["complete"] and predictions["complete"]
    assert old["model_wh"] == [768, 512]
    assert (old["seed"], old["inference_steps"], old["guidance"], old["eta"]) == (101, 20, 3., 0.)
    check_pin(old["checkpoint"])
    assert old["checkpoint"]["sha256"] == "a75423e126b93eca22e0ce587782b3dc4edd76654d6a929302118aa45bda14ae"
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository))
    from scripts.worldsim_v82.road_material_field.input_construction.input_builder import render_view
    lookup = {row["id"]: row for row in old["cases"]}
    refs = {row["id"]: row["outputs"]["B"] for row in predictions["cases"]}
    prepared_cases, jobs, vae_controls = [], [], []
    source_files = {str(path): pin(path) for path in [
        repository / "scripts/worldsim_v82/road_material_field/input_construction/input_builder.py",
        repository / "scripts/worldsim_v82/road_material_field/input_construction/independent_regions.py",
        repository / "scripts/worldsim_v82/road_material_field/input_construction/shadow_composite.py"]}
    for sid in IDS:
        source = lookup[sid]
        b = source["views"]["B"]
        for record in [source["observation"], source["common_native"], *b.values()]:
            check_pin(record)
        observation = rgb(source["observation"]["path"])
        native = roles(b["roles_native"]["path"])
        common = roles(source["common_native"]["path"])
        shape = observation.shape[:2]
        assert shape == (1280, 1920)
        assert np.array_equal(native["H_input"], common["H_B"])
        assert np.array_equal(native["W_output"], common["W_output"])
        assert np.array_equal(native["K"], common["K"])
        assert not (native["W_output"] & ~native["H_input"]).any()
        assert not (native["W_output"] & native["K"]).any()
        assert not native["pending_output"].any()
        folder = out / sid
        folder.mkdir()
        Image.fromarray(observation).save(folder / "observation.png")
        np.savez_compressed(folder / "roles_native.npz", **native)
        for name in ["W_output", "K", "H_input", "C_mat"]:
            save_mask(folder / (name + "_native.png"), native[name])
        construction = read(b["construction"]["path"])
        layers = dict(rgb=observation, masks=native, confidence={}, depth_hint=None,
                      supervision_rgb=None, target_sanitized=None,
                      loss_mask=np.zeros(shape, bool), neutral_rgb=np.array([127] * 3, np.uint8),
                      receipt=construction["native"])
        plans = dict(full768=dict(output_wh=[768, 512], window_xyxy=[0, 0, 1920, 1280]),
                     crop768=dict(output_wh=[768, 272], window_xyxy=[0, 600, 1920, 1280]),
                     full1152=dict(output_wh=[1152, 768], window_xyxy=[0, 0, 1920, 1280]))
        views = {}
        for name, plan in plans.items():
            x0, y0, x1, y1 = plan["window_xyxy"]
            native_valid = np.zeros(shape, bool)
            native_valid[y0:y1, x0:x1] = True
            omitted = int((native["W_output"] & ~native_valid).sum())
            if omitted:
                views[name] = dict(available=False, reason="fixed crop does not contain the entire frozen W",
                                   missing_W_pixels=omitted, plan=plan)
                continue
            view = render_view(layers, plan)
            hidden = view["masks"]["H_input"]
            assert np.all(view["I_geo"][hidden] == 127)
            assert not (view["masks"]["W_output"] & view["masks"]["K"]).any()
            assert not (view["masks"]["W_output"] & ~hidden).any()
            target_folder = folder / name
            target_folder.mkdir()
            Image.fromarray(view["I_geo"]).save(target_folder / "I_geo.png")
            Image.fromarray(view["observation_rgb"]).save(target_folder / "VAE_reference_model.png")
            Image.fromarray(view["I_mat"]).save(target_folder / "I_mat_audit_only.png")
            save_mask(target_folder / "H_input.png", hidden)
            save_mask(target_folder / "C_mat.png", view["masks"]["C_mat"])
            np.savez_compressed(target_folder / "roles_model.npz", **view["masks"])
            write(target_folder / "construction.json", view["receipt"])
            exact_checks = None
            if name == "full768":
                old_roles = roles(b["roles_model"]["path"])
                exact_checks = dict(I_geo=np.array_equal(view["I_geo"], rgb(b["I_geo"]["path"])),
                                    H_input=np.array_equal(hidden, binary(b["H_input"]["path"])),
                                    all_model_roles=old_roles.keys() == view["masks"].keys() and all(
                                        np.array_equal(value, view["masks"][key]) for key, value in old_roles.items()))
                assert all(exact_checks.values()), (sid, exact_checks)
            # 替换全部原生隐藏 RGB；合法 I_geo 必须逐位不变。
            poisoned = dict(layers)
            poisoned["rgb"] = observation.copy()
            poisoned["rgb"][native["H_input"]] = np.array([13, 201, 77], np.uint8)
            poison_view = render_view(poisoned, plan)
            poison_exact = np.array_equal(poison_view["I_geo"], view["I_geo"])
            assert poison_exact, (sid, name, "hidden RGB entered the legal model image")
            del poisoned, poison_view
            visible = view["masks"]["C_mat"]
            distance = distance_transform_edt(~visible)
            writable = view["masks"]["W_output"]
            record = dict(available=True, plan=plan, model_wh=plan["output_wh"],
                          I_geo=pin(target_folder / "I_geo.png"), H_input=pin(target_folder / "H_input.png"),
                          roles_model=pin(target_folder / "roles_model.npz"),
                          construction=pin(target_folder / "construction.json"),
                          C_mat=pin(target_folder / "C_mat.png"),
                          VAE_reference_model=pin(target_folder / "VAE_reference_model.png"),
                          exact_legacy_input_checks=exact_checks, hidden_RGB_poison_exact=poison_exact,
                          native_pixel_per_model_pixel=[(x1 - x0) / plan["output_wh"][0],
                                                        (y1 - y0) / plan["output_wh"][1]],
                          model_counts=dict(H=int(hidden.sum()), W=int(writable.sum()),
                                            reliable_material_context=int(visible.sum()),
                                            W_max_distance_to_reliable_material=float(distance[writable].max())))
            views[name] = record
            if sid in ["scene065_raw086", "scene065_raw110"] and name in ["full768", "full1152"]:
                visible_native = binary_erosion(native["C_mat"], iterations=20) & native_valid
                save_mask(target_folder / "vae_visible_native.png", visible_native)
                resampled, _ = inverse_view(view["observation_rgb"], shape, plan["window_xyxy"])
                vae_controls.append(dict(id=sid + "_" + name, case_id=sid, view=name,
                                         visible_native=pin(target_folder / "vae_visible_native.png"),
                                         reference_role="original legally visible road; no hidden-road truth",
                                         resampling_only=known_metrics(resampled, observation, visible_native)))
        for seed in SEEDS:
            for arm in ARMS:
                view_name = arm.rsplit("_", 1)[0]
                if not views[view_name]["available"]:
                    continue
                jobs.append(dict(id=f"{sid}__{arm}__seed{seed}", case_id=sid, arm=arm,
                                 view=view_name, seed=seed, adapter="none" if arm.endswith("_base") else "T0",
                                 legacy_exact_replay_expected=(arm == "full768_T0" and seed == 101)))
        for ref in [refs[sid][key] for key in ["raw_model", "raw_native", "final_native"]]:
            check_pin(ref)
        prepared_cases.append(dict(id=sid, domain=source["domain"], native_hw=list(shape),
                                   observation=pin(folder / "observation.png"),
                                   roles_native=pin(folder / "roles_native.npz"), views=views,
                                   legacy_B_expected={key: refs[sid][key] for key in
                                                      ["raw_model", "raw_native", "final_native"]},
                                   native_counts={key: int(native[key].sum()) for key in
                                                  ["H_input", "W_output", "K", "C_mat"]}))
        print(json.dumps(dict(prepared=sid, views=len(views), GPU_used=False)), flush=True)
    model = Path(old["model"])
    jobs.sort(key=lambda job: (not job["legacy_exact_replay_expected"], IDS.index(job["case_id"]),
                               SEEDS.index(job["seed"]), ARMS.index(job["arm"])))
    config_paths = [model / name for name in ["model_index.json", "vae/config.json", "unet/config.json",
                                             "scheduler/scheduler_config.json"]]
    manifest = dict(schema=SCHEMA, complete=True, repository=str(repository), runner=pin(__file__),
                    prepared_from=pin(legacy / "prepared/prepared.json"),
                    legacy_predictions=pin(legacy / "predictions/predictions.json"),
                    source_files=source_files, checkpoint=old["checkpoint"], model=str(model),
                    model_configs=[pin(path) for path in config_paths],
                    prompt=old["prompt"], negative_prompt="", scheduler="DDIM", inference_steps=20,
                    guidance=3., eta=0., strength=1., dtype="bfloat16", weight_variant="fp16",
                    lora_rank=4, lora_alpha=4, attention_targets=TARGETS,
                    seeds=SEEDS, arms=ARMS, cases=prepared_cases, jobs=jobs, vae_controls=vae_controls,
                    maximum_diffusion_calls=48, planned_diffusion_calls=len(jobs),
                    planned_VAE_encode_decode_controls=len(vae_controls),
                    GPU_execution_authorized_by_this_file=False,
                    GPU_execution_gate="run --execute-gpu must be explicitly supplied",
                    controller_started=False, GT_opened=False, training_updates=0,
                    expected_GPU_legacy_replay_validation="pending; CPU exact input reconstruction passed only",
                    VAE_reference_input_role="legal observation RGB at each view; VAE control only, never supplied to UNet",
                    OOM_policy="stop and preserve failure; no automatic changes to resolution, steps, seed or budget",
                    scientific_limits=[
                        "same numerical seed across resolutions does not pair identical latent pixels",
                        "crop changes field of view and latent shape at identical native pixel scale",
                        "high resolution changes scale and latent shape; it does not guarantee detail recovery",
                        "attention-only versus base isolates this fitted adapter, not all LoRA architectures",
                        "VAE controls assess known visible-road reconstruction, not hidden truth",
                        "higher highpass RMS is not an objective; false structure can also raise it",
                        "four synthetic tasks reuse two source frames; scene065 hidden roads have no GT",
                        "W/K are frozen; no broad writeback, image blending, or 3D rendering changes"])
    assert len(jobs) <= 48 and len(prepared_cases) == 6
    write(out / "pilot_manifest.json", manifest)
    return out / "pilot_manifest.json"


def upgrade_prepared(args):
    """保留首版 CPU 产物，只在新目录追加 VAE 独立参考与明确重放顺序。"""
    source = Path(args.source).resolve()
    m = read(source / "pilot_manifest.json")
    assert m["schema"] == SCHEMA and m["complete"]
    out = fresh(args.out)
    shutil.copytree(source, out, dirs_exist_ok=True)
    old_prefix = str(source) + "/"
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(old_prefix):
            return str(out) + "/" + value[len(old_prefix):]
        return value
    m = relocate(m)
    sys.path.insert(0, m["repository"])
    from scripts.worldsim_v82.road_material_field.input_construction.input_builder import _plan, _neighbors, _sample
    for case in m["cases"]:
        observation = rgb(case["observation"]["path"])
        for name, view in case["views"].items():
            if not view["available"]:
                continue
            _, _, uv, _ = _plan(view["plan"], observation.shape[:2])
            sampled = np.rint(np.clip(_sample(observation, _neighbors(uv, observation.shape[:2])), 0, 255)).astype(np.uint8)
            path = out / case["id"] / name / "VAE_reference_model.png"
            Image.fromarray(sampled).save(path)
            view["VAE_reference_model"] = pin(path)
            for control in m["vae_controls"]:
                if control["case_id"] == case["id"] and control["view"] == name:
                    inverse, _ = inverse_view(sampled, case["native_hw"], view["plan"]["window_xyxy"])
                    control["resampling_only"] = known_metrics(inverse, observation, binary(control["visible_native"]["path"]))
    m["jobs"].sort(key=lambda job: (not job["legacy_exact_replay_expected"], IDS.index(job["case_id"]),
                                  SEEDS.index(job["seed"]), ARMS.index(job["arm"])))
    m["upgrade_source"] = pin(source / "pilot_manifest.json")
    m["runner"] = pin(__file__)
    m["VAE_reference_input_role"] = "legal observation RGB at each view; VAE control only, never supplied to UNet"
    m["OOM_policy"] = "stop and preserve failure; no automatic changes to resolution, steps, seed or budget"
    m["scientific_limits"].append("higher highpass RMS is not an objective; false structure can also raise it")
    write(out / "pilot_manifest.json", m)
    print(json.dumps(dict(upgraded=str(out / "pilot_manifest.json"), GPU_used=False)), flush=True)


def preflight(manifest_path, inspect_checkpoint=True):
    manifest_path = Path(manifest_path).resolve()
    m = read(manifest_path)
    assert m["schema"] == SCHEMA and m["complete"]
    assert m["planned_diffusion_calls"] == len(m["jobs"]) <= m["maximum_diffusion_calls"] == 48
    assert m["training_updates"] == 0 and not m["GT_opened"] and not m["controller_started"]
    assert m["seeds"] == SEEDS and m["arms"] == ARMS
    for record in [m["runner"], m["checkpoint"], m["prepared_from"], m["legacy_predictions"], *m["model_configs"], *m["source_files"].values()]:
        check_pin(record)
    checked_views = 0
    for case in m["cases"]:
        check_pin(case["observation"])
        check_pin(case["roles_native"])
        for record in case["legacy_B_expected"].values():
            check_pin(record)
        native = roles(case["roles_native"]["path"])
        assert not (native["W_output"] & (~native["H_input"] | native["K"])).any()
        assert not native["pending_output"].any()
        for name, view in case["views"].items():
            if not view["available"]:
                continue
            for key in ["I_geo", "H_input", "C_mat", "roles_model", "construction", "VAE_reference_model"]:
                check_pin(view[key])
            inp, hole = rgb(view["I_geo"]["path"]), binary(view["H_input"]["path"])
            assert list(inp.shape[1::-1]) == view["model_wh"]
            assert min(view["model_wh"]) > 0 and all(value % 8 == 0 for value in view["model_wh"])
            assert np.all(inp[hole] == 127) and view["hidden_RGB_poison_exact"]
            assert rgb(view["VAE_reference_model"]["path"]).shape == inp.shape
            _, valid = inverse_view(inp, case["native_hw"], view["plan"]["window_xyxy"])
            assert not (native["W_output"] & ~valid).any()
            if name == "full768":
                assert all(view["exact_legacy_input_checks"].values())
            checked_views += 1
    cases = {case["id"]: case for case in m["cases"]}
    assert list(cases) == IDS and len(m["cases"]) == len(cases) == 6
    assert len({job["id"] for job in m["jobs"]}) == len(m["jobs"])
    assert len(m["jobs"]) >= 6 and all(job["legacy_exact_replay_expected"] for job in m["jobs"][:6])
    assert not any(job["legacy_exact_replay_expected"] for job in m["jobs"][6:])
    for control in m["vae_controls"]:
        check_pin(control["visible_native"])
        case = cases[control["case_id"]]
        assert list(binary(control["visible_native"]["path"]).shape) == case["native_hw"]
        assert binary(control["visible_native"]["path"]).any()
    dependencies = {}
    for package in ["torch", "diffusers", "transformers", "peft", "numpy", "Pillow", "scipy"]:
        module = "PIL" if package == "Pillow" else package
        assert importlib.util.find_spec(module) is not None, package
        dependencies[package] = importlib.metadata.version(package)
    model = Path(m["model"])
    weight_presence = {}
    for component in ["unet", "vae", "text_encoder"]:
        found = sorted(path.name for path in (model / component).glob("*")
                       if path.suffix in [".safetensors", ".bin"])
        assert found, component
        assert any("fp16" in name for name in found), (component, "fp16 variant missing")
        weight_presence[component] = found
    checkpoint = dict(inspected=False)
    if inspect_checkpoint:
        import torch
        torch.set_num_threads(1)
        state = torch.load(m["checkpoint"]["path"], map_location="cpu", weights_only=False)
        assert state["step"] == 600 and state["arm"] == "T0"
        assert state["schema"] == "road_singleframe_full_state.v1"
        count = sum(value.numel() for value in state["lora"].values())
        assert count == 797184
        steps = sorted({int(record["step"].item()) for record in state["optimizer"]["state"].values()})
        assert steps == [600]
        assert state["contract"]["model_image_wh"] == [768, 272]
        assert state["contract"]["formal_endpoint_selection"].startswith("fixed step600")
        checkpoint = dict(inspected=True, step=600, arm="T0", trainable_parameters=count,
                          optimizer_steps=steps, model_image_wh=[768, 272],
                          sampler_epoch=state["sampler"]["epoch"], sampler_position=state["sampler"]["position"])
        del state
    result = dict(schema="road_2d_teacher_CPU_preflight.v1", complete=True, passed=True,
                  manifest=pin(manifest_path), CPU_only=True, GPU_model_loaded=False,
                  checked_cases=len(m["cases"]), checked_views=checked_views,
                  planned_diffusion_calls=len(m["jobs"]), planned_VAE_controls=len(m["vae_controls"]),
                  package_versions=dependencies, model_weights_present=weight_presence,
                  checkpoint=checkpoint, training_updates=0, GT_opened=False,
                  legacy_GPU_replay="pending; not executed", GPU_execution_gate="explicit --execute-gpu")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(args):
    if not args.execute_gpu:
        result = preflight(args.manifest)
        if args.preflight_receipt:
            write(args.preflight_receipt, result)
        return
    # 此分支只能由显式 --execute-gpu 进入；不设服务、不排队、不改变实例模式。
    m = read(args.manifest)
    preflight(args.manifest)
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    assert torch.cuda.is_available(), "显式 GPU 执行需要已存在的 CUDA 环境；本程序不会开启实例 GPU"
    out = fresh(args.out)
    ledger = dict(schema="road_2d_teacher_pilot_predictions.v1", manifest=pin(args.manifest),
                  complete=False, attempted_diffusion_calls=0, completed_diffusion_calls=0,
                  maximum_diffusion_calls=48, GT_opened=False, training_updates=0,
                  model_load_seconds=None, VAE_controls=[], jobs=[])
    write(out / "predictions.json", ledger)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    started = time.perf_counter()
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            m["model"], variant="fp16", torch_dtype=torch.bfloat16,
            safety_checker=None, requires_safety_checker=False, local_files_only=True).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        assert pipe.unet.config.in_channels == 9 and pipe.vae_scale_factor == 8
        pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0., init_lora_weights=True,
                                        target_modules=TARGETS))
        for name, parameter in pipe.unet.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        state = torch.load(m["checkpoint"]["path"], map_location="cpu", weights_only=False)
        loaded = set_peft_model_state_dict(pipe.unet, state["lora"])
        assert not loaded.unexpected_keys
        initial = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(pipe.unet).items()}
        assert initial.keys() == state["lora"].keys()
        assert all(torch.equal(value, state["lora"][key]) for key, value in initial.items())
        del state
        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            module.requires_grad_(False).eval()
        ledger["model_load_seconds"] = time.perf_counter() - started
        cases = {case["id"]: case for case in m["cases"]}
        sys.path.insert(0, m["repository"])
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_composite import compose_shadow_prediction
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_probability import rgb_sha256
        # 已知可见道路压缩对照；不运行 UNet，不读取隐藏真值。
        for control in m["vae_controls"]:
            case = cases[control["case_id"]]
            view = case["views"][control["view"]]
            image = Image.open(view["VAE_reference_model"]["path"]).convert("RGB")
            value = pipe.image_processor.preprocess(image).to("cuda", torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = pipe.vae.encode(value).latent_dist.mode() * pipe.vae.config.scaling_factor
                decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
            reconstructed = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            torch.cuda.synchronize()
            folder = out / "VAE_controls" / control["id"]
            folder.mkdir(parents=True)
            reconstructed.save(folder / "roundtrip_model.png")
            native, valid = inverse_view(np.asarray(reconstructed), case["native_hw"], view["plan"]["window_xyxy"])
            Image.fromarray(native).save(folder / "roundtrip_native.png")
            visible = binary(control["visible_native"]["path"])
            resampled, _ = inverse_view(np.asarray(image), case["native_hw"], view["plan"]["window_xyxy"])
            result = dict(id=control["id"], seconds=time.perf_counter() - t0,
                          VAE_latent_mode="mode", UNet_calls=0,
                          known_road_vs_native_observation=known_metrics(native, rgb(case["observation"]["path"]), visible),
                          VAE_increment_over_resampling=known_metrics(native, resampled, visible),
                          roundtrip_model=pin(folder / "roundtrip_model.png"),
                          roundtrip_native=pin(folder / "roundtrip_native.png"))
            ledger["VAE_controls"].append(result)
            write(out / "predictions.json", ledger)
        # 只保存预测和范围约束；真值评分是独立下一阶段。
        for job in m["jobs"]:
            case = cases[job["case_id"]]
            view = case["views"][job["view"]]
            pipe.unet.disable_adapters() if job["adapter"] == "none" else pipe.unet.enable_adapters()
            folder = out / job["id"]
            folder.mkdir()
            ledger["attempted_diffusion_calls"] += 1
            assert ledger["attempted_diffusion_calls"] <= 48
            ledger["active_job"] = job["id"]
            write(out / "predictions.json", ledger)
            width, height = view["model_wh"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = pipe(prompt=m["prompt"], negative_prompt="",
                                 image=Image.open(view["I_geo"]["path"]).convert("RGB"),
                                 mask_image=Image.open(view["H_input"]["path"]).convert("L"),
                                 height=height, width=width, strength=1., num_inference_steps=20,
                                 guidance_scale=3., eta=0.,
                                 generator=torch.Generator("cuda").manual_seed(job["seed"])).images[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            generated.save(folder / "raw_model.png")
            raw_native, valid = inverse_view(np.asarray(generated), case["native_hw"], view["plan"]["window_xyxy"])
            Image.fromarray(raw_native).save(folder / "raw_native.png")
            save_mask(folder / "candidate_native_valid.png", valid)
            observation = rgb(case["observation"]["path"])
            native = roles(case["roles_native"]["path"])
            assert not (native["W_output"] & ~valid).any()
            final, receipt = compose_shadow_prediction(observation, raw_native, native,
                                                       expected_observation_sha256=rgb_sha256(observation),
                                                       region_mode="independent")
            assert np.array_equal(final[~native["W_output"]], observation[~native["W_output"]])
            assert np.array_equal(final[native["K"]], observation[native["K"]])
            Image.fromarray(final).save(folder / "final_native.png")
            write(folder / "composition.json", receipt)
            exact = None
            if job["legacy_exact_replay_expected"]:
                exact = {name: bool(np.array_equal(rgb(folder / (name + ".png")), rgb(record["path"])))
                         for name, record in case["legacy_B_expected"].items()}
            result = dict(**job, seconds=seconds, W_outside_RGB_exact=True, K_RGB_exact=True,
                          raw_model=pin(folder / "raw_model.png"), raw_native=pin(folder / "raw_native.png"),
                          candidate_native_valid=pin(folder / "candidate_native_valid.png"),
                          final_native=pin(folder / "final_native.png"), legacy_replay_exact=exact)
            ledger["jobs"].append(result)
            ledger["completed_diffusion_calls"] += 1
            write(out / "predictions.json", ledger)
            print(json.dumps(dict(completed=job["id"], seconds=seconds,
                                  completed_calls=ledger["completed_diffusion_calls"])), flush=True)
            if exact is not None and not all(exact.values()):
                raise RuntimeError("历史重放不精确，停止后续推理并保留现场: " + job["id"])
        pipe.unet.enable_adapters()
        now = get_peft_model_state_dict(pipe.unet)
        assert all(torch.equal(value, now[key].detach().cpu()) for key, value in initial.items())
        check_pin(m["checkpoint"])
        ledger.update(complete=True, active_job=None, adapter_unchanged=True, checkpoint_unchanged=True,
                      peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.perf_counter() - started)
        write(out / "predictions.json", ledger)
    except BaseException as error:
        ledger.update(complete=False, failed=True, error_type=type(error).__name__, error=str(error),
                      wall_seconds=time.perf_counter() - started)
        write(out / "predictions.json", ledger)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--legacy", type=Path, default=LEGACY)
    prep.add_argument("--repository", type=Path, default=REPOSITORY)
    prep.add_argument("--out", type=Path, default=ROOT / "prepared_r2")
    flight = commands.add_parser("preflight")
    flight.add_argument("--manifest", type=Path, default=ROOT / "prepared_r2/pilot_manifest.json")
    flight.add_argument("--receipt", type=Path, default=ROOT / "CPU_preflight.json")
    flight.add_argument("--skip-checkpoint-load", action="store_true")
    upgrade = commands.add_parser("upgrade-prepared")
    upgrade.add_argument("--source", type=Path, default=ROOT / "prepared")
    upgrade.add_argument("--out", type=Path, default=ROOT / "prepared_r2")
    execution = commands.add_parser("run")
    execution.add_argument("--manifest", type=Path, default=ROOT / "prepared_r2/pilot_manifest.json")
    execution.add_argument("--out", type=Path, default=ROOT / "GPU_pilot_opt_in")
    execution.add_argument("--preflight-receipt", type=Path)
    execution.add_argument("--execute-gpu", action="store_true",
                           help="显式运行已登记48次以内推理；没有本参数只做CPU预检")
    args = parser.parse_args()
    if not (args.command == "run" and args.execute_gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "prepare":
        path = prepare(args)
        print(json.dumps(dict(manifest=str(path), GPU_used=False)), flush=True)
    elif args.command == "preflight":
        result = preflight(args.manifest, inspect_checkpoint=not args.skip_checkpoint_load)
        write(args.receipt, result)
    elif args.command == "upgrade-prepared":
        upgrade_prepared(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
