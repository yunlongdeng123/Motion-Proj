"""固定full1152/T0的60次提示词配对入口；默认只做本地CPU计划检查。

历史源码不变。未来远端须重新prepare并通过runtime preflight；只有显式
--execute-gpu可加载GPU。该入口不评分、不选seed、不自动进入第二轮。
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
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")

import numpy as np
from PIL import Image

IDS = ["visible_scene081_raw000_v0", "visible_scene081_raw000_v1", "visible_scene081_raw050_v0",
       "visible_scene081_raw050_v1", "scene065_raw086", "scene065_raw110"]
SEEDS = [101, 202, 303, 404, 505]
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
INFERENCE = dict(model_wh=[1152, 768], view="full1152", adapter="T0", scheduler="DDIM", inference_steps=20,
                 guidance=3.0, eta=0.0, strength=1.0, dtype="bfloat16", weight_variant="fp16", negative_prompt="")
SCHEMA = "road_2d_teacher_prompt_round_plan.v1"
PREDICTION_SCHEMA = "road_2d_teacher_prompt_round_predictions.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "next_round_config.json"


class ContractError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ContractError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check_pin(record):
    need(Path(record["path"]).is_file(), "冻结文件不可用: " + record["path"])
    need(sha(record["path"]) == record["sha256"], "冻结文件变化: " + record["path"])


def rgb(path):
    with Image.open(path) as image:
        need(image.mode == "RGB", "需要显式RGB: " + str(path))
        return np.asarray(image).copy()


def binary(path):
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"))
    need(np.isin(value, [0, 255]).all(), "mask不是二值: " + str(path))
    return value > 127


def roles(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def validate_config(config, baseline):
    need(config.get("schema") == "road_2d_teacher_next_round.v1", "config schema不符")
    need(config.get("selected_arm") == "full1152_T0" and config.get("seeds") == SEEDS, "方法或5个冻结seed变化")
    need(config.get("expected_case_ids") == IDS, "六个case或顺序变化")
    need(config.get("inference") == INFERENCE, "固定推理合同变化")
    need(config.get("planned_diffusion_calls") == config.get("maximum_diffusion_calls") == 60, "必须完整重跑60次")
    need(config.get("historical_replay_expected") == 12 and config.get("training_updates") == 0, "重放/训练预算变化")
    need(config.get("seed_selection") == "fixed_before_generation" and config.get("automatic_candidate_selection") is False, "禁止自动选seed/候选")
    need(config.get("maximum_prompt_comparison_rounds") == 2 and config.get("current_prompt_comparison_round") in (1, 2), "提示词轮数合同变化")
    for name in ("reuse_historical_outputs", "second_prompt_round_automatically_enabled", "three_dimensional_rendering_changes", "GT_used_for_generation_or_candidate_selection"):
        need(config.get(name, False) is False, "禁止启用: " + name)
    need(config.get("additional_VAE_control_calls", 0) == 0, "本轮没有额外VAE控制预算")
    prompts = config.get("prompts", [])
    need([p.get("id") for p in prompts] == ["legacy", "material_match_v1"], "仅允许两个冻结prompt ID")
    need(all(isinstance(p.get("text"), str) and p["text"].strip() for p in prompts), "prompt文本为空")
    need(prompts[0]["text"] == baseline["prompt"], "legacy必须等于原完整英文prompt")
    need(prompts[0].get("role") == "historical_baseline" and prompts[1].get("role") == "candidate_pending_comparison", "prompt角色变化")
    need(baseline.get("schema") == "road_2d_teacher_pilot.v1" and baseline.get("complete") is True, "原manifest未完成")
    need(baseline["checkpoint"]["sha256"] == "a75423e126b93eca22e0ce587782b3dc4edd76654d6a929302118aa45bda14ae", "T0 step600来源变化")
    for key in ("scheduler", "inference_steps", "guidance", "eta", "strength", "dtype", "weight_variant"):
        need(baseline[key] == INFERENCE[key], "原推理合同不匹配: " + key)


def build_jobs(config, historical):
    need(historical.get("schema") == "road_2d_teacher_pilot_predictions.v1" and historical.get("complete") is True, "旧48预测未冻结")
    need(historical.get("attempted_diffusion_calls") == historical.get("completed_diffusion_calls") == 48, "旧48预测计数不符")
    need(historical.get("adapter_unchanged") is True and historical.get("checkpoint_unchanged") is True, "旧adapter绑定未通过")
    lookup = {(j["case_id"], j["arm"], j["seed"]): j for j in historical["jobs"]}
    need(len(lookup) == len(historical["jobs"]) == 48, "旧预测组合重复/缺失")
    jobs = []
    # 12次工程重放也实际生成，包含在60预算内；首个不通过即停止。
    order = [(sid, "legacy", seed) for seed in (101, 202) for sid in IDS]
    order += [(sid, prompt["id"], seed) for prompt in config["prompts"] for seed in SEEDS for sid in IDS
              if not (prompt["id"] == "legacy" and seed in (101, 202))]
    for sid, prompt_id, seed in order:
        replay = prompt_id == "legacy" and seed in (101, 202)
        row = dict(id=f"{sid}__full1152_T0__{prompt_id}__seed{seed}", case_id=sid, arm="full1152_T0",
                   view="full1152", adapter="T0", prompt_id=prompt_id, seed=seed, historical_replay_expected=replay)
        if replay:
            old = lookup[(sid, "full1152_T0", seed)]
            need(old.get("W_outside_RGB_exact") is True and old.get("K_RGB_exact") is True, "旧full1152范围未通过")
            row["historical_expected"] = {key: copy.deepcopy(old[key]) for key in ("raw_model", "raw_native", "final_native")}
            row["historical_job_id"] = old["id"]
        jobs.append(row)
    need(len(jobs) == len({j["id"] for j in jobs}) == 60, "60矩阵不完整")
    return jobs


def token_check(prompts, tokenizer_path, *, required):
    result = dict(checked=False, maximum_tokens_including_BOS_EOS=77, truncation_allowed=False,
                  status="pending_runtime_tokenizer", tokenizer_path=str(tokenizer_path) if tokenizer_path else None, prompts=[])
    if tokenizer_path is None or not Path(tokenizer_path).is_dir() or importlib.util.find_spec("transformers") is None:
        if required:
            raise ContractError("本地CLIP tokenizer/transformers不可用；必须完成真实77-token检查后才能执行")
        result["reason"] = "未找到可用的本地SD1.5 CLIP tokenizer与transformers；未估算/伪造token数"
        return result
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    need(tokenizer.model_max_length == 77, "实际tokenizer最大长度不是77")
    for prompt in prompts:
        ids = tokenizer.encode(prompt["text"], add_special_tokens=True, truncation=False)
        row = dict(prompt_id=prompt["id"], tokens_including_BOS_EOS=len(ids), exceeds_limit=len(ids) > 77)
        result["prompts"].append(row)
        need(len(ids) <= 77, f"prompt {prompt['id']} 共{len(ids)} tokens，超过77；停止，禁止自动截断/改词")
    result.update(checked=True, status="passed_no_truncation")
    return result


def prepare(config_path, baseline_path, historical_path, output, tokenizer_path=None):
    config, baseline, historical = map(read, (config_path, baseline_path, historical_path))
    validate_config(config, baseline)
    need(sha(baseline_path) == historical["manifest"]["sha256"], "原manifest快照不是旧GPU运行的实际绑定")
    need([c["id"] for c in baseline["cases"]] == IDS, "原case顺序变化")
    cases = []
    for old in baseline["cases"]:
        view = old["views"]["full1152"]
        need(view["available"] and view["model_wh"] == [1152, 768], "full1152合法输入不可用")
        need(view["plan"] == dict(output_wh=[1152, 768], window_xyxy=[0, 0, 1920, 1280]), "full1152视场变化")
        cases.append({key: copy.deepcopy(old[key]) for key in ("id", "domain", "native_hw", "observation", "roles_native")}
                     | dict(view=copy.deepcopy({key: view[key] for key in ("model_wh", "plan", "I_geo", "H_input", "C_mat", "roles_model", "construction")})))
    output = Path(output)
    need(not output.exists(), "保留已有计划，请选择新out")
    manifest = dict(schema=SCHEMA, complete=True, preparation_role="CPU_plan_only_not_runtime_preflight",
        source_pins=dict(config=pin(config_path), baseline_manifest=pin(baseline_path), historical_predictions=pin(historical_path), runner=pin(__file__)),
        configuration=config, model=baseline["model"], checkpoint=baseline["checkpoint"], model_configs=baseline["model_configs"],
        source_files=baseline["source_files"], repository=baseline["repository"], cases=cases, jobs=build_jobs(config, historical),
        token_check=token_check(config["prompts"], tokenizer_path, required=False),
        planned_diffusion_calls=60, maximum_diffusion_calls=60, historical_replay_expected=12,
        first_replay_required_before_remaining=True, training_updates=0, GT_opened=False,
        automatic_candidate_selection=False, best_of_N_score_implemented=False, additional_VAE_control_calls=0,
        runtime_preflight_passed=False, GPU_execution_performed=False,
        scoring_interface="new schema and prompt_id; freeze/pixel-scope gate only; old score_pilot.py hardcoded48 is NOT compatible",
        remote_reprepare_required="Local Windows plans must not be copied as executable Linux plans; rerun prepare remotely with Linux source paths.")
    output.mkdir(parents=True)
    write(output / "plan.json", manifest)
    return output / "plan.json"


def plan_check(manifest_path):
    m = read(manifest_path)
    need(m.get("schema") == SCHEMA and m.get("complete") is True, "计划schema/完成性不符")
    for record in m["source_pins"].values():
        check_pin(record)
    need(m["source_pins"]["runner"]["sha256"] == sha(__file__), "当前入口不是冻结版本")
    config = read(m["source_pins"]["config"]["path"])
    baseline = read(m["source_pins"]["baseline_manifest"]["path"])
    history = read(m["source_pins"]["historical_predictions"]["path"])
    validate_config(config, baseline)
    need(config == m["configuration"], "配置payload/pin不同")
    for key in ("model", "checkpoint", "model_configs", "source_files", "repository"):
        need(m[key] == baseline[key], "继承的冻结模型/源码字段变化: " + key)
    need(sha(m["source_pins"]["baseline_manifest"]["path"]) == history["manifest"]["sha256"], "原manifest绑定不同")
    need(m["jobs"] == build_jobs(config, history), "60计划或12个full1152历史引用变化")
    need(m.get("planned_diffusion_calls") == m.get("maximum_diffusion_calls") == 60
         and m.get("historical_replay_expected") == 12 and m.get("additional_VAE_control_calls") == 0,
         "顶层60/60/12或额外VAE计数变化")
    need([c["id"] for c in m["cases"]] == IDS and len(m["cases"]) == 6, "case集合变化")
    for record in m["cases"]:
        original = next(c for c in baseline["cases"] if c["id"] == record["id"])
        need(all(record[key] == original[key] for key in ("id", "domain", "native_hw", "observation", "roles_native")), "原生输入绑定变化")
        expected_view = {key: original["views"]["full1152"][key] for key in ("model_wh", "plan", "I_geo", "H_input", "C_mat", "roles_model", "construction")}
        need(record["view"] == expected_view, "full1152输入绑定变化")
    need(m["training_updates"] == 0 and m["GT_opened"] is False and m["automatic_candidate_selection"] is False, "执行范围变化")
    return m


def preflight(manifest_path, *, runtime=False, tokenizer_path=None):
    m = plan_check(manifest_path)
    result = dict(schema="road_2d_teacher_next_round_preflight.v1", local_plan_passed=True,
        runtime_preflight_passed=False, runtime_assets_verified=False, GPU_model_loaded=False,
        planned_diffusion_calls=60, historical_replay_expected=12, training_updates=0, GT_opened=False,
        token_check=token_check(m["configuration"]["prompts"], tokenizer_path, required=False))
    if not runtime:
        return result
    for record in [m["checkpoint"], *m["model_configs"], *m["source_files"].values()]:
        check_pin(record)
    for case in m["cases"]:
        for record in [case["observation"], case["roles_native"], *[case["view"][k] for k in ("I_geo", "H_input", "C_mat", "roles_model", "construction")]]:
            check_pin(record)
        r = roles(case["roles_native"]["path"])
        need(not (r["W_output"] & (~r["H_input"] | r["K"])).any() and not r["pending_output"].any(), "原生W/H/K角色冲突")
        image, hole = rgb(case["view"]["I_geo"]["path"]), binary(case["view"]["H_input"]["path"])
        need(image.shape == (768, 1152, 3) and hole.shape == (768, 1152) and np.all(image[hole] == 127), "full1152合法条件形状/中性化失败")
    by_case = {c["id"]: c for c in m["cases"]}
    for job in m["jobs"][:12]:
        need(job["historical_replay_expected"] and "full1152_T0" in job["historical_job_id"], "历史门不是full1152")
        for record in job["historical_expected"].values():
            check_pin(record)
        expected = job["historical_expected"]
        need(rgb(expected["raw_model"]["path"]).shape == (768, 1152, 3), "历史raw不是1152×768")
        old_raw, final = rgb(expected["raw_native"]["path"]), rgb(expected["final_native"]["path"])
        case = by_case[job["case_id"]]
        observation, r = rgb(case["observation"]["path"]), roles(case["roles_native"]["path"])
        need(old_raw.shape == final.shape == observation.shape == (1280, 1920, 3), "历史原生网格不符")
        need(np.array_equal(final[~r["W_output"]], observation[~r["W_output"]]) and np.array_equal(final[r["K"]], observation[r["K"]]), "历史W外/K像素门失败")
    model = Path(m["model"])
    for component in ("unet", "vae", "text_encoder"):
        need(any("fp16" in path.name and path.suffix in (".safetensors", ".bin") for path in (model / component).glob("*")), "fp16权重缺失: " + component)
    for package in ("torch", "diffusers", "transformers", "peft"):
        need(importlib.util.find_spec(package) is not None, "缺依赖: " + package)
    import torch
    state = torch.load(m["checkpoint"]["path"], map_location="cpu", weights_only=False)
    need(state["step"] == 600 and state["arm"] == "T0" and sum(v.numel() for v in state["lora"].values()) == 797184, "checkpoint不是冻结T0 step600")
    del state
    result["token_check"] = token_check(m["configuration"]["prompts"], tokenizer_path or model / "tokenizer", required=True)
    result.update(runtime_preflight_passed=True, runtime_assets_verified=True,
                  versions={name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "peft")})
    return result


def replay_exact_or_stop(job, folder):
    if not job["historical_replay_expected"]:
        return None
    exact = {name: bool(np.array_equal(rgb(Path(folder) / (name + ".png")), rgb(record["path"])))
             for name, record in job["historical_expected"].items()}
    need(all(exact.values()), "full1152历史三图不精确；立即停止，禁止自动改seed/prompt/方法: " + job["id"])
    return exact


def run(manifest_path, output, *, execute_gpu=False):
    if not execute_gpu:
        return preflight(manifest_path, runtime=False)
    flight = preflight(manifest_path, runtime=True)
    m = read(manifest_path)
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    need(torch.cuda.is_available(), "GPU不可用；入口不启动远端实例")
    output = Path(output)
    need(not output.exists(), "保留已有GPU现场，请指定新out")
    output.mkdir(parents=True)
    ledger = dict(schema=PREDICTION_SCHEMA, complete=False, manifest=pin(manifest_path), runtime_preflight=flight,
        attempted_diffusion_calls=0, completed_diffusion_calls=0, maximum_diffusion_calls=60,
        first_historical_replay_passed=False, historical_replays_passed=0, reuse_historical_outputs=False,
        additional_VAE_control_calls=0, training_updates=0, GT_opened=False, jobs=[], automatic_candidate_selection=False)
    write(output / "predictions.json", ledger)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(101); np.random.seed(101); torch.manual_seed(101); torch.cuda.manual_seed_all(101)
    started = time.perf_counter()
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(m["model"], variant="fp16", torch_dtype=torch.bfloat16,
            safety_checker=None, requires_safety_checker=False, local_files_only=True).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        need(pipe.unet.config.in_channels == 9 and pipe.vae_scale_factor == 8, "基础inpainting架构不同")
        pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0., init_lora_weights=True, target_modules=TARGETS))
        for name, parameter in pipe.unet.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        state = torch.load(m["checkpoint"]["path"], map_location="cpu", weights_only=False)
        loaded = set_peft_model_state_dict(pipe.unet, state["lora"])
        need(not loaded.unexpected_keys, "LoRA装载出现未知键")
        initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(pipe.unet).items()}
        need(initial.keys() == state["lora"].keys() and all(torch.equal(v, state["lora"][k]) for k, v in initial.items()), "LoRA装载不精确")
        del state
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            module.requires_grad_(False).eval()
        # 检查实际pipeline tokenizer，防路径/processor与CPU token检查不一致。
        for prompt in m["configuration"]["prompts"]:
            need(len(pipe.tokenizer.encode(prompt["text"], add_special_tokens=True, truncation=False)) <= 77, "实际pipeline将截断prompt")
        sys.path.insert(0, m["repository"])
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_composite import compose_shadow_prediction
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_probability import rgb_sha256
        cases = {c["id"]: c for c in m["cases"]}
        prompts = {p["id"]: p["text"] for p in m["configuration"]["prompts"]}
        for index, job in enumerate(m["jobs"]):
            need(index == 0 or ledger["first_historical_replay_passed"], "首个历史重放未通过，禁止后续调用")
            case = cases[job["case_id"]]
            folder = output / job["id"]
            folder.mkdir()
            ledger["attempted_diffusion_calls"] += 1
            need(ledger["attempted_diffusion_calls"] <= 60, "超出60预算")
            ledger["active_job"] = job["id"]
            write(output / "predictions.json", ledger)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = pipe(prompt=prompts[job["prompt_id"]], negative_prompt="",
                    image=Image.open(case["view"]["I_geo"]["path"]).convert("RGB"),
                    mask_image=Image.open(case["view"]["H_input"]["path"]).convert("L"), height=768, width=1152,
                    strength=1., num_inference_steps=20, guidance_scale=3., eta=0.,
                    generator=torch.Generator("cuda").manual_seed(job["seed"])).images[0]
            torch.cuda.synchronize(); seconds = time.perf_counter() - t0
            generated.save(folder / "raw_model.png")
            observation, r = rgb(case["observation"]["path"]), roles(case["roles_native"]["path"])
            raw_native = np.asarray(generated.resize((1920, 1280), Image.Resampling.BILINEAR)).copy()
            Image.fromarray(raw_native).save(folder / "raw_native.png")
            final, composition = compose_shadow_prediction(observation, raw_native, r,
                expected_observation_sha256=rgb_sha256(observation), region_mode="independent")
            need(np.array_equal(final[~r["W_output"]], observation[~r["W_output"]]) and np.array_equal(final[r["K"]], observation[r["K"]]), "W外/K改变")
            Image.fromarray(final).save(folder / "final_native.png")
            Image.fromarray(np.full((1280, 1920), 255, np.uint8)).save(folder / "candidate_native_valid.png")
            write(folder / "composition.json", composition)
            exact = replay_exact_or_stop(job, folder)
            if exact is not None:
                ledger["historical_replays_passed"] += 1
            if index == 0:
                need(exact is not None, "首个job必须是full1152历史门")
                ledger["first_historical_replay_passed"] = True
            ledger["jobs"].append(dict(**job, seconds=seconds, historical_replay_exact=exact,
                **{name: pin(folder / (name + ".png")) for name in ("raw_model", "raw_native", "final_native", "candidate_native_valid")},
                W_outside_RGB_exact=True, K_RGB_exact=True))
            ledger["completed_diffusion_calls"] += 1
            write(output / "predictions.json", ledger)
            print(json.dumps(dict(completed=job["id"], calls=ledger["completed_diffusion_calls"])), flush=True)
        current = get_peft_model_state_dict(pipe.unet)
        need(all(torch.equal(value, current[key].detach().cpu()) for key, value in initial.items()), "adapter变化")
        check_pin(m["checkpoint"])
        need(ledger["completed_diffusion_calls"] == 60 and ledger["historical_replays_passed"] == 12, "60/12最终计数不符")
        ledger.update(complete=True, active_job=None, adapter_unchanged=True, checkpoint_unchanged=True,
                      peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.perf_counter()-started)
        write(output / "predictions.json", ledger)
    except BaseException as error:
        ledger.update(complete=False, failed=True, error_type=type(error).__name__, error=str(error), wall_seconds=time.perf_counter()-started)
        write(output / "predictions.json", ledger)
        raise
    return ledger


def freeze_gate(predictions_path, manifest_path):
    """仅冻结/真实像素范围门，不读取GT、不输出质量分或best-of-N选择。"""
    m, p = plan_check(manifest_path), read(predictions_path)
    need(p.get("schema") == PREDICTION_SCHEMA and p.get("complete") is True, "60预测未冻结完成")
    need(p["manifest"] == pin(manifest_path), "预测与计划绑定不同")
    need(p["attempted_diffusion_calls"] == p["completed_diffusion_calls"] == 60 and p["historical_replays_passed"] == 12, "60/12计数不符")
    need(p["GT_opened"] is False and p["training_updates"] == 0 and p["automatic_candidate_selection"] is False, "预测范围冲突")
    need(p["adapter_unchanged"] and p["checkpoint_unchanged"] and p["first_historical_replay_passed"], "参数/首门未通过")
    need(len(p["jobs"]) == 60, "预测job数量不符")
    cases = {c["id"]: c for c in m["cases"]}
    for case in cases.values():
        for record in [case["observation"], case["roles_native"],
                       *[case["view"][key] for key in ("I_geo", "H_input", "C_mat", "roles_model", "construction")]]:
            check_pin(record)
    for actual, planned in zip(p["jobs"], m["jobs"]):
        need(all(actual.get(key) == value for key, value in planned.items()), "预测prompt/seed/历史引用与计划不同")
        for name in ("raw_model", "raw_native", "final_native", "candidate_native_valid"):
            check_pin(actual[name])
        case = cases[planned["case_id"]]
        model, raw, final = (rgb(actual[name]["path"]) for name in ("raw_model", "raw_native", "final_native"))
        need(model.shape == (768, 1152, 3) and raw.shape == final.shape == (1280, 1920, 3), "输出网格变化")
        need(np.array_equal(raw, np.asarray(Image.fromarray(model).resize((1920, 1280), Image.Resampling.BILINEAR))), "逆变换不一致")
        observation, r = rgb(case["observation"]["path"]), roles(case["roles_native"]["path"])
        need(binary(actual["candidate_native_valid"]["path"]).all(), "完整FOV出现无效来源")
        need(np.array_equal(final[~r["W_output"]], observation[~r["W_output"]]) and np.array_equal(final[r["K"]], observation[r["K"]]), "实际W外/K改变")
        need(np.array_equal(final[r["W_output"]], raw[r["W_output"]]), "W内不是原始生成硬回填")
        if planned["historical_replay_expected"]:
            need(actual.get("historical_replay_exact") and all(actual["historical_replay_exact"].values()), "历史重放未通过")
            for name, record in planned["historical_expected"].items():
                check_pin(record)
                need(np.array_equal(rgb(actual[name]["path"]), rgb(record["path"])), "历史三图实际像素不等")
    return dict(schema="road_2d_teacher_prompt_round_freeze_gate.v1", passed=True, jobs=60, historical_replays=12,
                actual_W_outside_exact=True, actual_K_exact=True, GT_opened=False, quality_scoring_implemented=False,
                predictions=pin(predictions_path), manifest=pin(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--baseline-manifest", type=Path, required=True)
    prep.add_argument("--historical-predictions", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--tokenizer-path", type=Path)
    flight = commands.add_parser("preflight")
    flight.add_argument("--manifest", type=Path, required=True)
    flight.add_argument("--runtime", action="store_true")
    flight.add_argument("--tokenizer-path", type=Path)
    flight.add_argument("--receipt", type=Path)
    execution = commands.add_parser("run")
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--out", type=Path, required=True)
    execution.add_argument("--execute-gpu", action="store_true")
    gate = commands.add_parser("freeze-gate")
    gate.add_argument("--manifest", type=Path, required=True)
    gate.add_argument("--predictions", type=Path, required=True)
    gate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not (args.command == "run" and args.execute_gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "prepare":
        result = dict(plan=str(prepare(args.config, args.baseline_manifest, args.historical_predictions, args.out, args.tokenizer_path)), GPU_used=False)
    elif args.command == "preflight":
        result = preflight(args.manifest, runtime=args.runtime, tokenizer_path=args.tokenizer_path)
        if args.receipt:
            write(args.receipt, result)
    elif args.command == "freeze-gate":
        result = freeze_gate(args.predictions, args.manifest)
        write(args.receipt, result)
    else:
        result = run(args.manifest, args.out, execute_gpu=args.execute_gpu)
    print(json.dumps(result if args.command != "run" or not args.execute_gpu else {"complete": result["complete"], "completed_diffusion_calls": result["completed_diffusion_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
