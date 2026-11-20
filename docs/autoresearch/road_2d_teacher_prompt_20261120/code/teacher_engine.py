"""固定二维教师的 GPU 执行组件；smoke 与正式分片复用相同生成路径。"""
from pathlib import Path
import random
import sys
import time

import infer_next_round as contract
from infer_next_round import np, Image, need, read, pin, check_pin, rgb, roles, write


class TeacherEngine:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.flight = contract.preflight(manifest_path, runtime=True)
        self.manifest = read(manifest_path)
        import torch
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
        need(torch.cuda.is_available(), "需要当前实例 GPU")
        self.torch, self.get_state = torch, get_peft_model_state_dict
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        started = time.perf_counter()
        m = self.manifest
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            m["model"], variant="fp16", torch_dtype=torch.bfloat16,
            safety_checker=None, requires_safety_checker=False, local_files_only=True).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        need(pipe.unet.config.in_channels == 9 and pipe.vae_scale_factor == 8, "inpainting架构变化")
        pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0., init_lora_weights=True,
                                        target_modules=contract.TARGETS))
        for name, parameter in pipe.unet.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        state = torch.load(m["checkpoint"]["path"], map_location="cpu", weights_only=False)
        loaded = set_peft_model_state_dict(pipe.unet, state["lora"])
        need(not loaded.unexpected_keys, "LoRA未知键")
        self.initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(pipe.unet).items()}
        need(self.initial.keys() == state["lora"].keys() and
             all(torch.equal(v, state["lora"][k]) for k, v in self.initial.items()), "LoRA加载不精确")
        del state
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            module.requires_grad_(False).eval()
        for prompt in m["configuration"]["prompts"]:
            need(len(pipe.tokenizer.encode(prompt["text"], add_special_tokens=True, truncation=False)) <= 77,
                 "实际pipeline截断prompt")
        self.pipe = pipe
        self.load_seconds = time.perf_counter() - started
        self.cases = {c["id"]: c for c in m["cases"]}
        self.prompts = {p["id"]: p["text"] for p in m["configuration"]["prompts"]}
        sys.path.insert(0, m["repository"])
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_composite import compose_shadow_prediction
        from scripts.worldsim_v82.road_material_field.input_construction.shadow_probability import rgb_sha256
        self.compose, self.rgb_sha = compose_shadow_prediction, rgb_sha256

    def predict(self, job, folder):
        folder = Path(folder)
        need(not folder.exists(), "保留已有生成现场: " + str(folder))
        folder.mkdir(parents=True)
        case, torch = self.cases[job["case_id"]], self.torch
        self.pipe.unet.enable_adapters()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.pipe(
                prompt=self.prompts[job["prompt_id"]], negative_prompt="",
                image=Image.open(case["view"]["I_geo"]["path"]).convert("RGB"),
                mask_image=Image.open(case["view"]["H_input"]["path"]).convert("L"),
                height=768, width=1152, strength=1., num_inference_steps=20,
                guidance_scale=3., eta=0., generator=torch.Generator("cuda").manual_seed(job["seed"])).images[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        generated.save(folder / "raw_model.png")
        raw = np.asarray(generated.resize((1920, 1280), Image.Resampling.BILINEAR)).copy()
        Image.fromarray(raw).save(folder / "raw_native.png")
        observation, regions = rgb(case["observation"]["path"]), roles(case["roles_native"]["path"])
        final, composition = self.compose(observation, raw, regions,
            expected_observation_sha256=self.rgb_sha(observation), region_mode="independent")
        need(np.array_equal(final[~regions["W_output"]], observation[~regions["W_output"]]) and
             np.array_equal(final[regions["K"]], observation[regions["K"]]), "实际W外或K变化")
        need(np.array_equal(final[regions["W_output"]], raw[regions["W_output"]]), "W内并非raw硬回填")
        Image.fromarray(final).save(folder / "final_native.png")
        Image.fromarray(np.full((1280, 1920), 255, np.uint8)).save(folder / "candidate_native_valid.png")
        write(folder / "composition.json", composition)
        exact = contract.replay_exact_or_stop(job, folder)
        return dict(**job, seconds=seconds, historical_replay_exact=exact,
            **{name: pin(folder / (name + ".png")) for name in
               ("raw_model", "raw_native", "final_native", "candidate_native_valid")},
            W_outside_RGB_exact=True, K_RGB_exact=True)

    def finalize(self):
        current = self.get_state(self.pipe.unet)
        need(all(self.torch.equal(v, current[k].detach().cpu()) for k, v in self.initial.items()), "adapter变化")
        check_pin(self.manifest["checkpoint"])
        return dict(adapter_unchanged=True, checkpoint_unchanged=True,
                    model_load_seconds=self.load_seconds,
                    peak_GPU_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    runtime_preflight=self.flight)
