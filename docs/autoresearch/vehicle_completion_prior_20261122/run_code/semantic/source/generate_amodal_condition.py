"""One SD1.5-base amodal vehicle condition; exact observed RGB is protected."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
import numpy as np
from PIL import Image

ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1")
MODEL = Path("/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
PROMPT = "rear view same black Dodge pickup truck, complete rear bumper and rear tires, coherent full vehicle body, realistic proportions, plain neutral background"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""): digest.update(block)
    return digest.hexdigest()


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
def read(path): return json.loads(Path(path).read_text())
def rgb(path): return np.asarray(Image.open(path).convert("RGB")).copy()
def mask(path): return np.asarray(Image.open(path).convert("L")) > 0


def prepare(out):
    if (out/"plan.json").exists():
        raise RuntimeError("Preserve fixed prepared plan; do not regenerate conditions")
    source = ROOT/"conditioning/train/110"
    image = rgb(source/"context_rgb.png"); partial = rgb(source/"partial_visible_rgb.png")
    visible = mask(source/"visible_mask.png"); unknown = mask(source/"unknown_potential_mask.png")
    potential = mask(source/"amodal_box_potential_mask.png"); background = mask(source/"known_background_mask.png")
    if not (image.shape == partial.shape == (*visible.shape, 3)) or image.shape != (364, 432, 3):
        raise RuntimeError("Fixed native crop dimensions changed")
    if (visible & unknown).any() or not np.array_equal(visible | unknown, potential) or not np.array_equal(background, ~potential):
        raise RuntimeError("V/U/B domain partition changed")
    if not np.array_equal(image[visible], partial[visible]): raise RuntimeError("Visible core is not original RGB")
    image_whole = np.full((512, 512, 3), 128, np.uint8)
    visible_whole = np.zeros((512, 512), bool); unknown_whole = np.zeros_like(visible_whole)
    x0, y0 = (512-432)//2, (512-364)//2
    rect = (slice(y0,y0+364), slice(x0,x0+432))
    image_whole[rect][visible] = image[visible]; visible_whole[rect] = visible; unknown_whole[rect] = unknown
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(MODEL/"tokenizer"), local_files_only=True)
    tokens = tokenizer.encode(PROMPT, add_special_tokens=True, truncation=False)
    if len(tokens)>77: raise RuntimeError("Fixed prompt exceeds CLIP token limit")
    if read(MODEL/"unet/config.json")["in_channels"] != 9: raise RuntimeError("Expected real SD1.5 inpaint base")
    for item in ("unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors"):
        if not (MODEL/item).is_file(): raise RuntimeError("Offline base weight is absent: "+item)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image_whole).save(out/"input512.png")
    Image.fromarray(unknown_whole.astype(np.uint8)*255).save(out/"unknown_mask512.png")
    np.savez_compressed(out/"roles.npz", original_crop_rgb=image, visible=visible, unknown=unknown,
        potential=potential, known_background=background, visible512=visible_whole, unknown512=unknown_whole)
    plan=dict(schema="a20.amodal_semantic_condition.v1", training_raw=110, heldout_used=False,
        model=str(MODEL), model_revision=MODEL.name, prompt=PROMPT, negative_prompt="", seed=101,
        scheduler="DDIM", inference_steps=20, guidance_scale=3., eta=0., strength=1.,
        dtype="bfloat16", weight_variant="fp16", use_lora=False, model_wh=[512,512],
        crop_wh=[432,364], crop_xyxy_in512=[x0,y0,x0+432,y0+364], resizing_used=False,
        neutral_rgb=[128,128,128], visible_pixels=int(visible.sum()), unknown_pixels=int(unknown.sum()),
        known_background_pixels=int(background.sum()), prompt_tokens=len(tokens),
        visible_policy="After diffusion, hard-copy original raw110 RGB on every V pixel; U only generated; background hard-copy neutral128",
        alpha_policy="Generated image-derived silhouette estimate: >24 RGB max-distance from neutral within U, V union, retain V-touching components, 5px close, fill enclosed holes; never GT silhouette",
        input_pins=[dict(path=str(source/name),sha256=sha(source/name)) for name in
            ["context_rgb.png","partial_visible_rgb.png","visible_mask.png","unknown_potential_mask.png","amodal_box_potential_mask.png","known_background_mask.png","mapping.json"]],
        prepared_pins=[dict(path=str(out/name),sha256=sha(out/name)) for name in ["input512.png","unknown_mask512.png","roles.npz"]],
        source_sha256=sha(__file__), gpu_preparation_used=False)
    write(out/"plan.json", plan); print(json.dumps(plan), flush=True)


def execute(out):
    if (out/"raw_model512.png").exists() or (out/"receipt.json").exists():
        raise RuntimeError("A single fixed SD candidate already exists; a second call is prohibited")
    plan = read(out/"plan.json")
    if plan["source_sha256"] != sha(__file__): raise RuntimeError("Prepared source changed")
    for pin in plan["input_pins"] + plan["prepared_pins"]:
        if sha(pin["path"]) != pin["sha256"]: raise RuntimeError("Frozen input changed: "+pin["path"])
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from scipy.ndimage import label, binary_closing, binary_fill_holes
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.45, 0)
    torch.cuda.reset_peak_memory_stats(); torch.manual_seed(101); np.random.seed(101)
    started=time.monotonic()
    pipe=StableDiffusionInpaintPipeline.from_pretrained(plan["model"], variant="fp16",
        torch_dtype=torch.bfloat16, safety_checker=None, requires_safety_checker=False,
        local_files_only=True, use_safetensors=True)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    for module in (pipe.unet,pipe.vae,pipe.text_encoder): module.eval().requires_grad_(False)
    if getattr(pipe.unet,"peft_config",None): raise RuntimeError("LoRA must not be loaded")
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=False)
    finite_calls=[]
    def forward_hook(module, inputs, result):
        value=result.sample if hasattr(result,"sample") else result[0]
        finite=bool(torch.isfinite(value).all().item()); finite_calls.append(finite)
        if not finite: raise RuntimeError("Nonfinite SD inpaint forward")
    pipe.unet.register_forward_hook(forward_hook)
    generation_started=time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        generated=pipe(prompt=plan["prompt"],negative_prompt="",image=Image.open(out/"input512.png").convert("RGB"),
            mask_image=Image.open(out/"unknown_mask512.png").convert("L"),height=512,width=512,
            strength=1.,num_inference_steps=20,guidance_scale=3.,eta=0.,
            generator=torch.Generator("cuda").manual_seed(101)).images[0]
    torch.cuda.synchronize(); generation_seconds=time.monotonic()-generation_started
    generated.save(out/"raw_model512.png")
    with np.load(out/"roles.npz",allow_pickle=False) as bank: roles={key:bank[key].copy() for key in bank.files}
    x0,y0,x1,y1=plan["crop_xyxy_in512"]; raw=np.asarray(generated)[y0:y1,x0:x1].copy()
    visible,unknown,potential=roles["visible"],roles["unknown"],roles["potential"]
    composite=np.full_like(raw,128); composite[unknown]=raw[unknown]; composite[visible]=roles["original_crop_rgb"][visible]
    if not np.array_equal(composite[visible],roles["original_crop_rgb"][visible]): raise RuntimeError("Original visible RGB changed")
    fg=visible | (unknown & (np.abs(composite.astype(np.int16)-128).max(2)>24))
    labels,count=label(fg); ids=np.unique(labels[visible]); ids=ids[ids!=0]; fg=np.isin(labels,ids)
    silhouette=(binary_fill_holes(binary_closing(fg,structure=np.ones((5,5),bool))) & potential) | visible
    rgba=np.concatenate((composite,(silhouette.astype(np.uint8)*255)[...,None]),axis=2)
    rgba[~silhouette,:3]=0
    Image.fromarray(composite).save(out/"amodal_rgb.png")
    Image.fromarray(silhouette.astype(np.uint8)*255).save(out/"generated_silhouette.png")
    Image.fromarray(rgba).save(out/"amodal_rgba.png")
    unknown_image=np.zeros_like(composite); unknown_image[unknown]=255
    preview=np.concatenate((roles["original_crop_rgb"],rgb(out/"input512.png")[y0:y1,x0:x1],unknown_image,composite),axis=1)
    Image.fromarray(preview).save(out/"preview_original_input_mask_output.png")
    receipt=dict(schema="a20.amodal_semantic_condition_receipt.v1",seed=101,candidates=1,denoising_steps_requested=20,
        scheduler=type(pipe.scheduler).__name__,guidance_scale=3.,eta=0.,strength=1.,prompt=plan["prompt"],
        model=plan["model"],model_revision=plan["model_revision"],lora_loaded=False,
        dtype="bfloat16",model_wh=[512,512],train_raw=110,heldout_used=False,
        visible_pixels=int(visible.sum()),visible_RGB_max_abs_error_uint8=int(np.abs(composite[visible].astype(np.int16)-roles["original_crop_rgb"][visible]).max()),
        unknown_pixels=int(unknown.sum()),generated_silhouette_pixels=int(silhouette.sum()),
        generated_silhouette_unknown_pixels=int((silhouette & unknown).sum()),
        alpha_origin="generated estimated silhouette, not observed full silhouette",source_sha256=sha(__file__),
        all_unet_forward_finite=all(finite_calls),actual_unet_forward_calls=len(finite_calls),
        generation_seconds=generation_seconds,total_wall_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        resource_limits=dict(cuda_memory_fraction=.45,omp_threads=2,mkl_threads=2),
        artifacts={name:dict(path=str(out/name),sha256=sha(out/name)) for name in
            ["raw_model512.png","amodal_rgb.png","amodal_rgba.png","generated_silhouette.png","preview_original_input_mask_output.png"]},
        visual_review_required_before_omni=True)
    write(out/"receipt.json",receipt);print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--execute-gpu",action="store_true");args=parser.parse_args()
    out=ROOT/"semantic/sd15_seed101_r1"
    if args.prepare_only: prepare(out)
    elif args.execute_gpu: execute(out)
    else: raise SystemExit("Choose explicit CPU preparation or GPU execution")
