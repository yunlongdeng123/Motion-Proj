"""Single frozen SD1.5 same-view inpaint with native observed RGB hard protection."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/a20_sd15_sameview_20261122_r1")
OLD = Path("/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1")
MODEL = Path("/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
TRACK = "bzFbWRrSGOgEaXxnuGV0tw"
PROMPT = "rear view same black Dodge pickup truck, complete rear bumper and rear tires, coherent full vehicle body, realistic proportions, plain neutral background"

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""): digest.update(block)
    return digest.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def read(path): return json.loads(Path(path).read_text())
def rgb(path): return np.asarray(Image.open(path).convert("RGB")).copy()
def mask(path): return np.asarray(Image.open(path).convert("L")) > 0
def save_mask(path, value): Image.fromarray(value.astype(np.uint8)*255).save(path)

def panel(path, tiles):
    width = max(im.width for _, im in tiles)
    height = max(im.height for _, im in tiles)
    sheet = Image.new("RGB", (width*len(tiles), height+40), (20,24,32))
    draw = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(tiles):
        draw.text((i*width+8,12), title, fill=(236,240,248))
        if im.mode == "RGBA":
            under = Image.new("RGBA", im.size, (128,128,128,255)); under.alpha_composite(im); im=under.convert("RGB")
        sheet.paste(im.convert("RGB"),(i*width+(width-im.width)//2,40))
    sheet.save(path)

def architecture(path):
    im=Image.new("RGB", (1400,220), (18,22,30)); d=ImageDraw.Draw(im)
    boxes=[(15,60,250,155,"raw110 native RGB + masks"),
           (300,60,560,155,"V/R protected | U=Oguard&P"),
           (610,60,850,155,"Frozen SD1.5 inpaint\nDDIM20 | seed101"),
           (900,60,1130,155,"U generated + V/R hard-copy"),
           (1180,60,1380,155,"Same-view candidate\nEstimated alpha")]
    for x0,y0,x1,y1,t in boxes:
        d.rounded_rectangle((x0,y0,x1,y1),radius=12,fill=(35,45,63),outline=(99,140,185),width=2)
        d.multiline_text((x0+12,y0+25),t,fill=(236,241,249),spacing=8)
    for left,right in zip(boxes,boxes[1:]):
        x0,x1=left[2]+5,right[0]-8;y=107;d.line((x0,y,x1,y),fill=(75,211,176),width=4)
        d.polygon([(x1,y),(x1-10,y-6),(x1-10,y+6)],fill=(75,211,176))
    d.text((20,185),"V: observed core | R: approximate visible edge | U: occluded completion | P: OBB potential support only | N/B: locked neutral",fill=(178,192,212))
    im.save(path)

def verify_native(raw, out):
    source=OLD/"conditioning/train"/f"{raw:03d}"
    mapping=read(source/"mapping.json")
    assert mapping["track"]==TRACK and mapping["native_instance_label"]==21
    assert mapping["occluder"]["native_instance_label"]==10
    x0,y0,x1,y1=mapping["crop_native_xyxy_half_open"]
    original=rgb(mapping["source"]["rgb"])
    target=mask(mapping["source"]["target_mask"])
    occluder=mask(mapping["source"]["occluder_mask"])
    labels=np.asarray(Image.open(mapping["source"]["labels"]))
    assert target.shape==occluder.shape==labels.shape==(320,480)
    assert not (target & occluder).any()
    assert (labels[target]==21).all() and (labels[occluder]==10).all()
    core=np.asarray(Image.fromarray((target & (labels==21)).astype(np.uint8)*255).filter(ImageFilter.MinFilter(3)))>0
    guard=np.asarray(Image.fromarray((occluder | (labels==10)).astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3)))>0
    resize=lambda v: np.asarray(Image.fromarray(v.astype(np.uint8)*255).resize(tuple(mapping["image_native_size"]),Image.Resampling.NEAREST))>0
    V=resize(core & ~guard)[y0:y1,x0:x1]
    O=resize(occluder)[y0:y1,x0:x1]
    G=resize(guard)[y0:y1,x0:x1]
    P=mask(source/"amodal_box_potential_mask.png")
    U=P & ~V;B=~P
    image=original[y0:y1,x0:x1].copy()
    assert np.array_equal(image,rgb(source/"context_rgb.png"))
    for filename,v in [("visible_mask.png",V),("occluder_mask.png",O),("occluder_guard_mask.png",G),("unknown_potential_mask.png",U),("known_background_mask.png",B)]:
        assert np.array_equal(v,mask(source/filename)), filename
    assert not (V & G).any() and not (V & U).any()
    assert np.array_equal(V | U | B,np.ones(V.shape,bool))
    assert np.array_equal(image[V],rgb(source/"partial_visible_rgb.png")[V])
    out.mkdir(parents=True,exist_ok=False)
    for filename in ["context_rgb.png","partial_visible_rgb.png","partial_visible_rgba.png","visible_mask.png","visible_mask_approx.png","occluder_mask.png","occluder_guard_mask.png","unknown_potential_mask.png","known_background_mask.png","amodal_box_potential_mask.png","mapping.json","a20_mask_native480_crop.png","a9_mask_native480_crop.png","labels_native480_crop.png"]:
        shutil.copy2(source/filename,out/filename)
    report=dict(raw=raw,role="build_main" if raw==110 else "observation_diagnostic_only",track=TRACK,native_label=21,
        crop_xyxy=mapping["crop_native_xyxy_half_open"],rgb_crop_max_abs_error=0,
        source_mask_wh=[480,320],native_image_wh=mapping["image_native_size"],mask_resampling="nearest; source labels only, not new annotation",
        V_count=int(V.sum()),O_count=int(O.sum()),U_count=int(U.sum()),B_count=int(B.sum()),
        O_in_U_count=int((O & U).sum()),V_intersect_O_guard_count=0,V_rgb_matches_native=True,
        source_approx_target_mask_count=int(target.sum()),source_core_count=int((core & ~guard).sum()),
        potential_origin="Existing fixed OBB projected support + 4 native px guard, union observed approximate mask; candidate support only",
        O_role="Occluding a9 white Mazda, native10. O can overlap U; O RGB never conditions diffusion.",
        domain_contract="V,U,B partition the crop; O is an independent observed occluder mask. U=P minus V, not full-object truth.",
        original_source_pins=[dict(path=mapping["source"][key],sha256=sha(mapping["source"][key])) for key in ["rgb","target_mask","occluder_mask","labels","observation"]],
        old_condition_pins=[dict(path=str(p),sha256=sha(p)) for p in sorted(source.glob("*")) if p.is_file()])
    write(out/"native_validation.json",report)
    return image,V,U,P,B,O,G,report

def prepare():
    out=ROOT/"sd15_seed101_occlusion_r2"
    if out.exists(): raise RuntimeError("Preserve fixed prepared plan; no overwrite")
    image,V,Q,P,B,O,G,validation=verify_native(110,ROOT/"conditioning_occlusion_r2/raw110")
    diagnostic=verify_native(86,ROOT/"diagnostic_occlusion_r2/raw086")[-1]
    source=ROOT/"conditioning_occlusion_r2/raw110"
    # Only observed a20/native21 source pixels outside the expanded a9 boundary are retained.
    R=mask(source/"visible_mask_approx.png") & ~G & ~V
    protected=V | R
    U=P & G & ~protected
    N=P & ~protected & ~U
    assert not (protected & G).any() and not (U & protected).any()
    assert np.array_equal(protected | U | N | B,np.ones(V.shape,bool))
    assert not (R & O).any()
    assert image.shape==(364,432,3)
    image512=np.full((512,512,3),128,np.uint8);v512=np.zeros((512,512),bool);u512=v512.copy()
    x0,y0=40,74;rect=np.s_[y0:y0+364,x0:x0+432]
    image512[rect][protected]=image[protected];v512[rect]=protected;u512[rect]=U
    assert np.array_equal(image512[rect][protected],image[protected])
    assert np.all(image512[~v512]==128)
    from transformers import CLIPTokenizer
    tokens=CLIPTokenizer.from_pretrained(str(MODEL/"tokenizer"),local_files_only=True).encode(PROMPT,add_special_tokens=True,truncation=False)
    assert len(tokens)<=77 and read(MODEL/"unet/config.json")["in_channels"]==9
    weights=[MODEL/p for p in ["unet/diffusion_pytorch_model.fp16.safetensors","vae/diffusion_pytorch_model.fp16.safetensors","text_encoder/model.fp16.safetensors"]]
    assert all(p.is_file() for p in weights)
    out.mkdir(parents=True,exist_ok=False)
    Image.fromarray(image512).save(out/"input512.png");save_mask(out/"inpaint_U_mask512.png",u512)
    np.savez_compressed(out/"roles.npz",original_crop_rgb=image,V=V,R=R,protected=protected,U=U,Q=Q,P=P,B=B,N=N,O=O,O_guard=G,V512=v512,U512=u512)
    for name,role in [("V_trusted_core",V),("R_observed_approx_boundary",R),("U_inpaint_permitted",U),("N_nonoccluded_potential_locked",N),("B_outside_potential",B),("O_observed_occluder",O),("O_guard",G)]:save_mask(out/(name+".png"),role)
    domain=np.full_like(image,(23,30,40));domain[N]=(70,75,85);domain[U]=(219,161,49);domain[O & U]=(214,80,164);domain[R]=(20,140,125);domain[V]=(23,214,155)
    Image.fromarray(domain).save(out/"domains_V_green_R_teal_O_magenta_U_amber_N_gray_B_dark.png")
    panel(out/"preview_input_mask_domains.png",[("raw110 original crop",Image.fromarray(image)),("V+R original RGB, neutral rest",Image.fromarray(image512[rect])),("U=Oguard&P only: white",Image.fromarray(U.astype(np.uint8)*255)),("V green | R teal | O magenta",Image.fromarray(domain))])
    architecture(ROOT/"architecture_components_occlusion_r2.png")
    plan=dict(schema="a20.sd15_sameview_occlusion.plan.v2",training_raw=110,diagnostic_raw=86,diagnostic_conditions_model=False,
        native_track=TRACK,native_instance_label=21,occluder_native_label=10,
        model=str(MODEL),model_revision=MODEL.name,model_type="base StableDiffusionInpaintPipeline, 9-channel UNet",
        prompt=PROMPT,negative_prompt="",seed=101,scheduler="DDIM",inference_steps=20,guidance_scale=3.,eta=0.,strength=1.,
        dtype="bfloat16",weight_variant="fp16",use_lora=False,weights_frozen=True,candidate_count=1,prompt_seed_sweep=False,training_updates=0,
        model_wh=[512,512],native_crop_wh=[432,364],crop_xyxy_in512=[40,74,472,438],rgb_resize_used=False,
        model_preprocessing="PIL RGB 432x364 native crop, center pad to512. Diffusers image_processor receives512x512 and keeps512; VAE encodes spatially downsampled latent at64x64. Decoder RGB is provisional until V hard-copy.",
        mask_preprocessing="480x320 categorical masks nearest-expand to1920x1280 then crop. Model mask remains512x512; pipeline nearest-downsamples mask for latent grid.",
        neutral_rgb=[128,128,128],trusted_V_pixels=int(V.sum()),observed_boundary_R_pixels=int(R.sum()),inpaint_U_pixels=int(U.sum()),original_potential_Q_pixels=int(Q.sum()),locked_nonoccluded_potential_N_pixels=int(N.sum()),known_B_pixels=int(B.sum()),observed_O_pixels=int(O.sum()),prompt_tokens=len(tokens),
        role_contract="V=conservative source21 core; R=source a20/native21 approximate visible ring outside a9 guard, retained conservatively as separate observed edge; O=source a9/native10; U=P & O_guard & ~(V|R); B=outside P; N=P minus V/R/U stays neutral. P is projected OBB support, not silhouette truth.",
        inpaint_permission="U is O_guard intersect P only, excluding every observed protected pixel. O_guard is a9/native10 union dilated one 480-mask pixel (4 native px) for boundary join. No other OBB-only unknown area is generated.",
        visible_policy="Only original raw110 RGB on V|R enters the model; a9 guard excluded. After diffusion hard-copy V|R exactly, use output only in U, force B|N neutral128. V and approximate observed R remain separately identified.",
        alpha_policy="Generated estimate in U: >24 RGB max-distance from neutral, union V|R; retain protected-connected components;5px close and hole fill; intersect V|R|U, union V|R. Alpha is separate from observed V and R.",
        role_validation=validation,diagnostic_validation=diagnostic,
        input_pins=[dict(path=str(p),sha256=sha(p)) for p in sorted(source.glob("*")) if p.is_file()],
        prepared_pins=[dict(path=str(out/p),sha256=sha(out/p)) for p in ["input512.png","inpaint_U_mask512.png","roles.npz"]],
        base_weight_pins=[dict(path=str(p),sha256=sha(p)) for p in weights],source_sha256=sha(__file__),gpu_preparation_used=False,
        next_stage="Stop after 2D candidate and await root actual image review; no Omni, texturing, or student execution")
    write(out/"plan.json",plan)
    print(json.dumps(dict(status="PREPARED",plan=str(out/"plan.json"),preview=str(out/"preview_input_mask_domains.png"),visible_pixels=int(V.sum()),U_pixels=int(U.sum()),raw86_diagnostic_only=True)),flush=True)

def execute():
    out=ROOT/"sd15_seed101_occlusion_r2"
    if (out/"raw_model512.png").exists() or (out/"launch_receipt.json").exists(): raise RuntimeError("Single fixed candidate already launched")
    plan=read(out/"plan.json")
    assert sha(__file__)==plan["source_sha256"]
    for pin in plan["input_pins"]+plan["prepared_pins"]+plan["base_weight_pins"]:
        assert sha(pin["path"])==pin["sha256"],pin["path"]
    import subprocess
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from scipy.ndimage import label,binary_closing,binary_fill_holes
    other=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
    if other: raise RuntimeError("Unexpected active GPU process; preserve other tasks: "+other)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.45,0)
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(101);np.random.seed(101)
    started=time.monotonic()
    write(out/"launch_receipt.json",dict(pid=os.getpid(),started_unix=time.time(),source_sha256=sha(__file__),single_candidate=True,prompt_seed_sweep=False,training_updates=0,other_gpu_processes=other))
    pipe=StableDiffusionInpaintPipeline.from_pretrained(plan["model"],variant="fp16",torch_dtype=torch.bfloat16,safety_checker=None,requires_safety_checker=False,local_files_only=True,use_safetensors=True)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    for module in (pipe.unet,pipe.vae,pipe.text_encoder): module.eval().requires_grad_(False)
    assert not getattr(pipe.unet,"peft_config",None)
    assert sum(p.numel() for m in (pipe.unet,pipe.vae,pipe.text_encoder) for p in m.parameters() if p.requires_grad)==0
    pipe.to("cuda")
    input_image=Image.open(out/"input512.png").convert("RGB");input_mask=Image.open(out/"inpaint_U_mask512.png").convert("L")
    preprocessed=pipe.image_processor.preprocess(input_image,height=512,width=512)
    assert tuple(preprocessed.shape)==(1,3,512,512)
    finite_calls=[]
    def forward_hook(module,inputs,result):
        value=result.sample if hasattr(result,"sample") else result[0]
        finite=bool(torch.isfinite(value).all().item());finite_calls.append(finite)
        assert finite,"Nonfinite inpaint forward"
    pipe.unet.register_forward_hook(forward_hook)
    generation_started=time.monotonic()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        generated=pipe(prompt=plan["prompt"],negative_prompt="",image=input_image,mask_image=input_mask,
            height=512,width=512,strength=1.,num_inference_steps=20,guidance_scale=3.,eta=0.,generator=torch.Generator("cuda").manual_seed(101)).images[0]
    torch.cuda.synchronize();generation_seconds=time.monotonic()-generation_started
    generated.save(out/"raw_model512.png")
    with np.load(out/"roles.npz",allow_pickle=False) as bank:r={k:bank[k].copy() for k in bank.files}
    raw=np.asarray(generated)[74:438,40:472].copy();V,U,P,B,O,R,protected,N=r["V"],r["U"],r["P"],r["B"],r["O"],r["R"],r["protected"],r["N"]
    composite=np.full_like(raw,128);composite[U]=raw[U];composite[protected]=r["original_crop_rgb"][protected]
    diff=np.abs(composite[V].astype(np.int16)-r["original_crop_rgb"][V].astype(np.int16))
    assert diff.max()==0 and np.all(composite[B | N]==128)
    assert np.array_equal(composite[R],r["original_crop_rgb"][R])
    assert not (V & O).any()
    fg=protected | (U & (np.abs(composite.astype(np.int16)-128).max(2)>24))
    labels,count=label(fg);ids=np.unique(labels[protected]);ids=ids[ids!=0];fg=np.isin(labels,ids)
    alpha=(binary_fill_holes(binary_closing(fg,structure=np.ones((5,5),bool))) & (protected | U)) | protected
    rgba=np.dstack([composite,alpha.astype(np.uint8)*255]);rgba[~alpha,:3]=0
    Image.fromarray(composite).save(out/"a20_sameview_rgb.png");Image.fromarray(rgba).save(out/"a20_sameview_rgba_estimated.png")
    save_mask(out/"generated_alpha_estimate.png",alpha);save_mask(out/"observed_V_mask.png",V);save_mask(out/"observed_R_boundary_mask.png",R);save_mask(out/"generated_alpha_U_only.png",alpha & U)
    labels_alpha=np.zeros_like(composite);labels_alpha[alpha & U]=(219,161,49);labels_alpha[R]=(20,140,125);labels_alpha[V]=(23,214,155)
    Image.fromarray(labels_alpha).save(out/"alpha_origin_V_green_generated_amber.png")
    panel(out/"preview_original_input_mask_output.png",[("raw110 native RGB crop",Image.fromarray(r["original_crop_rgb"])),("SD input: observed V+R",input_image.crop((40,74,472,438))),("Inpaint occlusion U white",Image.fromarray(U.astype(np.uint8)*255)),("Same view: U gen + V/R exact",Image.fromarray(composite))])
    panel(out/"preview_candidate_alpha_origin.png",[("Candidate RGB",Image.fromarray(composite)),("Estimated RGBA on gray",Image.fromarray(rgba)),("V green / R teal / gen amber",Image.fromarray(labels_alpha))])
    artifacts=["raw_model512.png","a20_sameview_rgb.png","a20_sameview_rgba_estimated.png","generated_alpha_estimate.png","observed_V_mask.png","observed_R_boundary_mask.png","generated_alpha_U_only.png","alpha_origin_V_green_generated_amber.png","preview_original_input_mask_output.png","preview_candidate_alpha_origin.png"]
    receipt=dict(schema="a20.sd15_sameview_occlusion.receipt.v2",status="2D_CANDIDATE_PENDING_ROOT_IMAGE_REVIEW",seed=101,candidates=1,denoising_steps=20,actual_unet_forward_calls=len(finite_calls),all_unet_forward_finite=all(finite_calls),
        scheduler=type(pipe.scheduler).__name__,guidance_scale=3.,eta=0.,strength=1.,prompt=plan["prompt"],negative_prompt="",model=plan["model"],model_revision=plan["model_revision"],lora_loaded=False,weights_frozen=True,trainable_parameters=0,training_updates=0,
        dtype="bfloat16",input_tensor_shape=list(preprocessed.shape),model_wh=[512,512],native_crop_wh=[432,364],rgb_resize_used=False,mask_source_resolution=[480,320],train_raw=110,raw86_diagnostic_only=True,
        visible_pixels=int(V.sum()),visible_RGB_max_abs_error_uint8=int(diff.max()),visible_RGB_mean_abs_error_uint8=float(diff.mean()),visible_RGB_changed_pixel_count=int((diff>0).any(1).sum()),V_intersect_O_pixels=0,
        approximate_observed_R_pixels=int(R.sum()),R_RGB_max_abs_error_uint8=0,background_max_abs_difference_from_neutral=0,locked_N_pixels=int(N.sum()),U_pixels=int(U.sum()),generated_alpha_pixels=int(alpha.sum()),generated_alpha_in_U_pixels=int((alpha & U).sum()),alpha_origin="generated image-derived silhouette estimate; observed V and approximate R saved separately",
        generation_seconds=generation_seconds,total_wall_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,resource_limits=dict(cuda_memory_fraction=.45,omp_threads=2),source_sha256=sha(__file__),
        artifacts={n:dict(path=str(out/n),sha256=sha(out/n)) for n in artifacts},next_stage="Stop and await root actual image review. No Omni, texturing, student, or shutdown executed.")
    write(out/"receipt.json",receipt)
    print(json.dumps(receipt),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--execute-gpu",action="store_true");args=parser.parse_args()
    if args.prepare_only:prepare()
    elif args.execute_gpu:execute()
    else:raise SystemExit("Choose explicit CPU preparation or GPU execution")
