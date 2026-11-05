"""One frozen offline SAM2 car matte; fixed box+known-upper positive points.

No grid, no negative-point tuning, no auto-multimask rescue, no downloads.
Raw SAM mask and mandatory-known union are both retained for parent review.
"""
import os
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
DEST=RUN.parent/'scene008_a6_insert_artifacts_20261105/vehicle/matte_contract'
RGB=RUN/'inputs/raw106/strong_candidate_v2_512.png'
SAMROOT=Path('/root/autodl-tmp/third_party/worldsim_v32/sam2')
CKPT=SAMROOT/'checkpoints/sam2.1_hiera_large.pt'
CONFIG='configs/sam2.1/sam2.1_hiera_l.yaml'
sys.path.insert(0,str(SAMROOT))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


@torch.inference_mode()
def main():
    DEST.mkdir(exist_ok=False);start=time.time()
    image=np.asarray(Image.open(RGB).convert('RGB'))
    assert image.shape==(512,512,3)
    source_hash=sha(RGB);weight_hash=sha(CKPT)
    pvd=np.asarray(Image.open(RUN/'inputs/raw106/object_mask_512.png'))>0
    known=np.asarray(Image.open(RUN/'inputs/raw106/known_mask_512.png'))>0
    y,x=np.where(pvd);box=np.array([max(0,x.min()-2),max(0,y.min()-2),min(511,x.max()+2),min(511,y.max()+2)],np.float32)
    ky,kx=np.where(known);anchors=np.column_stack((np.quantile(kx,[.25,.5,.75]),np.full(3,np.median(ky))))
    coords=np.column_stack((kx,ky))
    points=np.array([coords[np.square(coords-anchor).sum(1).argmin()] for anchor in anchors],np.float32)
    labels=np.ones(len(points),np.int32)
    torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=build_sam2(CONFIG,str(CKPT),device='cuda',mode='eval',apply_postprocessing=False)
    model.requires_grad_(False)
    predictor=SAM2ImagePredictor(model,mask_threshold=0.0,max_hole_area=0,max_sprinkle_area=0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(image)
        logits,scores,_=predictor.predict(point_coords=points,point_labels=labels,box=box,
            multimask_output=False,return_logits=True)
    assert len(logits)==1 and np.isfinite(logits).all()
    raw=logits[0]>0
    # This union is the explicitly required measured-known protection, not a
    # brightness-based wheel deletion or an alternate SAM candidate selection.
    matte=raw|known
    alpha=matte.astype(np.float32)
    dark=image.max(2)<16
    protected_dark=matte&dark
    excluded_canvas=pvd&~matte&dark
    shutil.copyfile(RGB,DEST/'teacher_rgb_512.png')
    assert np.array_equal(np.asarray(Image.open(DEST/'teacher_rgb_512.png')),image)
    Image.fromarray(raw.astype(np.uint8)*255).save(DEST/'sam_raw_mask_512.png')
    Image.fromarray(matte.astype(np.uint8)*255).save(DEST/'car_matte_512.png')
    Image.fromarray(alpha).save(DEST/'car_alpha_float.tiff')
    np.savez_compressed(DEST/'matte_contract.npz',alpha=alpha,foreground=matte,raw_sam_mask=raw,
        known_protected_mask=known,protected_dark_foreground=protected_dark,
        excluded_PVD_dark_canvas=excluded_canvas,box_xyxy=box,positive_points_xy=points,
        source_RGB_sha256=np.asarray(source_hash),sam_logits=logits[0].astype(np.float32))
    overlay=image.astype(np.float32)
    edge=matte&~binary_erosion(matte)
    p_edge=pvd&~binary_erosion(pvd)
    overlay[p_edge]=[255,40,40];overlay[edge]=[0,255,60]
    Image.fromarray(np.uint8(overlay)).save(DEST/'mask_overlay_512.png')
    protect=image.copy();protect[protected_dark]=[0,220,255];protect[excluded_canvas]=[255,0,190]
    Image.fromarray(protect).save(DEST/'black_tire_protection_512.png')
    prompt=Image.fromarray(image.copy());draw=ImageDraw.Draw(prompt)
    draw.rectangle(tuple(box),outline=(255,255,0),width=2)
    for px,py in points:draw.ellipse((px-3,py-3,px+3,py+3),fill=(0,255,0))
    prompt.save(DEST/'single_prompt_512.png')
    panel=Image.new('RGB',(512*4,542),(237,243,247));dr=ImageDraw.Draw(panel)
    for i,(name,title) in enumerate([('teacher_rgb_512.png','same strongv2 RGB'),('car_matte_512.png','single SAM2 matte +known'),('mask_overlay_512.png','green matte /red PVD'),('black_tire_protection_512.png','cyan darkFG /pink excludedBG')]):
        panel.paste(Image.open(DEST/name).convert('RGB'),(i*512,30));dr.text((i*512+8,8),title,fill=(12,25,38))
    panel.save(DEST/'matte_review_panel.png')
    assert sha(RGB)==source_hash and sha(CKPT)==weight_hash
    receipt=dict(status='SINGLE_FROZEN_SAM2_RESULT_PENDING_PARENT_VISUAL_QA',model='SAM2.1 hiera large',
        checkpoint=str(CKPT),checkpoint_sha256=weight_hash,config=CONFIG,env_python=sys.executable,
        source_RGB=str(RGB),source_RGB_sha256=source_hash,
        source_weights_unchanged=True,source_RGB_unchanged=True,downloads=0,model_training_updates=0,
        single_predict_call=True,multimask_output=False,dynamic_multimask_postprocessing=False,
        custom_hole_fill=False,prompt_box=box.tolist(),positive_known_points=points.tolist(),
        sam_quality_score=float(scores[0]),raw_mask_pixels=int(raw.sum()),matte_pixels=int(matte.sum()),
        known_protection_pixels=int(known.sum()),known_pixels_added_to_SAM=int((known&~raw).sum()),
        dark_foreground_pixels_protected=int(protected_dark.sum()),
        old_PVD_mask_dark_canvas_excluded=int(excluded_canvas.sum()),
        old_PVD_mask_all_pixels_excluded=int((pvd&~matte).sum()),
        matte_outside_PVD_pixels=int((matte&~pvd).sum()),
        alpha_definition='Binary candidate-specific foreground; texture filtering yields fractional coverage. SAM logits are retained but not treated as physical opacity.',
        colour_sampling_contract='Bilinear premultiplied_RGB=RGB*alpha and alpha separately, then unpremultiply only alpha>epsilon; real/observed rows protected.',
        dark_RGB_rule='Luminance is only used to DISPLAY preservation/exclusion, never to decide the SAM foreground or delete wheels.',
        seconds=time.time()-start,peak_vram_bytes=int(torch.cuda.max_memory_allocated()))
    write(DEST/'matte_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
