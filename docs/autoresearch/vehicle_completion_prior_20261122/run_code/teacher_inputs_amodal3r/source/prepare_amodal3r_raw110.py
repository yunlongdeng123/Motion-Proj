"""CPU-only source-grounded Amodal3R RGB/ternary-mask preparation."""
from pathlib import Path
import ast,json,hashlib,itertools,os
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFilter,ImageFont

ROOT=Path("/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1")
OUT=ROOT/"teacher_inputs_amodal3r/raw110"
OFFICIAL=ROOT/"teacher_inputs_amodal3r/source/amodal3r_image_to_3d.official.py"
REGISTRY=Path("/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/track_registry.json")
OFFICIAL_URL="https://github.com/Sm0kyWu/Amodal3R/blob/d00e083f224d0a4c32c1d3a92740ae57d53b58ca/amodal3r/pipelines/image_to_3d.py"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def binary(value):return Image.fromarray(np.uint8(value)*255)
def grow(value):return np.asarray(binary(value).filter(ImageFilter.MaxFilter(3)))>0
def shrink(value):return np.asarray(binary(value).filter(ImageFilter.MinFilter(3)))>0

def depth_interval(box,inv_camera):
    corners=np.asarray(list(itertools.product([-1,1],repeat=3)),float)*np.asarray(box["dimensions"])/2
    pose=np.asarray(box["world_from_actor"]).reshape(4,4)
    world=corners@pose[:3,:3].T+pose[:3,3]
    camera=world@inv_camera[:3,:3].T+inv_camera[:3,3]
    return [float(camera[:,0].min()),float(camera[:,0].max())]


def main():
    if OUT.exists():raise RuntimeError("Preserve existing teacher input directory")
    previous_mapping=ROOT/"conditioning/train/110/mapping.json"
    mapping=json.loads(previous_mapping.read_text());obs=json.loads(Path(mapping["source"]["observation"]).read_text())
    if obs["raw"]!=110:raise RuntimeError("Only raw110 is authorized for this packet")
    rgb=np.asarray(Image.open(mapping["source"]["rgb"]).convert("RGB"))
    labels=np.asarray(Image.open(mapping["source"]["labels"]))
    target=np.asarray(Image.open(mapping["source"]["target_mask"]))>0
    occluder=np.asarray(Image.open(mapping["source"]["occluder_mask"]))>0
    if labels.shape!=(320,480) or rgb.shape!=(1280,1920,3):raise RuntimeError("Source grids changed")
    if not np.array_equal(target,labels==21) or not np.array_equal(occluder,labels==10):raise RuntimeError("Accepted source identity mismatch")
    registry=json.loads(REGISTRY.read_text())["actors"];alias_track={v["alias"]:k for k,v in registry.items()}
    if alias_track["a20"]!=mapping["track"] or alias_track["a9"]!=mapping["occluder"]["track"]:raise RuntimeError("Track registry identity mismatch")
    boxes={b["track"]:b for b in obs["boxes"]}
    camera=np.asarray(obs["camera_pose_world"]).reshape(4,4)@np.asarray(obs["camera_extrinsic"]).reshape(4,4)
    inv=np.linalg.inv(camera);target_depth=depth_interval(boxes[mapping["track"]],inv)
    x0,y0,x1,y1=mapping["crop_native_xyxy_half_open"]
    crop_slice=np.s_[y0:y1,x0:x1]
    crop_labels=labels[y0//4:y1//4,x0//4:x1//4]
    foreground=np.zeros_like(target);background_actors=[];foreground_actors=[];unresolved=[]
    for identifier in sorted(set(np.unique(crop_labels).tolist())-{0,21}):
        alias="a"+str(identifier-1)
        alias_file=Path(mapping["source"]["labels"]).parent/(alias+".png")
        if alias not in alias_track or not alias_file.is_file():
            unresolved.append(dict(native_id=identifier,reason="missing identity binding"));continue
        own=np.asarray(Image.open(alias_file))>0
        if not np.array_equal(own,labels==identifier):
            unresolved.append(dict(native_id=identifier,reason="mask/native-label mismatch"));continue
        interval=depth_interval(boxes[alias_track[alias]],inv)
        row=dict(native_id=identifier,alias=alias,track=alias_track[alias],camera_forward_depth_interval_m=interval)
        if interval[1]<target_depth[0]:
            foreground|=own;foreground_actors.append(row)
        elif interval[0]>target_depth[1]:background_actors.append(row)
        else:unresolved.append(dict(**row,reason="overlapping OBB depth intervals do not prove foreground"))
    if unresolved:raise RuntimeError("Unresolved foreground semantics: "+json.dumps(unresolved))
    # Rebuild from accepted source masks; never load the previous whole-OBB unknown mask.
    core=shrink(target)&~grow(foreground)
    target_boundary=grow(target)&~shrink(target)
    foreground_guard=grow(foreground)
    up=lambda a:np.repeat(np.repeat(a,4,axis=0),4,axis=1)
    potential_image=Image.new("L",(1920,1280),0)
    ImageDraw.Draw(potential_image).polygon([tuple(p) for p in mapping["projected_obb_convex_hull_native"]],fill=255)
    potential=np.asarray(potential_image.filter(ImageFilter.MaxFilter(9)))>0
    potential|=up(target)
    scope=potential|up(grow(target))
    visible=up(core)
    black=(up(foreground_guard)|up(target_boundary))&scope&~visible
    ternary=np.full((1280,1920),255,np.uint8);ternary[black]=0;ternary[visible]=188
    # Preserve the complete potential actor crop and pad to a square without resizing.
    crop_rgb=rgb[crop_slice].copy();crop_mask=ternary[crop_slice].copy()
    height,width=crop_mask.shape;side=max(height,width);px=(side-width)//2;py=(side-height)//2
    teacher_rgb=np.zeros((side,side,3),np.uint8);teacher_mask=np.full((side,side),255,np.uint8)
    teacher_rgb[py:py+height,px:px+width]=crop_rgb;teacher_mask[py:py+height,px:px+width]=crop_mask
    OUT.mkdir(parents=True,exist_ok=False)
    Image.fromarray(teacher_rgb).save(OUT/"rgb.png");Image.fromarray(teacher_mask).save(OUT/"mask.png")
    Image.fromarray(crop_rgb).save(OUT/"rgb_source_crop.png");Image.fromarray(crop_mask).save(OUT/"mask_source_crop.png")
    for name,value in [("visible_core_source480",core),("actual_foreground_source480",foreground),("target_boundary_uncertain_source480",target_boundary)]:binary(value).save(OUT/(name+".png"))
    np.savez_compressed(OUT/"roles.npz",visible=teacher_mask==188,occluded_or_boundary_uncertain=teacher_mask==0,background=teacher_mask==255,
        original_crop_rgb=crop_rgb,actual_foreground_crop=up(foreground)[crop_slice],foreground_guard_crop=up(foreground_guard)[crop_slice],
        target_boundary_crop=up(target_boundary)[crop_slice],potential_extent_crop=scope[crop_slice],source_labels_crop=up(labels)[crop_slice])
    # Execute the exact official preprocessing method in isolation, without imports/model initialization.
    tree=ast.parse(OFFICIAL.read_text(encoding="utf-8-sig"));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Amodal3RImageTo3DPipeline")
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="preprocess_image_w_mask")
    module=ast.Module(body=[function],type_ignores=[]);namespace={"np":np,"cv2":cv2,"Image":Image}
    exec(compile(module,str(OFFICIAL),"exec"),namespace)
    effective_image,effective_hidden,effective_occ=namespace["preprocess_image_w_mask"](None,Image.fromarray(teacher_rgb),Image.fromarray(teacher_mask),kernel_size=3)
    effective_image.save(OUT/"official_preprocessed_rgb_518.png")
    Image.fromarray(np.uint8(effective_hidden)*255).save(OUT/"official_effective_hidden_mask_input_grid.png")
    Image.fromarray(np.uint8(effective_occ)*255).save(OUT/"official_effective_occlusion_mask_input_grid.png")
    source_visible=crop_mask==188
    if not np.array_equal(teacher_rgb[py:py+height,px:px+width][source_visible],crop_rgb[source_visible]):raise RuntimeError("Visible source RGB changed")
    if not np.isin(teacher_mask,[0,188,255]).all() or (visible&black).any():raise RuntimeError("Invalid ternary roles")
    outside_crop=scope.copy();outside_crop[crop_slice]=False
    if outside_crop.any():raise RuntimeError("Potential actor extent is truncated by crop")
    H=np.asarray([[1.,0.,x0-px],[0.,1.,y0-py],[0.,0.,1.]])
    s=518./side;pixel_map=np.asarray([[s,0.,s*(px-x0+.5)-.5],[0.,s,s*(py-y0+.5)-.5],[0.,0.,1.]])
    K=obs["camera_intrinsic_Brown"].copy();K[2]=K[2]-x0+px;K[3]=K[3]-y0+py
    counts=dict(visible=int((teacher_mask==188).sum()),occluded_or_uncertain=int((teacher_mask==0).sum()),background=int((teacher_mask==255).sum()),
        actual_foreground_crop_native=int(up(foreground)[crop_slice].sum()),black_on_actual_foreground=int((black&up(foreground))[crop_slice].sum()),
        black_boundary_or_guard_only=int((black&~up(foreground))[crop_slice].sum()),old_whole_OBB_unknown_count_from_mapping=mapping["counts"]["unknown_potential_crop_native"],
        old_whole_OBB_unknown_mask_loaded=False,effective_occlusion_after_official_kernel3=int(effective_occ.sum()))
    source_paths=[previous_mapping,Path(mapping["source"]["rgb"]),Path(mapping["source"]["target_mask"]),Path(mapping["source"]["occluder_mask"]),Path(mapping["source"]["labels"]),Path(mapping["source"]["observation"]),REGISTRY,OFFICIAL]
    out_mapping=dict(schema="a20.amodal3r_real_rgb_ternary_input.v1",raw=110,target_alias="a20",target_native_id=21,target_track=mapping["track"],
        teacher_api="pipeline.run_multi_image([Image.open(rgb_path)], [Image.open(mask_path).convert('L')], ...)",official_source_url=OFFICIAL_URL,
        role_values={"255":"background for target","188":"conservative visible target core","0":"actual foreground occluder or source boundary uncertainty"},
        official_preprocessing=dict(occluded_threshold="mask<127",background_threshold="mask>230",occlusion_dilation_kernel=3,RGB_hidden_and_background_zeroed=True,model_image_wh=[518,518],mask_patch_resize="torch F.interpolate nearest to518; patch14"),
        original_image_wh=[1920,1280],source_mask_wh=[480,320],crop_native_xyxy_half_open=[x0,y0,x1,y1],teacher_input_wh=[side,side],
        pad_left_top_right_bottom=[px,py,side-width-px,side-height-py],padding_RGB=[0,0,0],padding_mask_value=255,no_pre_model_resize=True,
        teacher_input_to_native_pixel_index_homography=H.tolist(),native_to_teacher_input_homography=np.linalg.inv(H).tolist(),native_pixel_index_to_model518_center_homography=pixel_map.tolist(),
        image_resize_rule="u_model=(u_input+0.5)*518/side-0.5; mask patch resize uses official torch nearest separately",
        intrinsics_Brown_teacher_input=K,camera_from_actor=mapping["camera_from_actor"],camera_to_world=mapping["camera_to_world"],world_from_actor=mapping["world_from_actor"],
        model_receives_camera_or_lidar=False,target_camera_forward_depth_interval_m=target_depth,actual_foreground_actors=foreground_actors,behind_target_actors_not_occluders=background_actors,
        construction="Recomputed visible core and occluder/uncertain boundary from accepted original source masks. Projected OBB only scopes possible actor extent and crop; it does not label all interior as unknown or identify hidden silhouette.",
        conservative_visible_rule="erode target/native21 agreement by1 source480 pixel; exclude1-pixel expanded actual foreground",
        occlusion_rule="actual closer actor mask expanded1 source480 pixel + target silhouette1-pixel uncertain band, clipped to projected potential extent plus observed boundary; visible takes priority",
        mask_is_full_hidden_shape_GT=False,source_masks_are_480_resolution_proxies=True,source_RGB_visible_exact=True,frames_read=[110],heldout_used=False,gpu_calls=0,model_inference_calls=0,weights_downloaded=0,
        counts=counts,sources=[dict(path=str(p),sha256=sha(p)) for p in source_paths],source_sha256=sha(__file__))
    write(OUT/"mapping.json",out_mapping)
    panels=[Image.fromarray(teacher_rgb),Image.fromarray(teacher_mask).convert("RGB"),effective_image.resize((side,side),Image.Resampling.LANCZOS)]
    titles=["Original RGB + square padding","Mask: V188 / occluder0 / B255","Official CPU preprocessing (518)"]
    preview=Image.new("RGB",(side*3,side+42),(18,22,30));draw=ImageDraw.Draw(preview)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    for i,panel in enumerate(panels):preview.paste(panel,(i*side,42));draw.text((i*side+10,12),titles[i],fill=(235,235,235),font=font)
    preview.save(OUT/"preview.png")
    verification=dict(status="CPU_INPUT_READY_NO_MODEL_INFERENCE",mask_values=sorted(np.unique(teacher_mask).tolist()),visible_source_rgb_max_abs_uint8=0,
        V_intersect_occlusion_pixels=0,full_potential_extent_inside_crop=True,source_boundary_uncertainty_not_hidden_GT=True,unresolved_actor_semantics=[],counts=counts,
        no_previous_unknown_mask_loaded=True,no_native_GS_used=True,official_preprocess_method_executed_in_isolation=True,frames_read=[110],gpu_calls=0,model_calls=0,
        files={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in ["rgb.png","mask.png","mapping.json","roles.npz","preview.png","official_preprocessed_rgb_518.png"]})
    write(OUT/"verification.json",verification);print(json.dumps(verification),flush=True)

if __name__=="__main__":main()
