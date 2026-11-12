#!/usr/bin/env python3
"""A/B/C/D 冻结检索控制：同源二维特征与相机可见融合世界特征。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

if __package__:
    from .prepare_models import import_superpoint
else:
    from prepare_models import import_superpoint


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def resolve(root, name):
    path = Path(name)
    return path if path.is_absolute() else root/path


def load_module(path):
    spec = importlib.util.spec_from_file_location("world_feature_field", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_sources(root, manifest):
    value = json.loads(resolve(root, manifest["sources"]).read_text())
    return value["sources"] if isinstance(value, dict) else value


def audit_encoder_fine_holes(root,manifest,sources,args):
    """独立检查真实fine texel四个双线性邻像素，不能用coarse中心审计替代。"""
    wf=load_module(args.world_module);points=[]
    for record in manifest["samples"]:
        with np.load(resolve(root,record["tile_npz"])) as packed:
            points.append(packed["world_xyz"][packed["hole"].astype(bool)])
    points=np.concatenate(points);records=[];violations=0;rgb_violations=0;started=time.monotonic()
    for source in sources:
        allowed=np.asarray(Image.open(resolve(root,source["feature_allowed_mask"])).convert("L"))>127
        image=np.asarray(Image.open(resolve(root,source["feature_input"])).convert("RGB"))
        nh,nw=source["image_hw"]
        assert allowed.shape==(nh,nw) and image.shape==(nh,nw,3)
        uv,depth,inside=wf.project_world_to_camera(points,source["camera_K_native"],source["world_to_camera"],source["image_hw"],source["distortion_Brown"])
        uv=uv[inside];low=np.floor(uv-.5).astype(np.int64)
        any_allowed=np.zeros(len(uv),bool);any_rgb=np.zeros(len(uv),bool)
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            pixel=np.clip(low+[dx,dy],[0,0],[nw-1,nh-1]);x,y=pixel.T
            any_allowed |= allowed[y,x]
            any_rgb |= (image[y,x]!=128).any(1)
        n=int(any_allowed.sum());r=int(any_rgb.sum());violations+=n;rgb_violations+=r
        records.append(dict(raw=source["raw"],projected_fine_query_texels=len(uv),
            allowed_bilinear_neighbor_violations=n,masked_RGB_neighbor_violations=r,
            violating_uv_native_first5=uv[any_allowed|any_rgb][:5].tolist()))
    receipt=dict(status="PASS" if violations==0 and rgb_violations==0 else "FAIL",fine_query_texels=len(points),
        sources_checked=len(sources),allowed_violations=violations,masked_RGB_violations=rgb_violations,
        test="all fine world query texels, all four native-image bilinear support neighbors forbidden and RGB128 before encoder",
        target_RGB_read=False,seconds=time.monotonic()-started,sources=records)
    write_json(args.output/"encoder_fine_hole_audit.json",receipt)
    if receipt["status"]!="PASS":raise RuntimeError(f"fine128 encoder leakage audit failed: allowed={violations},RGB={rgb_violations}")
    print("FINE_ENCODER_AUDIT_PASS",len(points),len(sources),flush=True)
    return receipt


def extract(root, manifest, sources, args):
    out = args.output/"source_features"; out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/"extraction_receipt.json"
    if receipt_path.exists() and not args.reextract:
        receipt = json.loads(receipt_path.read_text())
        assert receipt["max_side"] == args.max_side
        assert receipt["source_raws"] == [s["raw"] for s in sources]
        assert receipt["dataset_path"] == str(args.dataset)
        return receipt
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.dino, local_files_only=True).eval().requires_grad_(False).to(args.device)
    sp = import_superpoint()({"max_keypoints":1024,"weights_path":args.superpoint_weights}).eval().requires_grad_(False).to(args.device)
    mean = torch.tensor([.485,.456,.406], device=args.device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=args.device)[None,:,None,None]
    started = time.monotonic(); started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    records = []
    for i, source in enumerate(sources):
        input_path = resolve(root, source["feature_input"])
        allowed_path = resolve(root, source["feature_allowed_mask"])
        # 原始RGB仅由数据构建器读取。encoder入口必须是全部查询hole已移除的输入。
        image = np.asarray(Image.open(input_path).convert("RGB")).copy()
        allowed = np.asarray(Image.open(allowed_path).convert("L")) > 127
        assert image.shape[:2] == allowed.shape
        assert image.shape[:2] == tuple(source["image_hw"]), "feature_input必须保留原相机像素尺度，禁止静默混用uv"
        image[~allowed] = 128
        nh,nw = image.shape[:2]
        scale = args.max_side/max(nh,nw)
        width = max(16,int(nw*scale)//16*16); height = max(16,int(nh*scale)//16*16)
        resized = np.asarray(Image.fromarray(image).resize((width,height),Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(resized).permute(2,0,1).float()[None].to(args.device)/255
        with torch.inference_mode():
            output = model(pixel_values=(tensor-mean)/std).last_hidden_state
            skip = 1+model.config.num_register_tokens
            dense = torch.nn.functional.normalize(output[:,skip:],dim=-1)
            dense = dense.reshape(height//16,width//16,-1).cpu().numpy().astype(np.float16)
            gray = .2989*tensor[:,0:1]+.5870*tensor[:,1:2]+.1140*tensor[:,2:3]
            sparse = sp({"image":gray})
        assert np.isfinite(dense).all()
        xy = sparse["keypoints"][0].cpu().numpy()
        native_xy = (xy+.5)*[nw/width,nh/height]
        nearest = np.clip(np.floor(native_xy).astype(int),[0,0],[nw-1,nh-1])
        good = allowed[nearest[:,1],nearest[:,0]]
        desc = sparse["descriptors"][0].cpu().numpy().T[good]
        scores = sparse["scores"][0].cpu().numpy()[good]
        raw = source["raw"]
        feature_path = out/f"{int(raw):03d}_dinov3.npy"
        sparse_path = out/f"{int(raw):03d}_superpoint.npz"
        np.save(feature_path,dense)
        np.savez_compressed(sparse_path,uv_native=native_xy[good].astype(np.float32),descriptors=desc.astype(np.float32),scores=scores)
        rec = dict(raw=raw,feature_path=str(feature_path),superpoint_path=str(sparse_path),input_hw=[height,width],
                   native_hw=[nh,nw],feature_shape=list(dense.shape),superpoint_keypoints=int(good.sum()),
                   input_path=str(input_path),allowed_mask_path=str(allowed_path),mask_policy=source["mask_policy"])
        records.append(rec)
        print(f"EXTRACT {i+1}/{len(sources)} raw={raw} tokens={height//16*width//16} kp={good.sum()} seconds={time.monotonic()-started:.1f}",flush=True)
    receipt = dict(schema="road_native_feature_extraction.v1",dataset_path=str(args.dataset),source_raws=[s["raw"] for s in sources],
                   max_side=args.max_side,device=args.device,model=args.dino,model_type=model.config.model_type,
                   frozen=True,parameter_updates=0,seconds=time.monotonic()-started,started_at_utc=started_utc,
                   ended_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                   shared_gpu_context=args.shared_gpu_context,
                   peak_torch_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
                   same_extraction_used_by=["C_native_dinov3_2d","D_native_dinov3_world"],
                   all_hidden_world_positions_masked_before_encoder=True,
                   superpoint_parameters={"max_keypoints_per_source":1024,"nms_radius":4,"keypoint_threshold":.005},
                   dino_parameters={"patch_size":16,"class_and_register_tokens_excluded":5,"per_token_l2_normalization":True,"storage_dtype":"float16"},
                   dino_feature_total_tokens=sum(np.prod(r["feature_shape"][:2]).item() for r in records),sources=records)
    write_json(receipt_path,receipt)
    del model,sp
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return receipt


def load_query(root, record):
    with np.load(resolve(root,record["tile_npz"])) as source:
        # target 在排序完成前不载入；候选生成也不读取任何 supervision 文件。
        tile = {k:source[k].copy() for k in ["input","hole","valid","atlas_indices","world_xyz"]}
    tile.update(id=record["id"],record=record)
    return tile


def projection_density_inventory(root,manifest,sources,args):
    """只读几何，测量128材质texel获得多少原相机像素支持，不读取任何目标RGB。"""
    wf=load_module(args.world_module);by_raw={s["raw"]:s for s in sources};records=[]
    for record in manifest["samples"]:
        raw=record.get("target_source_raw")
        if raw not in by_raw:continue
        source=by_raw[raw]
        with np.load(resolve(root,record["tile_npz"])) as packed:world=packed["world_xyz"]
        h,w=world.shape[:2]
        uv,depth,inside=wf.project_world_to_camera(world.reshape(-1,3),source["camera_K_native"],source["world_to_camera"],source["image_hw"],source["distortion_Brown"])
        uv=uv.reshape(h,w,2);du=uv[:-1,1:]-uv[:-1,:-1];dv=uv[1:,:-1]-uv[:-1,:-1]
        jacobian=np.stack([du,dv],axis=-1);singular=np.linalg.svd(jacobian,compute_uv=False)
        records.append(dict(id=record["id"],split=record["split"],raw=raw,
            native_footprint_span_px=np.ptp(uv.reshape(-1,2),axis=0).tolist(),
            native_pixels_per_fine_texel_sigma_min_p05_median_p95=np.quantile(singular[...,1],[.05,.5,.95]).tolist(),
            native_pixels_per_fine_texel_sigma_max_p05_median_p95=np.quantile(singular[...,0],[.05,.5,.95]).tolist(),
            median_native_pixel_area_per_fine_texel=float(np.median(abs(np.linalg.det(jacobian)))),
            inside_camera_fraction=float(inside.mean()),fine_texel_hw=[h,w]))
    write_json(args.output/"projection_density_inventory.json",dict(dataset=str(args.dataset),target_RGB_read=False,
        interpretation="minimum singular value measures the worst resolved road direction in native camera pixels per fine material texel",samples=records))
    print(json.dumps(records,ensure_ascii=False,indent=2),flush=True)


def integral_sum(mask, size, stride):
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
    h,w = mask.shape
    for y in range(0,h-size+1,stride):
        for x in range(0,w-size+1,stride):
            yield x,y,int(ii[y+size,x+size]-ii[y,x+size]-ii[y+size,x]+ii[y,x])


def candidates(root, manifest, args):
    atlas = manifest["atlas"]; h,w=atlas["shape_hw"]; size=manifest["image_size"][0]
    if "candidates" in manifest:
        excluded=np.load(resolve(root,atlas["exclude_world_mask"]),mmap_mode="r")
        out=args.output/"candidate_bank";out.mkdir(parents=True,exist_ok=True)
        result=[];records=[]
        for record in manifest["candidates"][:args.max_candidates]:
            with np.load(resolve(root,record["tile_npz"])) as packed:
                tile={k:packed[k].copy() for k in ["input","valid","hole","atlas_indices","world_xyz"]}
            if tile["hole"].any() or excluded[tile["atlas_indices"]].any():
                raise RuntimeError(f"候选含禁区: {record['id']}")
            if tile["valid"].mean()+1e-8<args.candidate_min_valid:
                continue
            tile["id"]=record["id"];result.append(tile)
            records.append({**record,"valid_fraction_verified":float(tile["valid"].mean()),
                "excluded_cells_verified":0,"fine_rgb_shape":list(tile["input"].shape),
                "unique_coarse_field_cells":len(np.unique(tile["atlas_indices"]))})
            Image.fromarray(np.clip(tile["input"]*255,0,255).astype(np.uint8)).save(out/(tile["id"]+".png"))
        write_json(out/"manifest.json",dict(candidate_count=len(result),provided_candidates=len(manifest["candidates"]),
            min_valid_fraction=args.candidate_min_valid,candidates=records,
            policy="data builder clean single-source whole tile warp; no supervision_RGB read; reject any eval or query-hole world cell"))
        if not result:raise RuntimeError("无满足干净单source合同的候选；未放宽valid或读取隐藏target")
        return result
    valid = np.load(resolve(root,atlas["observed_valid"]),mmap_mode="r").reshape(h,w)
    excluded = np.load(resolve(root,atlas["exclude_world_mask"]),mmap_mode="r").reshape(h,w)
    rgb = np.load(resolve(root,atlas["input_rgb"]),mmap_mode="r").reshape(h,w,3)
    world = np.load(resolve(root,atlas["world_xyz"]),mmap_mode="r").reshape(h,w,3)
    blocked = {(x,y):n for x,y,n in integral_sum(excluded,size,args.candidate_stride)}
    options=[]
    for x,y,n in integral_sum(valid,size,args.candidate_stride):
        if blocked[(x,y)]==0 and n/(size*size)>=args.candidate_min_valid:
            options.append((n,x,y))
    options.sort(key=lambda z:(-z[0],z[2],z[1]))
    chosen=[]
    for n,x,y in options:
        if len(chosen)>=args.max_candidates: break
        # 候选间最多50%线性重叠；固定顺序不使用query或隐藏GT选择库。
        if any(abs(x-c[1])<size//2 and abs(y-c[2])<size//2 for c in chosen): continue
        chosen.append((n,x,y))
    result=[]; records=[]; out=args.output/"candidate_bank";out.mkdir(parents=True,exist_ok=True)
    for n,x,y in chosen:
        yy,xx=np.mgrid[y:y+size,x:x+size];ids=yy*w+xx
        tile=dict(id=f"candidate_x{x:05d}_y{y:05d}",input=np.array(rgb[y:y+size,x:x+size]),
                  valid=np.array(valid[y:y+size,x:x+size]),hole=np.zeros((size,size),bool),atlas_indices=ids,
                  world_xyz=np.array(world[y:y+size,x:x+size]))
        rec=dict(id=tile["id"],atlas_pixel_bounds=[x,y,x+size,y+size],valid_fraction=n/(size*size),excluded_cells=0,
                 provenance="different_world_visible_source_RGB; no supervision_RGB_read")
        result.append(tile); records.append(rec)
        Image.fromarray(np.clip(tile["input"]*255,0,255).astype(np.uint8)).save(out/(tile["id"]+".png"))
    write_json(out/"manifest.json",dict(candidate_count=len(result),eligible_windows=len(options),stride=args.candidate_stride,
              min_valid_fraction=args.candidate_min_valid,max_candidates=args.max_candidates,
              policy="reject any complete eval tile or query-hole world cell",candidates=records))
    if not result:
        raise RuntimeError("无合法完整候选；保留失败收据，不回退读取训练或测试target")
    return result


def geometry_color_spectrum(tile, hole):
    mask=tile["valid"] & ~hole; rgb=tile["input"];pixels=rgb[mask]
    if len(pixels)<100: return None
    hist=np.concatenate([np.histogram(pixels[:,c],bins=8,range=(0,1),density=False)[0]/len(pixels) for c in range(3)])
    stats=np.concatenate([pixels.mean(0),pixels.std(0),np.quantile(pixels,[.1,.5,.9],axis=0).ravel()])
    gray=rgb@[.2989,.587,.114];gray=np.where(mask,gray,gray[mask].mean())
    power=abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean())))**2
    yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]];radius=np.hypot(xx-gray.shape[1]/2,yy-gray.shape[0]/2)
    edges=np.linspace(1,radius.max()+1,9)
    spectrum=np.array([power[(radius>=lo)&(radius<hi)].sum() for lo,hi in zip(edges[:-1],edges[1:])])
    spectrum=spectrum/max(spectrum.sum(),1e-8)
    xyz=tile["world_xyz"];dy=xyz[-1,0]-xyz[0,0];dx=xyz[0,-1]-xyz[0,0]
    normal=unit(np.cross(dx,dy));normal=normal if normal[2]>=0 else -normal
    return np.concatenate([hist,stats,spectrum,normal*.25]).astype(np.float32)


def project_bbox(world, camera, wf):
    # 外围轮廓九点；Brown畸变仍使用官方观测内参。
    h,w=world.shape[:2];ys=np.array([0,0,0,h//2,h//2,h//2,h-1,h-1,h-1]);xs=np.array([0,w//2,w-1,0,w//2,w-1,0,w//2,w-1])
    uv,depth,good=wf.project_world_to_camera(world[ys,xs],camera["camera_K_native"],camera["world_to_camera"],None,camera["distortion_Brown"])
    if np.count_nonzero((depth>.1)&np.isfinite(uv).all(1))<9:return None
    return np.r_[uv.min(0),uv.max(0)]


def hole_bbox(world, hole, camera, wf):
    yy,xx=np.where(hole)
    if not len(xx):return None
    y0,y1=yy.min(),yy.max();x0,x1=xx.min(),xx.max()
    points=world[[y0,y0,y1,y1],[x0,x1,x0,x1]]
    uv,depth,good=wf.project_world_to_camera(points,camera["camera_K_native"],camera["world_to_camera"],None,camera["distortion_Brown"])
    if np.any(depth<=.1) or not np.isfinite(uv).all():return None
    return np.r_[uv.min(0),uv.max(0)]


def rect_contains(uv,box):
    return (uv[:,0]>=box[0])&(uv[:,0]<=box[2])&(uv[:,1]>=box[1])&(uv[:,1]<=box[3])


def two_d_descriptor(tile,hole,source,cache,feature,wf,bins):
    box=project_bbox(tile["world_xyz"],source,wf)
    channels=feature.shape[-1];desc=np.zeros((bins*bins,channels),np.float32);counts=np.zeros(bins*bins,np.int32)
    if box is None:return desc,counts,None
    bh,bw=feature.shape[:2];nh,nw=source["image_hw"]
    uv=cache["_token_uv_native"]
    region=rect_contains(uv,box)
    inner=hole_bbox(tile["world_xyz"],hole,source,wf)
    if inner is not None:region &= ~rect_contains(uv,inner)
    # 相同pre-encoder遮盖之后，再屏蔽裁片中的非道路或无观测token中心。
    region &= cache["_token_allowed"]
    relative=(uv-box[:2])/np.maximum(box[2:]-box[:2],1e-6)
    cell=np.clip(np.floor(relative*bins).astype(int),0,bins-1)
    group=cell[:,1]*bins+cell[:,0]
    flat=np.asarray(feature,np.float32).reshape(-1,channels)
    for b in range(bins*bins):
        take=region&(group==b);counts[b]=take.sum()
        if counts[b]:desc[b]=flat[take].mean(0)
    return unit(desc),counts,box


def sparse_in_region(tile,hole,source,sparse,wf):
    box=project_bbox(tile["world_xyz"],source,wf)
    if box is None:return np.empty((0,256),np.float32)
    take=rect_contains(sparse["uv_native"],box)
    inner=hole_bbox(tile["world_xyz"],hole,source,wf)
    if inner is not None:take &= ~rect_contains(sparse["uv_native"],inner)
    return sparse["descriptors"][take]


def descriptor_distance(first,second):
    common=(first[1]>0)&(second[1]>0)
    if not common.any():return None,0
    return float(np.mean(1-np.sum(first[0][common]*second[0][common],axis=-1))),int(common.sum())


def sparse_distance(a,b):
    if len(a)<1 or len(b)<1:return None
    similarity=a@b.T
    return float(1-.5*(similarity.max(1).mean()+similarity.max(0).mean()))


def build_field(root,manifest,queries,bank,sources,features,wf,args):
    used_ids=np.unique(np.concatenate([t["atlas_indices"].ravel() for t in queries+bank]))
    atlas=manifest["atlas"];world=np.load(resolve(root,atlas["world_xyz"]),mmap_mode="r")
    remap=np.full(len(world),-1,np.int32);remap[used_ids]=np.arange(len(used_ids))
    excluded=np.load(resolve(root,atlas["feature_exclude_world_mask"]),mmap_mode="r")[used_ids]
    observations=[];global_observations=[];source_counts=[]
    for source,feature in zip(sources,features):
        with np.load(resolve(root,source["observations"])) as packed:
            obs={k:packed[k].copy() for k in packed.files}
        mapped=remap[obs["point_ids"]];take=mapped>=0
        subset={k:v[take] for k,v in obs.items()};subset["point_ids"]=mapped[take]
        global_observations.append(obs);observations.append(subset)
        nh,nw=source["image_hw"];hf,wf_shape=feature.shape[:2]
        token_xy=np.floor(obs["uv_native"][take]*[wf_shape/nw,hf/nh]).astype(int)
        token_xy=np.clip(token_xy,[0,0],[wf_shape-1,hf-1])
        unique_tokens=len(np.unique(token_xy[:,1]*wf_shape+token_xy[:,0]))
        source_counts.append(dict(raw=source["raw"],all_legal_observations=len(obs["point_ids"]),
                                  observations_in_compact_field=int(take.sum()),unique_nearest_source_tokens_in_field=unique_tokens))
    start=time.monotonic()
    result=wf.fuse_dense_features(world[used_ids],features,sources,observations,exclude_world_mask=excluded)
    assert not result["known"][excluded].any()
    field=result.pop("field").astype(np.float16)
    out=args.output/"world_features";out.mkdir(parents=True,exist_ok=True)
    np.save(out/"world_feature_field.npy",field);np.save(out/"atlas_point_ids.npy",used_ids)
    np.savez_compressed(out/"visibility_weights.npz",**result)
    write_json(out/"manifest.json",dict(schema="road_world_feature_field.v1",world_cells=len(used_ids),known_cells=int(result["known"].sum()),
               excluded_cells=int(excluded.sum()),channels=field.shape[1],fusion_seconds=time.monotonic()-start,
               source_raws=[s["raw"] for s in sources],source_features_shared_with_C=True,
               field_storage_shape=list(field.shape),field_storage_dtype=str(field.dtype),source_observation_counts=source_counts,
               weights="data builder certified camera-visible observation weight; same-world feature weighted average then L2 normalize",
               geometry_parameter_updates=0,model_parameter_updates=0))
    return field,remap,result,global_observations


def gpu_binding_smoke(root,manifest,queries,field,remap,weights,observations,sources,features,args):
    """实际GPU重放Brown投影→token双线性采样→visibility融合，核对独立CPU实现。"""
    if not torch.cuda.is_available():raise RuntimeError("真实CUDA特征绑定验收要求GPU可用")
    start=time.monotonic();started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    global_world=np.load(resolve(root,manifest["atlas"]["world_xyz"]),mmap_mode="r")
    context_ids=np.unique(np.concatenate([q["atlas_indices"][q["valid"]&~q["hole"]] for q in queries]))
    compact=remap[context_ids];good=(compact>=0)&weights["known"][np.maximum(compact,0)]
    context_ids=context_ids[good];compact=compact[good]
    order=np.argsort(-weights["observation_count"][compact],kind="stable")[:256]
    pids=context_ids[order];compact=compact[order]
    if not len(pids):raise RuntimeError("世界外围没有可验收特征，不能声称绑定通过")
    lookup=np.full(len(global_world),-1,np.int32);lookup[pids]=np.arange(len(pids))
    world=torch.as_tensor(np.array(global_world[pids]),dtype=torch.float64,device="cuda")
    accum=torch.zeros((len(pids),field.shape[1]),device="cuda");sum_weight=torch.zeros(len(pids),device="cuda")
    examples=[];max_uv_error=0.;pairs=0;source_pairs=[]
    for source,obs,feature in zip(sources,observations,features):
        local=lookup[obs["point_ids"]];take=local>=0
        if not take.any():source_pairs.append(dict(raw=source["raw"],pairs=0));continue
        ids=torch.as_tensor(local[take].astype(np.int64),device="cuda")
        matrix=torch.as_tensor(np.asarray(source["world_to_camera"]),dtype=torch.float64,device="cuda")
        K=torch.as_tensor(np.asarray(source["camera_K_native"]),dtype=torch.float64,device="cuda")
        xyz=world[ids]@matrix[:3,:3].T+matrix[:3,3]
        xy=xyz[:,:2]/xyz[:,2:3]
        x,y=xy.unbind(1);k1,k2,p1,p2,k3=source["distortion_Brown"]
        r2=x*x+y*y;radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        distorted=torch.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y],1)
        projected=distorted@K[:2,:2].T+K[:2,2]
        stored_uv=torch.as_tensor(obs["uv_native"][take],dtype=torch.float64,device="cuda")
        max_uv_error=max(max_uv_error,float((projected-stored_uv).abs().max().item()))
        nh,nw=source["image_hw"]
        grid=(projected/torch.tensor([nw,nh],dtype=torch.float64,device="cuda")*2-1).float()
        tensor=torch.as_tensor(np.asarray(feature,np.float32).transpose(2,0,1).copy(),device="cuda")[None]
        sampled=torch.nn.functional.grid_sample(tensor,grid[None,None],mode="bilinear",padding_mode="border",align_corners=False)[0,:,0].T
        weight=torch.as_tensor(obs["weight"][take],device="cuda")
        accum.index_add_(0,ids,sampled*weight[:,None]);sum_weight.index_add_(0,ids,weight)
        pairs+=int(take.sum());source_pairs.append(dict(raw=source["raw"],pairs=int(take.sum())))
        projected_cpu=projected.cpu().numpy();local_cpu=ids.cpu().numpy();selected_global=np.flatnonzero(take)
        for j,local_id in enumerate(local_cpu):
            if local_id>=3:continue
            uv=projected_cpu[j];hf,wf=feature.shape[:2]
            feature_xy=uv*[wf/nw,hf/nh]-.5;low=np.floor(feature_xy);frac=feature_xy-low
            ww=float(obs["weight"][selected_global[j]])
            examples.append(dict(atlas_point_id=int(pids[local_id]),world_xyz=global_world[pids[local_id]].tolist(),
                raw=source["raw"],uv_native_stored=obs["uv_native"][selected_global[j]].tolist(),uv_native_GPU=uv.tolist(),
                feature_grid_xy=feature_xy.tolist(),feature_grid_low_xy=low.astype(int).tolist(),
                bilinear_coefficients=[float((1-frac[0])*(1-frac[1])),float(frac[0]*(1-frac[1])),float((1-frac[0])*frac[1]),float(frac[0]*frac[1])],
                visibility_weight=ww,normalized_source_weight=ww/max(float(weights["weight"][compact[local_id]]),1e-12)))
    fused=torch.nn.functional.normalize(accum/torch.clamp(sum_weight[:,None],min=1e-12),dim=1).cpu().numpy()
    reference=field[compact].astype(np.float32)
    error=float(np.max(np.abs(fused-reference)))
    cosine=np.sum(unit(fused)*unit(reference),axis=1)
    weight_error=float(np.max(np.abs(sum_weight.cpu().numpy()-weights["weight"][compact])))
    assert max_uv_error<.1,("camera_projection_mismatch",max_uv_error)
    assert error<.003,("GPU_binding_feature_mismatch",error)
    assert np.isfinite(fused).all() and float(cosine.min())>.999
    receipt=dict(status="PASS_REAL_GPU_PROJECT_SAMPLE_VISIBLE_WEIGHT_FUSION",device=torch.cuda.get_device_name(),
        started_at_utc=started,ended_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),seconds=time.monotonic()-start,
        shared_gpu_context=args.shared_gpu_context,world_points=len(pids),source_observation_pairs=pairs,
        source_pairs=source_pairs,camera_Brown_GPU_vs_stored_uv_max_abs_px=max_uv_error,
        GPU_fusion_vs_CPU_float16_field_max_abs=error,minimum_cosine_GPU_CPU=float(cosine.min()),
        summed_visibility_weight_max_abs_error=weight_error,examples=examples)
    write_json(args.output/"gpu_projection_binding_smoke.json",receipt)
    print("GPU_BINDING_SMOKE",json.dumps({k:v for k,v in receipt.items() if k not in ["examples","source_pairs"]}),flush=True)


def world_descriptor(tile,hole,field,remap,known,bins):
    h,w=hole.shape;ids=remap[tile["atlas_indices"]]
    valid=tile["valid"]&~hole&(ids>=0)
    valid &= known[np.maximum(ids,0)]
    desc=np.zeros((bins*bins,field.shape[1]),np.float32);counts=np.zeros(bins*bins,np.int32)
    for by in range(bins):
        for bx in range(bins):
            ys=slice(by*h//bins,(by+1)*h//bins);xs=slice(bx*w//bins,(bx+1)*w//bins)
            usable=valid[ys,xs];i=by*bins+bx;counts[i]=usable.sum()
            if counts[i]:desc[i]=field[ids[ys,xs][usable]].astype(np.float32).mean(0)
    return unit(desc),counts


def fill_candidate(query,candidate):
    prediction=query["input"].copy();hole=query["hole"]&query["valid"]
    values=candidate["input"].copy();available=candidate["valid"]
    if not available.all():
        # 仅合法候选的缺测像素使用其最近已观测像素，单独记账；不调用任何target。
        from scipy.ndimage import distance_transform_edt
        nearest=distance_transform_edt(~available,return_distances=False,return_indices=True)
        values[~available]=values[nearest[0][~available],nearest[1][~available]]
    prediction[hole]=values[hole]
    return prediction,float((hole&~available).sum()/max(hole.sum(),1))


def evaluate_after_ranking(root,query,prediction):
    with np.load(resolve(root,query["record"]["tile_npz"])) as packed:target=packed["target"]
    mask=query["hole"]&query["valid"];error=np.abs(prediction-target)
    mae=float(error[mask].mean());mse=float(((prediction-target)**2)[mask].mean())
    return dict(hole_pixels=int(mask.sum()),hole_mae=mae,hole_psnr=float(-10*np.log10(max(mse,1e-12))),
                known_context_max_abs_change=float(np.max(np.abs(prediction-query["input"])[~query["hole"]]))),target


def make_contact(path,images,labels):
    side=192;bar=32;out=Image.new("RGB",(side*len(images),side+bar),(17,23,31));draw=ImageDraw.Draw(out)
    for i,(arr,label) in enumerate(zip(images,labels)):
        tile=Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8)).resize((side,side),Image.Resampling.NEAREST)
        out.paste(tile,(i*side,bar));draw.text((i*side+7,8),label,fill=(226,232,240))
    out.save(path)


def package_results(root,manifest,sources,args):
    """冻结结果的小包与原相机来源图；不重新排序、不读取评估target。"""
    result_path=args.output/"retrieval_results.json"
    result=json.loads(result_path.read_text())
    by_id={c["id"]:c for c in manifest["candidates"]};by_raw={s["raw"]:s for s in sources}
    extraction=json.loads((args.output/"source_features/extraction_receipt.json").read_text())
    field=json.loads((args.output/"world_features/manifest.json").read_text())
    gpu=json.loads((args.output/"gpu_projection_binding_smoke.json").read_text())
    references=[];denominators=[]
    for query in result["queries"]:
        D=query["methods"]["D_native_dinov3_world"]
        if D["status"]!="ran":
            references.append(dict(query_id=query["query_id"],status="unknown_no_reference",reason=D["status"]))
            continue
        top=D["top5"][0];candidate=by_id[top["candidate_id"]]
        path=resolve(root,candidate["input"])
        assert path.exists()
        references.append(dict(query_id=query["query_id"],status="selected",candidate_id=candidate["id"],
            reference_png=str(path),reference_valid_mask=str(resolve(root,candidate["valid_mask"])),
            candidate_tile_npz=str(resolve(root,candidate["tile_npz"])),target_source_raw=candidate["target_source_raw"],
            retrieval_distance=top["distance"],eligible_candidates=D["eligible_candidates"],
            atlas_pixel_bounds=candidate["atlas_pixel_bounds"],world_from_uv=candidate["world_from_uv"],
            query_or_eval_world_overlap_cells=candidate.get("query_or_eval_world_overlap_cells",0)))
        denominators.append(dict(query_id=query["query_id"],hole_pixels=query["hole_pixels"],
            C_eligible=query["methods"]["C_native_dinov3_2d"]["eligible_candidates"],D_eligible=D["eligible_candidates"],
            common_C_D_candidates=query["C_D_matched_candidate_pool"]["candidate_count"],
            D_context_fine_texels=query["D_context_fine_texels_with_world_feature"],
            D_context_unique_coarse_world_cells=query["D_context_unique_coarse_world_cells"]))
    write_json(args.output/"D_top1_reference_manifest.json",dict(schema="road_D_top1_reference.v1",
        dataset=str(args.dataset),native_D_results=str(result_path),selection="frozen native D world-feature context retrieval, top1 minimum distance",
        heldout_target_used_for_selection=False,oracle=False,fallback=False,model_parameter_updates=0,references=references))
    write_json(args.output/"summary.json",dict(aggregate=result["aggregate"],aggregate_C_D_matched_candidate_pool=result["aggregate_C_D_matched_candidate_pool"],
        denominators=denominators,extraction={k:extraction[k] for k in ["source_raws","dino_feature_total_tokens","frozen","parameter_updates","started_at_utc","ended_at_utc","seconds","peak_torch_allocated_bytes","shared_gpu_context"]},
        world_field={k:field[k] for k in ["world_cells","known_cells","excluded_cells","channels","field_storage_shape","field_storage_dtype","fusion_seconds"]},
        GPU_binding={k:gpu[k] for k in gpu if k not in ["source_pairs","examples"]},
        limits="4 pseudo-occlusion holes in one scene, no cross-scene claim, no true hidden GT; no method dominance inferred from this small pilot"))
    source_cache={};crop_cache={};galleries=[];mappings=[]
    sample_records={s["id"]:s for s in manifest["samples"]};wf=load_module(args.world_module)
    for query in result["queries"]:
        qrecord=sample_records[query["query_id"]];qraw=qrecord["target_source_raw"];qcamera=by_raw[qraw]
        with np.load(resolve(root,qrecord["tile_npz"])) as packed:qworld=packed["world_xyz"]
        quv,_,_=wf.project_world_to_camera(qworld.reshape(-1,3),qcamera["camera_K_native"],qcamera["world_to_camera"],qcamera["image_hw"],qcamera["distortion_Brown"])
        mapping=dict(query_id=query["query_id"],input=str(resolve(root,qrecord["input"])),
            hole_mask=str(resolve(root,qrecord["hole_mask"])),valid_mask=str(resolve(root,qrecord["valid_mask"])),
            query_world_from_uv=qrecord["world_from_uv"],query_atlas_pixel_bounds=qrecord["atlas_pixel_bounds"],
            metric_target_source_raw=qraw,metric_target_native_bbox_xyxy=np.r_[quv.min(0),quv.max(0)].tolist(),
            metric_target_role="heldout metric only; not read for candidate selection",methods={})
        for method,record in query["methods"].items():
            if record["status"]!="ran":continue
            side=208;header=60;rowbar=36
            canvas=Image.new("RGB",(5*side,header+2*(side+rowbar)),(16,22,31));draw=ImageDraw.Draw(canvas)
            draw.text((12,10),f"{query['query_id']}  |  {method}  |  frozen top5",fill=(234,239,246))
            draw.text((12,30),"Top: aligned metric-world RGB. Bottom: legal original-camera crop with projected tile boundary.",fill=(151,166,187))
            provenance=[]
            for rank,top in enumerate(record["top5"]):
                candidate=by_id[top["candidate_id"]];cid=candidate["id"];raw=candidate["target_source_raw"]
                if cid not in crop_cache:
                    if raw not in source_cache:
                        source_cache[raw]=Image.open(resolve(root,by_raw[raw]["feature_input"])).convert("RGB")
                    source=source_cache[raw]
                    with np.load(resolve(root,candidate["tile_npz"])) as packed:uv=packed["uv_native"].copy()
                    uv=uv.reshape(-1,2) if uv.ndim==2 else uv
                    if uv.ndim==2:uv=uv.reshape(128,128,2)
                    low=np.floor(uv.min((0,1))-8).astype(int);high=np.ceil(uv.max((0,1))+8).astype(int)
                    low=np.maximum(low,[0,0]);high=np.minimum(high,source.size)
                    crop=source.crop(tuple([*low,*high]));mark=ImageDraw.Draw(crop)
                    boundary=uv[[0,0,-1,-1,0],[0,-1,-1,0,0]]-low
                    mark.line([tuple(p) for p in boundary],fill=(94,230,204),width=2)
                    crop_cache[cid]=(crop,[*low.tolist(),*high.tolist()],list(source.size))
                crop,bbox,native_wh=crop_cache[cid]
                aligned=Image.open(resolve(root,candidate["input"])).convert("RGB").resize((side,side),Image.Resampling.NEAREST)
                x=rank*side
                draw.text((x+6,header+5),f"{rank+1}. {cid}  d={top['distance']:.4f}",fill=(232,238,246))
                canvas.paste(aligned,(x,header+rowbar))
                y=header+rowbar+side
                draw.text((x+6,y+5),f"raw {raw}: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} native px",fill=(232,238,246))
                thumb=ImageOps.contain(crop,(side,side),Image.Resampling.BILINEAR)
                canvas.paste(thumb,(x+(side-thumb.width)//2,y+rowbar+(side-thumb.height)//2))
                provenance.append(dict(rank=rank+1,candidate_id=cid,distance=top["distance"],raw=raw,
                    aligned_rgb=str(resolve(root,candidate["input"])),legal_source_rgb=str(resolve(root,by_raw[raw]["feature_input"])),
                    native_crop_xyxy=bbox,native_image_wh=native_wh,
                    atlas_pixel_bounds=candidate["atlas_pixel_bounds"],world_from_uv=candidate["world_from_uv"]))
            folder=args.output/"predictions"/query["query_id"]/method
            canvas.save(folder/"top5_source_and_world.png")
            write_json(folder/"top5_source_and_world_receipt.json",provenance)
            mapping["methods"][method]=provenance
            galleries.append(dict(query_id=query["query_id"],method=method,path=str(folder/"top5_source_and_world.png")))
        mappings.append(mapping)
    write_json(args.output/"top5_gallery_manifest.json",galleries)
    write_json(args.output/"source_target_mapping.json",dict(queries=mappings,
        cameras={str(s["raw"]):{k:s[k] for k in ["camera_K_native","world_to_camera","distortion_Brown","image_hw"]} for s in sources}))
    import zipfile
    files=list(args.output.glob("*.json"))+list(args.output.glob("*_comparison.png"))
    files += [args.output/"source_features/extraction_receipt.json",args.output/"world_features/manifest.json",
              args.output/"world_features/visibility_weights.npz",args.output/"world_features/atlas_point_ids.npy"]
    files += [p for p in (args.output/"predictions").rglob("*") if p.is_file()]
    files += [p for p in (args.output/"candidate_bank").rglob("*") if p.is_file()]
    with zipfile.ZipFile(args.output/"local_report_package.zip","w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            if path.exists():archive.write(path,path.relative_to(args.output))
    print("PACKAGE_READY",str(args.output),len(references),len(galleries),flush=True)


def source_examples(query,candidate,field,remap,weights,observations,sources,features,wf,C_details,atlas_world):
    tile_ids=remap[query["atlas_indices"]]
    context=query["valid"]&~query["hole"]&(tile_ids>=0)
    context &= weights["known"][np.maximum(tile_ids,0)]
    coords=np.argwhere(context)
    rank=np.argsort(-weights["observation_count"][tile_ids[context]],kind="stable")
    selected=[]
    for point in coords[rank]:
        if all(np.linalg.norm(point-other)>20 for other in selected):selected.append(point)
        if len(selected)==3:break
    result=[]
    for y,x in selected:
        pid=int(query["atlas_indices"][y,x]);fid=int(remap[pid]);entries=[]
        for source,obs,feature in zip(sources,observations,features):
            hits=np.flatnonzero(obs["point_ids"]==pid)
            if not len(hits):continue
            j=int(hits[0]);uv=obs["uv_native"][j];weight=float(obs["weight"][j])
            value=wf.bilinear_sample(np.asarray(feature,np.float32),uv[None],source["image_hw"])[0]
            entries.append(dict(raw=source["raw"],uv_native=uv.tolist(),visibility_weight=weight,
                normalized_contribution=weight/max(float(weights["weight"][fid]),1e-12),
                cosine_source_to_world=float(unit(value)@field[fid].astype(np.float32))))
        result.append(dict(atlas_point_id=pid,tile_uv=[int(x),int(y)],world_xyz_field_cell=atlas_world[pid].tolist(),
                           fine_texel_world_xyz=query["world_xyz"][y,x].tolist(),
                           summed_weight=float(weights["weight"][fid]),observation_count=int(weights["observation_count"][fid]),sources=entries))
    return dict(query_id=query["id"],D_top_candidate=candidate["id"],C_view_comparison=C_details,world_context_point_examples=result)


def retrieve(root,manifest,sources,receipt,args):
    wf=load_module(args.world_module)
    queries=[load_query(root,s) for s in manifest["samples"] if s["split"]=="eval"]
    bank=[] if args.binding_interface_only else candidates(root,manifest,args)
    features=[np.load(r["feature_path"]).astype(np.float32) for r in receipt["sources"]]
    for source,cache,feature in zip(sources,receipt["sources"],features):
        bh,bw=feature.shape[:2];nh,nw=source["image_hw"]
        yy,xx=np.mgrid[:bh,:bw];uv=np.c_[(xx.ravel()+.5)*nw/bw,(yy.ravel()+.5)*nh/bh]
        allowed=np.asarray(Image.open(cache["allowed_mask_path"]).convert("L"))>127
        pixel=np.clip(np.floor(uv).astype(int),[0,0],[nw-1,nh-1])
        cache["_token_uv_native"]=uv;cache["_token_allowed"]=allowed[pixel[:,1],pixel[:,0]]
    sparse=[]
    for r in receipt["sources"]:
        with np.load(r["superpoint_path"]) as packed:sparse.append({k:packed[k] for k in packed.files})
    field,remap,weights,observations=build_field(root,manifest,queries,bank,sources,features,wf,args)
    atlas_world=np.load(resolve(root,manifest["atlas"]["world_xyz"]),mmap_mode="r")
    gpu_binding_smoke(root,manifest,queries,field,remap,weights,observations,sources,features,args)
    if args.binding_interface_only:
        write_json(args.output/"engineering_interface_only.json",dict(status="actual_GPU_binding_interface_executed",
            formal_retrieval_or_quality_evaluation=False,hidden_target_loaded=False,sources=[s["raw"] for s in sources],
            dataset=str(args.dataset),purpose="validate real camera-to-world feature wiring while revised dataset is being built"))
        return
    records=[];examples=[];started=time.monotonic()
    methods=["A_geometry_color_spectrum","B_native_superpoint_2d","C_native_dinov3_2d","D_native_dinov3_world"]
    for qi,query in enumerate(queries):
        qhole=query["hole"];qA=geometry_color_spectrum(query,qhole)
        bA=[geometry_color_spectrum(c,qhole) for c in bank]
        validA=[d for d in bA if d is not None]
        scale=np.maximum(np.std(np.stack(validA),axis=0),.05)
        scoreA=[float(np.mean(((qA-d)/scale)**2)) if d is not None else None for d in bA]
        qD=world_descriptor(query,qhole,field,remap,weights["known"],args.bins)
        bD=[world_descriptor(c,qhole,field,remap,weights["known"],args.bins) for c in bank]
        scoreD=[descriptor_distance(qD,d)[0] for d in bD]
        scoreC=[[] for _ in bank];scoreB=[[] for _ in bank];detailsC=[[] for _ in bank];detailsB=[[] for _ in bank]
        C_counts=[];B_counts=[]
        for si,(source,cache,feature,sp) in enumerate(zip(sources,receipt["sources"],features,sparse)):
            qC=two_d_descriptor(query,qhole,source,cache,feature,wf,args.bins)
            qB=sparse_in_region(query,qhole,source,sp,wf)
            C_counts.append(int(qC[1].sum()));B_counts.append(len(qB))
            if qC[1].sum()==0 and len(qB)<1:continue
            for ci,candidate in enumerate(bank):
                cC=two_d_descriptor(candidate,qhole,source,cache,feature,wf,args.bins)
                dc,n=descriptor_distance(qC,cC)
                if dc is not None:
                    scoreC[ci].append(dc);detailsC[ci].append(dict(raw=source["raw"],distance=dc,common_bins=n,
                        query_tokens=int(qC[1].sum()),candidate_tokens=int(cC[1].sum()),query_bbox=qC[2].tolist(),candidate_bbox=cC[2].tolist()))
                cB=sparse_in_region(candidate,qhole,source,sp,wf);db=sparse_distance(qB,cB)
                if db is not None:
                    scoreB[ci].append(db);detailsB[ci].append(dict(raw=source["raw"],distance=db,
                        query_keypoints=len(qB),candidate_keypoints=len(cB)))
        for per_candidate in detailsC:
            for per_source in per_candidate:per_source["normalized_C_view_weight"]=1/len(per_candidate)
        scores={methods[0]:scoreA,methods[1]:[float(np.mean(v)) if v else None for v in scoreB],
                methods[2]:[float(np.mean(v)) if v else None for v in scoreC],methods[3]:scoreD}
        qrec=dict(query_id=query["id"],hole_pixels=int((qhole&query["valid"]).sum()),candidate_bank_count=len(bank),
                  context_pixels=int((query["valid"]&~qhole).sum()),D_context_fine_texels_with_world_feature=int(qD[1].sum()),
                  D_context_unique_coarse_world_cells=len(np.unique(query["atlas_indices"][(query["valid"]&~qhole)&weights["known"][remap[query["atlas_indices"]]]])),
                  C_query_tokens_per_source=C_counts,B_query_keypoints_per_source=B_counts,methods={})
        images=[query["input"]];labels=["legal query input"];target_for_contact=None
        for method in methods:
            order=sorted([i for i,s in enumerate(scores[method]) if s is not None and np.isfinite(s)],key=lambda i:(scores[method][i],bank[i]["id"]))
            if not order:
                qrec["methods"][method]=dict(status="no_legal_feature_comparison",eligible_candidates=0,top5=[])
                continue
            top=order[:5];first=top[0];prediction,inferred=fill_candidate(query,bank[first])
            metrics,target=evaluate_after_ranking(root,query,prediction);target_for_contact=target
            out=args.output/"predictions"/query["id"]/method;out.mkdir(parents=True,exist_ok=True)
            np.save(out/"prediction.npy",prediction);Image.fromarray(np.clip(prediction*255,0,255).astype(np.uint8)).save(out/"prediction.png")
            ranking=[]
            for ci in top:
                record=dict(candidate_id=bank[ci]["id"],distance=scores[method][ci])
                if method==methods[1]:
                    record["shared_source_views"]=len(scoreB[ci]);record["per_source"]=detailsB[ci]
                if method==methods[2]:record["shared_source_views"]=len(scoreC[ci]);record["per_source"]=detailsC[ci]
                if method==methods[3]:
                    record["common_world_bins"]=descriptor_distance(qD,bD[ci])[1]
                    record["candidate_context_fine_texels_with_world_feature"]=int(bD[ci][1].sum())
                ranking.append(record)
            qrec["methods"][method]=dict(status="ran",eligible_candidates=len(order),top5=ranking,metrics=metrics,
               candidate_missing_center_nearest_legal_fraction=inferred,prediction_path=str(out/"prediction.npy"),
               origin="different_world_reference_copy_into_query_hole; known query pixels preserved exactly")
            images.append(prediction);labels.append(method.split("_")[0]+f" MAE {metrics['hole_mae']:.4f}")
            make_contact(out/"top5_candidates.png",[bank[i]["input"] for i in top],[f"{j+1}: {bank[i]['id']}" for j,i in enumerate(top)])
        # 另给相同可比较候选交集上的C/D，区分世界融合与共有视图覆盖产生的差异。
        common_candidates=[i for i in range(len(bank)) if scores[methods[2]][i] is not None and scores[methods[3]][i] is not None]
        qrec["C_D_matched_candidate_pool"]={"candidate_count":len(common_candidates),"methods":{}}
        for method in methods[2:]:
            order=sorted(common_candidates,key=lambda i:(scores[method][i],bank[i]["id"]))
            if not order:continue
            prediction,inferred=fill_candidate(query,bank[order[0]])
            metrics,_=evaluate_after_ranking(root,query,prediction)
            out=args.output/"predictions"/query["id"]/(method+"_matched_candidate_pool");out.mkdir(parents=True,exist_ok=True)
            np.save(out/"prediction.npy",prediction)
            Image.fromarray(np.clip(prediction*255,0,255).astype(np.uint8)).save(out/"prediction.png")
            qrec["C_D_matched_candidate_pool"]["methods"][method]=dict(metrics=metrics,
                top5=[dict(candidate_id=bank[i]["id"],distance=scores[method][i]) for i in order[:5]],
                prediction_path=str(out/"prediction.npy"),candidate_missing_center_nearest_legal_fraction=inferred)
        if target_for_contact is not None:images.append(target_for_contact);labels.append("heldout metric target")
        make_contact(args.output/f"{query['id']}_comparison.png",images,labels)
        if scores[methods[3]] and any(v is not None for v in scoreD):
            topD=min((i for i,v in enumerate(scoreD) if v is not None),key=lambda i:scoreD[i])
            example=source_examples(query,bank[topD],field,remap,weights,observations,sources,features,wf,detailsC[topD],atlas_world)
            if qrec["methods"][methods[2]]["status"]=="ran":
                example["C_top_candidate"]=qrec["methods"][methods[2]]["top5"][0]["candidate_id"]
                example["C_top_candidate_source_comparisons"]=qrec["methods"][methods[2]]["top5"][0]["per_source"]
            examples.append(example)
        records.append(qrec)
        write_json(args.output/"per_query_progress.json",records)
        print(f"RETRIEVE {qi+1}/{len(queries)} query={query['id']} candidates={len(bank)} seconds={time.monotonic()-started:.1f}",flush=True)
    aggregate={}
    for method in methods:
        ran=[r["methods"][method] for r in records if r["methods"][method]["status"]=="ran"]
        pixels=sum(r["metrics"]["hole_pixels"] for r in ran)
        aggregate[method]=dict(queries_ran=len(ran),queries_total=len(queries),hole_pixels_scored=pixels,
             hole_mae_pixel_weighted=sum(r["metrics"]["hole_mae"]*r["metrics"]["hole_pixels"] for r in ran)/pixels if pixels else None,
             mean_hole_psnr=float(np.mean([r["metrics"]["hole_psnr"] for r in ran])) if ran else None,
             known_context_max_abs_change=max([r["metrics"]["known_context_max_abs_change"] for r in ran],default=None))
    comparable=[r for r in records if all(r["methods"][m]["status"]=="ran" for m in methods)]
    matched_aggregate={}
    for method in methods[2:]:
        ran=[r["C_D_matched_candidate_pool"]["methods"][method] for r in records if method in r["C_D_matched_candidate_pool"]["methods"]]
        pixels=sum(r["metrics"]["hole_pixels"] for r in ran)
        matched_aggregate[method]=dict(queries_ran=len(ran),hole_pixels_scored=pixels,
            hole_mae_pixel_weighted=sum(r["metrics"]["hole_mae"]*r["metrics"]["hole_pixels"] for r in ran)/pixels if pixels else None)
    result=dict(schema="road_retrieval_controls.v1",run_id=args.run_id or manifest.get("run_id",manifest.get("scene_id","road_retrieval")),dataset=str(args.dataset),
          native_frozen_baselines=True,learned_ranking_head_updates=0,query_count=len(queries),candidate_count=len(bank),
          matched_all_method_queries=len(comparable),same_frame_subset_C_D=True,same_encoder_tokens_C_D=True,
          source_raws=[s["raw"] for s in sources],aggregate=aggregate,aggregate_C_D_matched_candidate_pool=matched_aggregate,queries=records,
          evidence_boundary="same-scene disjoint-world pseudo-occlusion; observed appearance not intrinsic albedo; true hidden road has no GT; retrieval does not establish generated completion quality",
          failure_ledger_delta="none")
    write_json(args.output/"retrieval_results.json",result)
    write_json(args.output/"source_weight_examples.json",examples)
    print(json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--world-module",type=Path,default=Path(__file__).resolve().parents[1]/"world_feature_field.py")
    parser.add_argument("--dino",default=os.environ.get("DINO_V3_MODEL"))
    parser.add_argument("--superpoint-weights",type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--max-side",type=int,default=640)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--bins",type=int,default=4)
    parser.add_argument("--candidate-stride",type=int,default=64)
    parser.add_argument("--candidate-min-valid",type=float,default=.98)
    parser.add_argument("--max-candidates",type=int,default=64)
    parser.add_argument("--extract-only",action="store_true")
    parser.add_argument("--retrieve-only",action="store_true")
    parser.add_argument("--reextract",action="store_true")
    parser.add_argument("--binding-interface-only",action="store_true")
    parser.add_argument("--projection-density-only",action="store_true")
    parser.add_argument("--package-only",action="store_true")
    parser.add_argument("--shared-gpu-context",default="concurrent road native/LoRA pilot; wallclock is not exclusive GPU performance")
    args=parser.parse_args();args.dataset=args.dataset.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    if (args.output/"INVALIDATED.json").exists():
        raise RuntimeError("此目录已因数据泄漏作废，保留现场；必须使用新output完整重放，禁止旧cache/参考交接")
    os.environ.setdefault("OMP_NUM_THREADS",str(args.threads))
    root=args.dataset.parent;manifest=json.loads(args.dataset.read_text());sources=load_sources(root,manifest)
    if args.package_only:
        package_results(root,manifest,sources,args)
        return
    if args.projection_density_only:
        projection_density_inventory(root,manifest,sources,args)
        return
    if args.binding_interface_only:sources=sources[-2:]
    if not args.retrieve_only and not (args.output/"source_features/extraction_receipt.json").exists():
        if not args.dino or args.superpoint_weights is None:
            parser.error("首次提取必须显式提供 --dino 和 --superpoint-weights（模型不随代码提交）")
    audit_encoder_fine_holes(root,manifest,sources,args)
    if args.retrieve_only:
        receipt=json.loads((args.output/"source_features/extraction_receipt.json").read_text())
    else:receipt=extract(root,manifest,sources,args)
    if not args.extract_only:retrieve(root,manifest,sources,receipt,args)


if __name__=="__main__":main()
