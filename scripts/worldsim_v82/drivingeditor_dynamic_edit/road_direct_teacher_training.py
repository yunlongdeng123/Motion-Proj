"""固定二维道路教师到持久道路高斯：直接c_i，不训练共享prior或曝光变量。

原生光栅按sRGB外观数值合成。损失比较黑背景预乘RGB，不使用RGB/alpha，
也不把该数值称为线性物理辐射度。A为可见真实道路，C为固定补全伪教师。
"""
from pathlib import Path
import argparse
import copy
import json
import os
import random
import shutil
import time
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(name,"2")
import numpy as np
import torch
from .road_texture_training import file_digest, differences


TRAIN_FRAMES=(86,110)
DIAGNOSTIC_FRAMES=(134,179,198)


def load(path):
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def rgb_tensor(array,device):
    a=np.asarray(array)
    if a.ndim!=3:raise ValueError("RGB必须是三维数组")
    if a.shape[0]!=3 and a.shape[-1]==3:a=a.transpose(2,0,1)
    if a.shape[0]!=3:raise ValueError("RGB必须有三个通道")
    integer=a.dtype.kind in "ui"
    a=a.astype(np.float32)
    if integer:a/=255
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError("RGB须为有限[0,1]")
    return torch.as_tensor(a.copy(),device=device)


def masked_rgb_l1(prediction,target,mask,weight=None):
    q=mask.to(prediction.dtype)
    if weight is not None:q=q*weight
    return ((prediction-target).abs()*q[None]).sum()/(3*q.sum().clamp_min(1))


def masked_alpha_mse(prediction,target,mask,weight=None):
    q=mask.to(prediction.dtype)
    if weight is not None:q=q*weight
    return ((prediction-target).square()*q).sum()/q.sum().clamp_min(1)


def masked_edge_l1(prediction,target,mask,weight=None):
    q=mask.to(prediction.dtype) if weight is None else mask*weight
    horizontal=mask[:,1:]&mask[:,:-1]
    vertical=mask[1:]&mask[:-1]
    qx=torch.minimum(q[:,1:],q[:,:-1])*horizontal
    qy=torch.minimum(q[1:],q[:-1])*vertical
    dx=(prediction[:,:,1:]-prediction[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])
    dy=(prediction[:,1:]-prediction[:,:-1])-(target[:,1:]-target[:,:-1])
    return (dx.abs().mul(qx[None]).sum()+dy.abs().mul(qy[None]).sum())/(3*(qx.sum()+qy.sum()).clamp_min(1))


class DirectRoadParameters(torch.nn.Module):
    """独立可审计的RGB、有效opacity与有绝对上界的面内scale参数。"""
    def __init__(self,rgb,opacity,scales,color_ids,opacity_ids,scale_ids,scale_upper):
        super().__init__()
        for name,value in (("base_rgb",rgb),("base_opacity",opacity.reshape(-1)),("base_scales",scales)):
            self.register_buffer(name,value.detach().clone())
        n=len(rgb)
        for name,value in (("color_ids",color_ids),("opacity_ids",opacity_ids),("scale_ids",scale_ids)):
            value=torch.as_tensor(value,device=rgb.device,dtype=torch.long)
            if value.ndim!=1 or len(torch.unique(value))!=len(value) or (value<0).any() or (value>=n).any():
                raise ValueError("参数行范围无效: "+name)
            self.register_buffer(name,value)
        self.colors=torch.nn.Parameter(self.base_rgb[self.color_ids].clone())
        initial=torch.logit(self.base_opacity[self.opacity_ids].clamp(1e-5,1-1e-5))
        self.register_buffer("opacity_base_logit",initial.clone())
        self.register_buffer("opacity_base_sigmoid",initial.sigmoid())
        self.opacity_logits=torch.nn.Parameter(torch.zeros_like(initial))
        maximum=float(torch.logit(initial.new_tensor(1-1e-5)))
        self.register_buffer("opacity_upper_logit",(maximum-initial).clamp_min(0))
        self.register_buffer("scale_upper",torch.as_tensor(scale_upper,device=rgb.device,dtype=scales.dtype).clone())
        if self.scale_upper.shape!=self.scale_ids.shape or (self.scale_upper<1).any():raise ValueError("scale上界无效")
        self.scale_factor=torch.nn.Parameter(torch.ones_like(self.scale_upper))

    def full_colors(self,detach=False):
        value=self.base_rgb.index_copy(0,self.color_ids,self.colors)
        return value.detach() if detach else value

    def full_opacity(self,detach=False):
        p=self.base_opacity[self.opacity_ids]+((self.opacity_base_logit+self.opacity_logits).sigmoid()-self.opacity_base_sigmoid)
        p=torch.minimum(p,p.new_full(p.shape,1-1e-5))
        value=self.base_opacity.index_copy(0,self.opacity_ids,p)[:,None]
        return value.detach() if detach else value

    def full_scales(self,detach=False):
        factor=self.scale_factor
        selected=torch.cat([self.base_scales[self.scale_ids,:2]*factor[:,None],self.base_scales[self.scale_ids,2:3]],dim=1)
        value=self.base_scales.index_copy(0,self.scale_ids,selected)
        return value.detach() if detach else value

    def packet(self,base,route):
        if route not in ("photo","coverage","live"):raise ValueError("未知梯度路由")
        return dict(base,colors_precomp=self.full_colors(detach=route=="coverage"),
            opacities=self.full_opacity(detach=route=="photo"),scales=self.full_scales(detach=route=="photo"))

    def project(self):
        with torch.no_grad():
            self.colors.clamp_(0,1)
            self.opacity_logits.copy_(self.opacity_logits.clamp_min(0).minimum(self.opacity_upper_logit))
            self.scale_factor.copy_(torch.maximum(torch.ones_like(self.scale_factor),torch.minimum(self.scale_factor,self.scale_upper)))


def photo_loss(pack,target,baseline,colors,initial_colors,weights):
    """调用方必须用photo route光栅，不能只detach RHS而保留coverage梯度。"""
    alpha=pack["alpha"][0].detach()
    prediction=pack["rgb"]
    truth_A=alpha[None]*target["target_A_srgb"]
    truth_C=alpha[None]*target["selected_C_srgb"]
    a=masked_rgb_l1(prediction,truth_A,target["photo_A"],target["q_A"])
    c=masked_rgb_l1(prediction,truth_C,target["photo_C"],target["q"])
    ea=masked_edge_l1(prediction,truth_A,target["photo_A"],target["q_A"])
    ec=masked_edge_l1(prediction,truth_C,target["photo_C"],target["q"])
    keep=masked_rgb_l1(prediction,baseline["rgb"],target["outside"])
    retained=masked_rgb_l1(prediction,baseline["rgb"],target["K"])
    prior=(colors-initial_colors).square().mean() if colors.numel() else prediction.sum()*0
    total=(a+weights["C_photo"]*c+weights["edge"]*(ea+weights["C_edge_relative"]*ec)
        +weights["outside_RGB"]*keep+weights["retained_RGB"]*retained+weights["RGB_prior_squared"]*prior)
    return total,dict(A=a,C=c,edge_A=ea,edge_C=ec,outside_RGB=keep,retained_RGB=retained,RGB_prior=prior)


def coverage_loss(pack,target,baseline,weights):
    alpha=pack["alpha"][0]
    a=masked_alpha_mse(alpha,torch.ones_like(alpha),target["coverage_A"],target["coverage_q_A"])
    c=masked_alpha_mse(alpha,torch.ones_like(alpha),target["coverage_C"],target["coverage_q_C"])
    keep=masked_alpha_mse(alpha,baseline["alpha"][0],target["outside"])
    retained=masked_alpha_mse(alpha,baseline["alpha"][0],target["K"])
    return (a+weights["alpha_C"]*c+weights["outside_alpha"]*keep+weights["retained_alpha"]*retained),dict(
        alpha_A=a,alpha_C=c,outside_alpha=keep,retained_alpha=retained)


def default_loss_weights():
    return dict(C_photo=.5,edge=.1,C_edge_relative=.1,alpha_C=.5,outside_RGB=1.,retained_RGB=4.,
        outside_alpha=1.,retained_alpha=4.,RGB_prior_squared=.001)


def detached_metric(values):
    return {name:float(value.detach()) for name,value in values.items()}


def verify_supervision(directory):
    directory=Path(directory)
    frozen=json.loads((directory/"freeze.json").read_text())
    if frozen.get("schema")!="immutable_direct_teacher_supervision.v1":raise ValueError("不是固定直接教师监督")
    for item in frozen["source_files"]+frozen["target_files"]+[frozen["manifest"]]:
        if file_digest(item["path"])!=item["sha256"]:raise ValueError("监督内容已改变: "+item["path"])
    return frozen


def prepare_targets(data,device,role):
    def boolean(name):return torch.as_tensor(np.asarray(data[name],bool).copy(),device=device)
    a=boolean("A_mask");c=boolean("C_mask")
    coverage_a=boolean("coverage_A") if "coverage_A" in data else boolean("A_visible")
    coverage_c=boolean("coverage_C") if "coverage_C" in data else boolean("C_teacher_hole")&boolean("F_control_valid")
    k=boolean("K");outside=~(coverage_a|coverage_c|k)
    q_a=torch.as_tensor(data["q_A"].copy(),device=device,dtype=torch.float32)
    q_c=torch.as_tensor(data["q_C"].copy(),device=device,dtype=torch.float32)
    q_ca=torch.as_tensor(data.get("q_A_coverage",data.get("q_coverage_A",data["q_A_native_footprint"])).copy(),device=device,dtype=torch.float32)*coverage_a
    q_cc=torch.as_tensor(data.get("q_C_coverage",data.get("q_coverage_C",data["q_C_native_footprint"])).copy(),device=device,dtype=torch.float32)*coverage_c
    target=dict(target_A_srgb=rgb_tensor(data["target_A_srgb"],device),target_C_srgb=rgb_tensor(data["target_C_srgb"],device),
        selected_C_srgb=rgb_tensor(data["target_C_srgb"] if role=="direct_C" else data["F_control_C_srgb"],device),
        photo_A=a,photo_C=c,q_A=q_a,q=q_c,coverage_A=coverage_a,coverage_C=coverage_c,
        coverage_q_A=q_ca,coverage_q_C=q_cc,K=k,outside=outside,
        fixed_initial_alpha=torch.as_tensor(data["fixed_initial_alpha"].copy(),device=device,dtype=torch.float32))
    if bool((a&c).any()) or bool(((a|c)&k).any()) or bool((outside&(coverage_a|coverage_c)).any()):raise ValueError("A/C/K/outside域不一致")
    return target


def conservative_retained_guard(scene,provider,active_count,scale_upper_factor,frame_rows):
    """实际最大sigma的CUDA radii包围盒与K相交即冻结，独立于互相遮挡。

    单独的all-max alphaT可能由前排遮住后排而漏掉中间状态；半径包围盒
    保守保护车后的road行，随后还必须逐stage验证实际joint像素。
    """
    from PIL import Image
    frames={row["key"]:row for row in frame_rows}
    guard=np.zeros(active_count,bool);records=[];masks={}
    for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
        camera=scene.camera_provider(key);packet=provider.rows(key,camera,scene.pipe)
        scale=packet["scales"].detach().clone()
        scale[:,:2]*=torch.as_tensor(np.maximum(scale_upper_factor,1),device=scale.device,dtype=scale.dtype)[:,None]
        with torch.no_grad():
            out=scene._bindings().raster(camera,scene.pipe,0,dict(packet,scales=scale),scene._background)
            radius=out["radii"].detach().cpu().numpy()
            point=packet["means3D"]
            clip=torch.cat([point,torch.ones((len(point),1),device=point.device,dtype=point.dtype)],dim=1)@camera.full_proj_transform
            xy=(clip[:,:2]/clip[:,3:4]).detach().cpu().numpy()
            xy=(xy+1)*[camera.image_width/2,camera.image_height/2]-.5
        labels=np.asarray(Image.open(frames[key]["instance_mask"]).resize((camera.image_width,camera.image_height),Image.Resampling.NEAREST))
        mask=(labels>0)&(labels!=10);masks[key]=mask
        integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        # 当前实际diff_gaussian_rasterization源码getRect以16×16 tile分桶；
        # 光栅在整tile内仅按alpha阈值裁剪，不能只保护3sigma圆或裸radii盒。
        # 源安装direct_url指向drivingeditor_repro/deps/diff-gaussian-rasterization。
        tile=16
        lo=np.maximum(np.trunc((xy-radius[:,None]-1)/tile).astype(np.int64),0)*tile
        exclusive=np.maximum(np.trunc((xy+radius[:,None]+tile)/tile).astype(np.int64),0)
        hi=exclusive*tile-1
        lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[camera.image_width-1,camera.image_height-1])
        valid=(radius>0)&(hi[:,0]>=lo[:,0])&(hi[:,1]>=lo[:,1])&packet["mask"].detach().cpu().numpy()
        ids=np.flatnonzero(valid);x0,y0=lo[ids].T;x1,y1=hi[ids].T
        count=integral[y1+1,x1+1]-integral[y0,x1+1]-integral[y1+1,x0]+integral[y0,x0]
        selected=ids[count>0];guard[selected]=True
        records.append(dict(raw=key,K_pixels=int(mask.sum()),intersecting_maximum_radius_rows=len(selected)))
    return guard,masks,dict(method="actual native CUDA radii at absolute maximum sigma and16x16 getRect tile-expanded bbox, plus1px projection-rounding margin, intersects full retained mask",
        native_python_module="/root/autodl-tmp/envs/drivingeditor/lib/python3.9/site-packages/diff_gaussian_rasterization/__init__.py",
        native_install_source="/root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization",
        independent_of_road_mutual_occlusion=True,five_frame_union_rows=int(guard.sum()),rows=records)


def setup(args):
    from .persistent_road_handoff import load_persistent_handoff
    from .native_row_contribution import attribute_prepared_rows
    from .road_coverage_training import tangent_scale_scope
    torch.set_num_threads(2)
    torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
    verify_supervision(args.supervision)
    scene,manifest,loader=load_persistent_handoff(args.manifest,args.proof)
    for component in (scene.static,scene.dynamic,scene.environment):
        for parameter in component.parameters() if hasattr(component,"parameters") else ():
            parameter.requires_grad_(False)
    if Path(manifest.road).resolve()!=(Path(args.asset)/"road.npz").resolve():raise ValueError("scene与asset road不一致")
    road=load(manifest.road);prior=load(Path(args.asset)/"provenance.npz")
    r8=load(Path(args.r8_asset)/"road.npz")
    if not all(np.array_equal(road[key],r8[key]) for key in ("model_xyz","normal_world","active")):
        raise ValueError("直接教师拟合要求固定原R8几何/拓扑")
    provider=next(value for value in scene._providers if value.kind=="road")
    targets={};baseline={};packets={};a_mass=np.zeros(int(road["active"].sum()),np.float64);c_mass=a_mass.copy();cov_mass=a_mass.copy()
    cov_a_mass=a_mass.copy();cov_c_mass=a_mass.copy()
    for key in TRAIN_FRAMES:
        data=load(Path(args.supervision)/f"{key:03d}.npz")
        targets[key]=prepare_targets(data,"cuda",args.role)
        camera=scene.camera_provider(key);packet=provider.rows(key,camera,scene.pipe)
        packet={name:value.detach() if torch.is_tensor(value) else value for name,value in packet.items()}
        if not torch.equal(scene._background,torch.zeros_like(scene._background)):raise ValueError("photo必须使用黑背景预乘RGB")
        with torch.no_grad():
            out=scene._bindings().raster(camera,scene.pipe,0,packet,scene._background)
        baseline[key]={name:out[name].detach().clone() for name in ("rgb","alpha")}
        if float((out["alpha"][0]-targets[key]["fixed_initial_alpha"]).abs().max())>1e-6:raise ValueError("fixed initialalpha不属于当前输入资产")
        regions={name:targets[key][mask].detach().cpu().numpy() for name,mask in (("A","photo_A"),("C","photo_C"))}
        regions["coverage"]=(targets[key]["coverage_A"]|targets[key]["coverage_C"]).detach().cpu().numpy()
        regions["coverage_A"]=targets[key]["coverage_A"].detach().cpu().numpy()
        regions["coverage_C"]=targets[key]["coverage_C"].detach().cpu().numpy()
        owners=torch.zeros(len(packet["means3D"]),device="cuda",dtype=torch.long)
        attribution=attribute_prepared_rows(scene,packet,owners,camera,regions)
        a_mass=np.maximum(a_mass,attribution.alpha_mass[:,0]);c_mass=np.maximum(c_mass,attribution.alpha_mass[:,1]);cov_mass=np.maximum(cov_mass,attribution.alpha_mass[:,2])
        cov_a_mass=np.maximum(cov_a_mass,attribution.alpha_mass[:,3]);cov_c_mass=np.maximum(cov_c_mass,attribution.alpha_mass[:,4])
        packets[key]=packet
    active=np.flatnonzero(road["active"]);guard=prior.get("guarded_world_rows",np.zeros(len(road["active"]),bool))
    physical=(prior["surface_supported"]&prior["legal_write"]&~guard&(road.get("minimum_model_depth",np.zeros(len(guard)))<=0))[active]
    known=prior["known"][active]
    color_scope=physical&((a_mass>0)|((c_mass>0)&~known))
    opacity_scope=physical&(cov_mass>0)
    # 绝对包络不能因本轮A/C是否见到某行而缩回1；旧S1已合法增长的
    # 非本轮训练行仍必须保留。先对全物理合法行计算原R8绝对包络。
    physical_world=np.zeros(len(guard),bool);physical_world[active]=physical
    _,r8_upper,_,_=tangent_scale_scope(r8,prior,physical_world,max_factor=1.15)
    absolute_upper=r8["gaussian_sigma_m"][:,:2]*r8_upper[:,None]
    lower=road["gaussian_sigma_m"][:,:2]
    if np.any(lower>absolute_upper+1e-8):raise ValueError("当前S2 sigma已超原R8绝对上界")
    ratio=np.minimum(absolute_upper[:,0]/lower[:,0],absolute_upper[:,1]/lower[:,1])
    bbox_guard,K_masks,guard_receipt=conservative_retained_guard(scene,provider,len(active),ratio[active],manifest.data["frames"])
    physical&=~bbox_guard
    color_scope&=~bbox_guard;opacity_scope&=~bbox_guard
    scale_scope=opacity_scope&(ratio[active]>1.000001)
    packet=packets[TRAIN_FRAMES[0]]
    model=DirectRoadParameters(packet["colors_precomp"],packet["opacities"],packet["scales"],
        np.flatnonzero(color_scope),np.flatnonzero(opacity_scope),np.flatnonzero(scale_scope),ratio[active][scale_scope])
    return dict(scene=scene,manifest=manifest,loader=loader,provider=provider,road=road,prior=prior,r8=r8,
        targets=targets,baseline=baseline,packets=packets,active=active,model=model,
        scope=dict(color=color_scope,opacity=opacity_scope,scale=scale_scope,A_seen=a_mass>0,C_seen=c_mass>0,
            actual_A_mass=a_mass,actual_C_mass=c_mass,actual_coverage_mass=cov_mass,
            actual_A_coverage_mass=cov_a_mass,actual_C_coverage_mass=cov_c_mass,absolute_sigma_upper_m=absolute_upper,
            scale_upper_factor=ratio[active],physical_legal=physical,conservative_K_guard=bbox_guard),role=args.role,
        K_masks=K_masks,guard_receipt=guard_receipt)


def raster(state,key,route="live",colors=None):
    scene=state["scene"];camera=scene.camera_provider(key)
    base=state["packets"].get(key)
    if base is None:base=state["provider"].rows(key,camera,scene.pipe)
    packet=state["model"].packet(base,route)
    if colors is not None:packet=dict(packet,colors_precomp=colors)
    return scene._bindings().raster(camera,scene.pipe,0,packet,scene._background)


def joint_rgb(state,key,*,apply_model=True,maximum=False,color_stress=None):
    scene=state["scene"]
    values,owners,camera,_=scene._rows(key,(10,),(),False)
    values=dict(values,mask=values["mask"]&~scene._contains(owners,(10,)))
    if apply_model:
        model=state["model"];n=len(model.base_rgb);offset=len(values["means3D"])-n
        base=state["provider"].rows(key,camera,scene.pipe)
        packet=model.packet(base,"live")
        if maximum:
            opacity=packet["opacities"].clone();opacity[model.opacity_ids]=1-1e-5
            scales=packet["scales"].clone();scales[model.scale_ids,:2]=model.base_scales[model.scale_ids,:2]*model.scale_upper[:,None]
            packet=dict(packet,opacities=opacity,scales=scales)
        if color_stress is not None:
            colors=packet["colors_precomp"].clone();colors[model.color_ids]=color_stress
            packet=dict(packet,colors_precomp=colors)
        if not torch.equal(values["means3D"][offset:],packet["means3D"]):raise RuntimeError("联合场景road行不是固定完整尾部")
        for name in ("colors_precomp","opacities","scales"):
            if values[name] is None:raise ValueError("当前原生covariance模式不支持显式road scale覆盖")
            value=values[name].clone();value[offset:]=packet[name];values=dict(values,**{name:value})
    out=scene._bindings().raster(camera,scene.pipe,0,values,scene._background)
    env=(1-out["alpha"])*scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    return out["rgb"]+env,out["alpha"]


def export_asset(state,args,output,record):
    from .persistent_road_handoff import save_persistent_road_handoff
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    road={name:value.copy() for name,value in state["road"].items()};model=state["model"];active=state["active"]
    color_world=active[model.color_ids.detach().cpu().numpy()]
    opacity_local=model.opacity_ids.detach().cpu().numpy();changed_opacity=model.opacity_logits.detach().cpu().numpy()>0
    scale_local=model.scale_ids.detach().cpu().numpy();changed_scale=model.scale_factor.detach().cpu().numpy()!=1
    road["reference_srgb"][color_world]=model.colors.detach().cpu().numpy()
    road["road_opacity"][active[opacity_local[changed_opacity]]]=model.full_opacity().detach().cpu().numpy()[opacity_local[changed_opacity],0]
    actual_scales=model.full_scales().detach().cpu().numpy().astype(np.float64)/float(road["meter_to_model"])
    road["gaussian_sigma_m"][active[scale_local[changed_scale]],:2]=actual_scales[scale_local[changed_scale],:2]
    np.savez_compressed(output/"road.npz",**road)
    prior=state["prior"];p={key:value.copy() for key,value in prior.items()}
    if "row_rgb" in p:p["prior_field_row_rgb"]=p.pop("row_rgb")
    p["prior_reference_srgb"]=state["road"]["reference_srgb"].copy()
    p["effective_learned_rgb"]=road["reference_srgb"].copy()
    p["effective_row_rgb"]=road["reference_srgb"].copy()
    p["effective_opacity"]=road["road_opacity"].copy();p["effective_sigma_m"]=road["gaussian_sigma_m"].copy()
    for name in ("color","opacity","scale","A_seen","C_seen","conservative_K_guard"):
        full=np.zeros(len(road["active"]),bool);full[active]=state["scope"][name]
        p["direct_teacher_"+name]=full
    p["asset_mode"]=np.asarray("direct_perGaussian_RGB")
    p["source_prior_role"]=np.asarray("prior F/observations provenance is immutable; effective shader RGB is separately learned_override")
    np.savez_compressed(output/"provenance.npz",**p)
    source=Path(args.asset)
    for old,new in (("atlas.npz","prior_atlas.npz"),("base_atlas.npz","prior_base_atlas.npz"),("texture_correction.npz","prior_texture_correction.npz")):
        if (source/old).exists():shutil.copy2(source/old,output/new)
    write(output/"receipt.json",dict(schema="direct_world_road_asset.v1",asset_mode="direct_perGaussian_RGB",shader_mode="direct_per_gaussian_RGB",
        prior_files="prior_atlas/prior_base_atlas/prior_texture_correction remain frozen priors; none is claimed as current shader field",
        source_asset=str(source.resolve()),color_rows=len(color_world),changed_opacity_rows=int(changed_opacity.sum()),
        changed_scale_rows=int(changed_scale.sum()),training=record,observed_source_prior_preserved=True,
        learned_RGB_not_original_measurements=True,new_theta_or_z_or_eta_parameters=0))
    save_persistent_road_handoff(args.manifest,args.proof,output/"road.npz",output/"persistent",training_record=record)
    return output


def reload_check(state,asset,output):
    from .persistent_road_handoff import load_persistent_handoff
    scene,_,_=load_persistent_handoff(Path(asset)/"persistent/scene.json",Path(asset)/"persistent/proof.json")
    provider=next(value for value in scene._providers if value.kind=="road")
    rows=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            expected=raster(state,key)
            camera=scene.camera_provider(key);packet=provider.rows(key,camera,scene.pipe)
            actual=scene._bindings().raster(camera,scene.pipe,0,packet,scene._background)
            error=float((expected["rgb"]-actual["rgb"]).abs().max());alpha=float((expected["alpha"]-actual["alpha"]).abs().max())
            rgb8=int((torch.round(expected["rgb"]*255)-torch.round(actual["rgb"]*255)).abs().max())
            rows.append(dict(raw=key,RGB_float_max=error,alpha_float_max=alpha,RGB8_max=rgb8))
    scene.assert_persistent_unchanged()
    write(output,dict(public_loader=True,actual_road_raster=True,tolerance=1e-6,rows=rows))
    if any(row["RGB_float_max"]>1e-6 or row["alpha_float_max"]>1e-6 for row in rows):raise RuntimeError("独立公共loader重载偏差超过1e-6")


def objectives(state,key):
    model=state["model"];weights=default_loss_weights()
    photo,pm=photo_loss(raster(state,key,"photo"),state["targets"][key],state["baseline"][key],model.colors,
        model.base_rgb[model.color_ids],weights)
    coverage,cm=coverage_loss(raster(state,key,"coverage"),state["targets"][key],state["baseline"][key],weights)
    return photo,coverage,pm,cm


def probe(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    state=setup(args);model=state["model"]
    np.savez_compressed(output/"row_scopes.npz",active_world_rows=state["active"],**state["scope"])
    before={};initial_error=0.;premult_error=0.;gradient_rows=[]
    joint_baseline={};K_maximum=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            joint_baseline[key]=joint_rgb(state,key,apply_model=False)[0]
            mask=torch.as_tensor(state["K_masks"][key],device="cuda")
            maximum=0.
            for stress in (0.,1.):
                value=joint_rgb(state,key,maximum=True,color_stress=stress)[0]
                maximum=max(maximum,float((value-joint_baseline[key]).abs()[:,mask].max()))
            K_maximum.append(dict(raw=key,K_pixels=int(mask.sum()),maximum_parameters_joint_K_max=maximum))
    if any(row["maximum_parameters_joint_K_max"]!=0 for row in K_maximum):raise RuntimeError("最大参数jointK保护失败")
    for key in TRAIN_FRAMES:
        photo,coverage,pm,cm=objectives(state,key)
        before[key]=dict(photo=float(photo.detach()),coverage=float(coverage.detach()),**detached_metric(pm),**detached_metric(cm))
        gradient_A=torch.autograd.grad(pm["A"],model.colors,retain_graph=True)[0]
        gradient_C=torch.autograd.grad(pm["C"],model.colors,retain_graph=True)[0]
        gradients=torch.autograd.grad(photo,(model.colors,model.opacity_logits,model.scale_factor),allow_unused=True,retain_graph=True)
        coverage_grad=torch.autograd.grad(coverage,(model.colors,model.opacity_logits,model.scale_factor),allow_unused=True)
        row=dict(raw=key,color_gradient_max=float(gradients[0].abs().max()),
            A_gradient_max=float(gradient_A.abs().max()),C_gradient_max=float(gradient_C.abs().max()),
            photo_opacity_gradient=0. if gradients[1] is None else float(gradients[1].abs().max()),
            photo_scale_gradient=0. if gradients[2] is None else float(gradients[2].abs().max()),
            coverage_RGB_gradient=0. if coverage_grad[0] is None else float(coverage_grad[0].abs().max()),
            coverage_opacity_gradient=0. if coverage_grad[1] is None or not coverage_grad[1].numel() else float(coverage_grad[1].abs().max()),
            coverage_scale_gradient=0. if coverage_grad[2] is None or not coverage_grad[2].numel() else float(coverage_grad[2].abs().max()))
        if row["color_gradient_max"]==0 or any(row[name]!=0 for name in ("photo_opacity_gradient","photo_scale_gradient","coverage_RGB_gradient")):
            raise RuntimeError("真实直接教师梯度路由不满足协议")
        gradient_rows.append(row)
        with torch.no_grad():
            live=raster(state,key)
            initial_error=max(initial_error,float((live["rgb"]-state["baseline"][key]["rgb"]).abs().max()),float((live["alpha"]-state["baseline"][key]["alpha"]).abs().max()))
            white=raster(state,key,colors=torch.ones_like(model.base_rgb))
            premult_error=max(premult_error,float((white["rgb"]-white["alpha"]).abs().max()))
    if initial_error!=0 or premult_error>1e-6:raise RuntimeError("initial baseline/premultiplied-alpha审计失败")
    optimizer=torch.optim.Adam([model.colors],lr=.01)
    optimizer.zero_grad(set_to_none=True)
    objective=sum(objectives(state,key)[0] for key in TRAIN_FRAMES)/2
    objective.backward();optimizer.step();model.project()
    after={}
    with torch.no_grad():
        for key in TRAIN_FRAMES:
            photo,coverage,pm,cm=objectives(state,key)
            after[key]=dict(photo=float(photo),coverage=float(coverage),**detached_metric(pm),**detached_metric(cm))
    torch.save(dict(schema="direct_teacher_color_probe.v1",actual_updates=1,model=model.state_dict(),optimizer=optimizer.state_dict(),
        source_manifest=str(args.manifest),source_supervision=str(args.supervision),role=args.role),output/"checkpoint000001.pt")
    K_actual=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            mask=torch.as_tensor(state["K_masks"][key],device="cuda")
            actual=joint_rgb(state,key)[0]
            delta=(actual-joint_baseline[key]).abs().amax(0)
            K_actual.append(dict(raw=key,K_pixels=int(mask.sum()),joint_K_max=float(delta[mask].max()),changed_pixels=int((delta[mask]>0).sum())))
    if any(row["joint_K_max"]!=0 for row in K_actual):raise RuntimeError("1步控制jointK保护失败")
    asset=export_asset(state,args,output/"asset",dict(role="phase0_probe",extra_updates=1,formal_updates=0,teacher_role=args.role))
    reload_check(state,asset,output/"reload.json")
    write(output/"probe.json",dict(schema="direct_teacher_real_color_probe.v1",formal_updates=0,extra_color_updates=1,
        source_supervision=str(args.supervision),initial_render_max_difference=initial_error,premult_white_minus_alpha_max=premult_error,
        gradient_rows=gradient_rows,before=before,after=after,color_rows=int(len(model.color_ids)),
        opacity_rows=int(len(model.opacity_ids)),scale_rows=int(len(model.scale_ids)),
        shader="native black-background premult sRGB; no RGB/alpha",known_A_seen_trainable=True,C_only_known_frozen=True,
        retained_maximum=K_maximum,retained_actual=K_actual,guard_receipt=state["guard_receipt"]))
    state["scene"].assert_persistent_unchanged()
    print(json.dumps(dict(probe=str(output/"probe.json"),before=before,after=after,gradient_rows=gradient_rows)),flush=True)


def stage_for_step(step,schedule):
    if not 1<=step<=sum(schedule):raise ValueError("step不在固定阶段预算内")
    return ("color" if step<=schedule[0] else "coverage" if step<=sum(schedule[:2]) else "joint")


def make_optimizer(model):
    return torch.optim.Adam([dict(params=[model.colors],lr=.01,name="color"),
        dict(params=[model.opacity_logits],lr=.05,name="opacity"),
        dict(params=[model.scale_factor],lr=.005,name="scale")],betas=(.9,.999),eps=1e-8)


def set_stage(optimizer,stage):
    rates={"color":(.01,0.,0.),"coverage":(0.,.05,.005),"joint":(.003,.05,.005)}[stage]
    for group,rate in zip(optimizer.param_groups,rates):group["lr"]=rate


def training_binding(args,schedule):
    from . import road_asset_supervision,persistent_road_handoff,native_row_contribution,road_coverage_training
    frozen=verify_supervision(args.supervision)
    paths=[args.manifest,args.proof,Path(args.asset)/"road.npz",Path(args.asset)/"provenance.npz",
        Path(args.r8_asset)/"road.npz",Path(args.supervision)/"freeze.json",Path(__file__).resolve(),
        Path(road_asset_supervision.__file__).resolve(),Path(persistent_road_handoff.__file__).resolve(),
        Path(native_row_contribution.__file__).resolve(),Path(road_coverage_training.__file__).resolve()]
    paths.extend(item["path"] for item in frozen["source_files"]+frozen["target_files"]+[frozen["manifest"]])
    if args.protocol:paths.append(args.protocol)
    binding={str(Path(path).resolve()):file_digest(path) for path in paths}
    return dict(schema="direct_teacher_frozen_training_contract.v1",role=args.role,seed=101,
        formal=args.mode=="formal",schedule=list(schedule),train_frames=list(TRAIN_FRAMES),diagnostic_only=list(DIAGNOSTIC_FRAMES),
        input_binding=binding,loss=default_loss_weights(),optimizer=dict(name="Adam",betas=[.9,.999],eps=1e-8,
            stage_rates=dict(color=[.01,0.,0.],coverage=[0.,.05,.005],joint=[.003,.05,.005])),
        photo="native black-background premult sRGB; photo raster detaches opacity AND scales; no RGB/alpha",
        coverage="RGB detached; full legal Avisible and Cteacher geometry, independent of initial alpha or F color",
        shader="direct_per_gaussian_RGB; immutable source-prior ledger distinct from learned RGB override",
        constraints="opacity effective baseline lower to1-1e-5; tangent scales original R8 absolute1.15/world-footprint envelope; tile-expanded K guard")


def evaluate_state(state,evaluation_directory):
    rows=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            out=raster(state,key);alpha=out["alpha"][0]
            frozen=load(Path(evaluation_directory)/f"{key:03d}.npz")["frozen_evaluation_domain"].astype(bool)
            mask=torch.as_tensor(frozen,device=alpha.device)
            row=dict(raw=key,fixed_road_pixels=int(mask.sum()),supported_alpha_gt_point9=int((mask&(alpha>.9)).sum()),
                gap_alpha_lt_point05=int((mask&(alpha<.05)).sum()),thin_alpha_point05_to_point9=int((mask&(alpha>=.05)&(alpha<=.9)).sum()))
            if key in TRAIN_FRAMES:
                target=state["targets"][key]
                row.update(A=float(masked_rgb_l1(out["rgb"],alpha[None]*target["target_A_srgb"],target["photo_A"],target["q_A"])),
                    C_actual_teacher=float(masked_rgb_l1(out["rgb"],alpha[None]*target["target_C_srgb"],target["photo_C"],target["q"])),
                    C_selected_objective=float(masked_rgb_l1(out["rgb"],alpha[None]*target["selected_C_srgb"],target["photo_C"],target["q"])),
                    photo_A_pixels=int(target["photo_A"].sum()),photo_C_pixels=int(target["photo_C"].sum()),
                    coverage_A_pixels=int(target["coverage_A"].sum()),coverage_C_pixels=int(target["coverage_C"].sum()))
            rows.append(row)
    return dict(rows=rows,fixed_road_pixels=sum(row["fixed_road_pixels"] for row in rows),
        supported_alpha_gt_point9=sum(row["supported_alpha_gt_point9"] for row in rows),
        diagnostic_frames_never_used_in_loss=list(DIAGNOSTIC_FRAMES))


def joint_K_check(state,baseline):
    rows=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            mask=torch.as_tensor(state["K_masks"][key],device="cuda")
            delta=(joint_rgb(state,key)[0]-baseline[key]).abs().amax(0)
            rows.append(dict(raw=key,K_pixels=int(mask.sum()),joint_K_max=float(delta[mask].max()),changed_pixels=int((delta[mask]>0).sum())))
    if any(row["joint_K_max"]!=0 for row in rows):raise RuntimeError("stage真实joint K非exact零")
    return dict(protected_pixel_frames=sum(row["K_pixels"] for row in rows),all_exact=True,rows=rows)


def maximum_joint_K_check(state,baseline):
    rows=[]
    with torch.no_grad():
        for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES):
            mask=torch.as_tensor(state["K_masks"][key],device="cuda");maximum=0.
            for stress in (0.,1.):
                value=joint_rgb(state,key,maximum=True,color_stress=stress)[0]
                maximum=max(maximum,float((value-baseline[key]).abs()[:,mask].max()))
            rows.append(dict(raw=key,K_pixels=int(mask.sum()),joint_K_max=maximum))
    if any(row["joint_K_max"]!=0 for row in rows):raise RuntimeError("maximum真实joint K非exact零")
    return dict(rows=rows,all_exact=True,conservative_guard=state["guard_receipt"])


def real_gradient_probe(state):
    model=state["model"];parameters=(model.colors,model.opacity_logits,model.scale_factor);rows=[]
    def maximum(gradient):return 0. if gradient is None or not gradient.numel() else float(gradient.abs().max())
    for key in TRAIN_FRAMES:
        photo,coverage,pm,cm=objectives(state,key)
        pg=torch.autograd.grad(photo,parameters,allow_unused=True,retain_graph=True)
        ag=torch.autograd.grad(pm["A"],model.colors,retain_graph=True)[0]
        cg=torch.autograd.grad(pm["C"],model.colors,retain_graph=True)[0]
        kg=torch.autograd.grad(coverage,parameters,allow_unused=True)
        target=state["targets"][key];thin=target["coverage_C"]&~target["photo_C"]
        rendered=raster(state,key,"coverage")
        thin_loss=masked_alpha_mse(rendered["alpha"][0],torch.ones_like(rendered["alpha"][0]),thin,target["coverage_q_C"])
        tg=torch.autograd.grad(thin_loss,parameters,allow_unused=True)
        row=dict(raw=key,A_color_gradient_max=maximum(ag),C_color_gradient_max=maximum(cg),
            photo_color_gradient_max=maximum(pg[0]),photo_opacity_gradient_max=maximum(pg[1]),photo_scale_gradient_max=maximum(pg[2]),
            coverage_RGB_gradient_max=maximum(kg[0]),coverage_opacity_gradient_max=maximum(kg[1]),coverage_scale_gradient_max=maximum(kg[2]),
            C_coverage_outside_photo_pixels=int(thin.sum()),C_coverage_outside_photo_weight=float(target["coverage_q_C"][thin].sum()),
            C_coverage_outside_photo_opacity_gradient_max=maximum(tg[1]),C_coverage_outside_photo_scale_gradient_max=maximum(tg[2]))
        if row["photo_color_gradient_max"]==0 or any(row[name]!=0 for name in ("photo_opacity_gradient_max","photo_scale_gradient_max","coverage_RGB_gradient_max")):
            raise RuntimeError("真实梯度路由probe失败")
        rows.append(row)
    return dict(actual_native_raster=True,rows=rows,diagnostic_frames_used_in_loss=False)


def train(args):
    schedule=tuple(int(value) for value in args.schedule.split(","))
    if len(schedule)!=3 or any(value<=0 for value in schedule):raise ValueError("schedule须为三个正整数")
    if args.mode=="formal":
        if schedule!=(200,300,300):raise ValueError("正式预算必须200/300/300")
        if not args.protocol:raise ValueError("正式必须有根代理锁定协议")
        protocol=json.loads(Path(args.protocol).read_text())
        if protocol.get("formal_authorized") is not True:raise ValueError("根代理尚未授权formal")
    total=sum(schedule);stop=total if args.stop_step is None else args.stop_step
    if not 0<=stop<=total:raise ValueError("stop-step超出固定预算")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    state=setup(args);model=state["model"];optimizer=make_optimizer(model)
    contract=training_binding(args,schedule);write(output/"frozen_contract.json",contract)
    np.savez_compressed(output/"row_scopes.npz",active_world_rows=state["active"],**state["scope"])
    write(output/"guard_receipt.json",state["guard_receipt"])
    raw=state["road"]["road_opacity"][state["active"]]
    effective=model.base_opacity.detach().cpu().numpy()
    write(output/"effective_opacity_prior.json",dict(raw_prior_min=float(raw.min()),raw_prior_max=float(raw.max()),
        effective_min=float(effective.min()),effective_max=float(effective.max()),raw_vs_effective_max=float(np.max(np.abs(raw-effective))),
        upper=1-1e-5,frozen_raw_fields_not_rewritten=True))
    sampler=random.Random(101);queue=[];trace=[];start=0
    def snapshot(step):
        return dict(schema="direct_teacher_complete_state.v1",step=step,model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),sampler=sampler.getstate(),queue=queue.copy(),trace=trace.copy(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),contract=contract,formal_run=args.mode=="formal",
            next_stage=stage_for_step(step+1,schedule) if step<total else "complete")
    if args.resume:
        restored=torch.load(args.resume,map_location="cpu")
        if restored.get("schema")!="direct_teacher_complete_state.v1" or restored["contract"]!=contract:raise ValueError("恢复合同/来源/算法已改变")
        start=restored["step"]
        if type(start) is not int or not 0<=start<=stop:raise ValueError("恢复step无效")
        current=model.state_dict();parameter_keys=set(dict(model.named_parameters()))
        if current.keys()!=restored["model"].keys():raise ValueError("恢复模型字段无效")
        for key,value in restored["model"].items():
            if value.shape!=current[key].shape or not bool(torch.isfinite(value).all()):raise ValueError("恢复参数shape/finite无效")
            if key not in parameter_keys and not torch.equal(value.cpu(),current[key].cpu()):raise ValueError("恢复试图改变冻结buffer:"+key)
        model.load_state_dict(restored["model"]);optimizer.load_state_dict(restored["optimizer"])
        sampler.setstate(restored["sampler"]);queue=restored["queue"].copy();trace=restored["trace"].copy()
        if any(key not in TRAIN_FRAMES for key in queue) or len(trace)!=start:raise ValueError("恢复sampler/trace无效")
        torch.set_rng_state(restored["torch_rng"]);torch.cuda.set_rng_state_all(restored["cuda_rng"])
        np.random.set_state(restored["numpy_rng"]);random.setstate(restored["python_rng"])
        mismatch=differences(restored,snapshot(start))
        if mismatch:raise RuntimeError("完整状态恢复不一致:"+str(mismatch))
        write(output/"restore.json",dict(full_state_restored=True,step=start,differences=mismatch,
            model_optimizer_paramgroups_sampler_queue_trace_all_RNG_exact=True,next_stage=restored["next_stage"]))
    else:torch.save(snapshot(0),output/"checkpoint000000.pt")
    write(output/"gradient_probe.json",real_gradient_probe(state))
    # 评价不改变采样器/随机状态；保留原始五帧道路评价分母。
    write(output/"evaluation_before.json",evaluate_state(state,args.evaluation))
    with torch.no_grad():baseline_joint={key:joint_rgb(state,key,apply_model=False)[0] for key in (*TRAIN_FRAMES,*DIAGNOSTIC_FRAMES)}
    write(output/"maximum_joint_K.json",maximum_joint_K_check(state,baseline_joint))
    records=[];entries=[];begin=time.monotonic();last_stage=None
    boundaries=np.cumsum(schedule).tolist()
    for step in range(start+1,stop+1):
        stage=stage_for_step(step,schedule)
        if stage!=last_stage:
            set_stage(optimizer,stage)
            entries.append(dict(step=step,stage=stage,learning_rates=[group["lr"] for group in optimizer.param_groups]))
            last_stage=stage
        if not queue:queue=list(TRAIN_FRAMES);sampler.shuffle(queue)
        key=queue.pop();optimizer.zero_grad(set_to_none=True)
        if stage=="color":
            value,metrics=photo_loss(raster(state,key,"photo"),state["targets"][key],state["baseline"][key],model.colors,model.base_rgb[model.color_ids],default_loss_weights())
        elif stage=="coverage":
            value,metrics=coverage_loss(raster(state,key,"coverage"),state["targets"][key],state["baseline"][key],default_loss_weights())
        else:
            photo,coverage,pm,cm=objectives(state,key);value=photo+coverage;metrics=dict(**pm,**cm)
        if not bool(torch.isfinite(value)):raise FloatingPointError("训练loss非finite")
        value.backward()
        for name,parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):raise FloatingPointError("非finite梯度:"+name)
        if stage=="color" and (model.opacity_logits.grad is not None or model.scale_factor.grad is not None):raise RuntimeError("photo泄露coverage梯度")
        if stage=="coverage" and model.colors.grad is not None:raise RuntimeError("coverage泄露RGB梯度")
        gradient_max={name:0. if parameter.grad is None or not parameter.grad.numel() else float(parameter.grad.abs().max()) for name,parameter in model.named_parameters()}
        optimizer.step();model.project();trace.append(key)
        record=dict(step=step,stage=stage,raw=key,loss=float(value.detach()),gradient_max=gradient_max,**detached_metric(metrics))
        records.append(record)
        if step==start+1 or step%25==0:print(json.dumps(record),flush=True)
        if step in boundaries or step%100==0 or step==stop:
            torch.save(snapshot(step),output/f"checkpoint{step:06d}.pt")
        if step in boundaries:
            stage_dir=output/("stage_"+stage);stage_dir.mkdir()
            write(stage_dir/"joint_K.json",joint_K_check(state,baseline_joint))
            write(stage_dir/"evaluation.json",evaluate_state(state,args.evaluation))
            exported=export_asset(state,args,stage_dir/"asset",dict(role=args.role,mode=args.mode,step=step,stage=stage,
                actual_updates=step-start,formal_updates=step if args.mode=="formal" else 0,source_contract=str(output/"frozen_contract.json")))
            if args.mode!="formal":reload_check(state,exported,stage_dir/"reload.json")
            print(json.dumps(dict(stage_complete=stage,step=step,persistent=str(exported/"persistent"))),flush=True)
    write(output/"stage_entries.json",entries);write(output/"training.json",records)
    write(output/"evaluation_after.json",evaluate_state(state,args.evaluation))
    write(output/"joint_K_after.json",joint_K_check(state,baseline_joint))
    state["scene"].assert_persistent_unchanged()
    write(output/"completed.json",dict(completed=stop==total,actual_updates=stop-start,step=stop,formal=args.mode=="formal",schedule=schedule,
        checkpoint=str(output/f"checkpoint{stop:06d}.pt"),elapsed_seconds=time.monotonic()-begin,seed=101,teacher_role=args.role))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","proof","asset","r8-asset","supervision","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--role",choices=("direct_C","matched_F_control"),default="direct_C")
    parser.add_argument("--mode",choices=("probe","smoke","formal"),default="probe")
    parser.add_argument("--schedule",default="200,300,300")
    parser.add_argument("--stop-step",type=int)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--evaluation",type=Path,default=Path("/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase0/supervision_triangle_r8"))
    args=parser.parse_args()
    probe(args) if args.mode=="probe" else train(args)


if __name__=="__main__":main()
