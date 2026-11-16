"""CPU准备：固定道路上的逐高斯RGB修正与共享世界XY纹理修正。

本模块不导入原生renderer，不调用加速器API，不启动训练。输入/参数默认CPU，
共享场是冻结c0以外的RGB修正，不把原先验或伪教师称为隐藏道路真值。
"""
from pathlib import Path
import json
import os
import numpy as np
import torch


def require_cpu_only_environment():
    """只检查进程环境，不探测任何加速器或设备驱动。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU准备须显式设置CUDA_VISIBLE_DEVICES为空；不执行设备探测")


import importlib.util
import sys
_basis_spec=importlib.util.spec_from_file_location('road_world_texture_basis_cpu',Path(__file__).with_name('road_world_texture_basis.py'))
_basis_module=importlib.util.module_from_spec(_basis_spec)
sys.modules[_basis_spec.name]=_basis_module
_basis_spec.loader.exec_module(_basis_module)
for _name in ('bilinear_basis','ColorBasisPlan','make_color_basis_plan','basis_statistics'):
    globals()[_name]=getattr(_basis_module,_name)


class RoadColorParameterization(torch.nn.Module):
    """同c0+零修正初始化；shared_world_4cm与independent_ci只改变basis。"""
    def __init__(self,plan,mode):
        super().__init__()
        if mode not in ("independent_ci","shared_world"):raise ValueError("未知颜色basis")
        self.plan=plan;self.mode=mode
        self.register_buffer("base_srgb",torch.from_numpy(plan.base_srgb.copy()))
        self.register_buffer("eligible",torch.from_numpy(plan.eligible.copy()))
        self.register_buffer("corner_ids",torch.from_numpy(plan.corner_ids.copy()))
        self.register_buffer("weights",torch.from_numpy(plan.weights.copy()))
        parameter_ids=np.flatnonzero(plan.eligible if mode=="independent_ci" else plan.mutable_texels)
        self.register_buffer("parameter_ids",torch.from_numpy(parameter_ids))
        self.delta_values=torch.nn.Parameter(torch.zeros((len(parameter_ids),3),dtype=torch.float32))

    def full_texel_delta(self):
        if self.mode!="shared_world":raise ValueError("独立ci没有连续世界纹理")
        field=self.delta_values.new_zeros((self.plan.shape[0]*self.plan.shape[1],3))
        return field.index_copy(0,self.parameter_ids,self.delta_values)

    def row_delta(self):
        if self.mode=="independent_ci":
            delta=self.delta_values.new_zeros(self.base_srgb.shape).index_copy(0,self.parameter_ids,self.delta_values)
        else:
            field=self.full_texel_delta()
            delta=(field[self.corner_ids].to(self.weights.dtype)*self.weights[...,None]).sum(1).to(self.base_srgb.dtype)
        return delta*self.eligible[:,None]

    def affine_colors(self):
        """未clip的c0+Bδ，供固定coverage线性容量与约束求解接口。"""
        return self.base_srgb+self.row_delta()

    def forward(self):
        return self.affine_colors().clamp(0,1)

    def project(self):
        # 两arm都只把最终shader RGB限制到[0,1]；不加入旧±.08修正上限。
        # I0逐行delta的该界冗余；W4节点界可能缩小共享场可达集合，属于显式方法约束。
        with torch.no_grad():self.delta_values.clamp_(-1,1)

    def query_world_delta(self,world_xy):
        if self.mode!="shared_world":raise ValueError("independent_ci不提供世界连续查询")
        ids,weight,inside,_=bilinear_basis(world_xy,self.plan.origin_xy,self.plan.spacing_m,self.plan.shape)
        field=self.full_texel_delta()
        index=torch.as_tensor(ids,device=field.device,dtype=torch.long)
        weights=torch.as_tensor(weight,device=field.device,dtype=torch.float64)
        value=(field[index].to(weights.dtype)*weights[...,None]).sum(1).to(field.dtype)
        return value,inside


def world_delta_tv(model,edges,distance_m,reference_length_m=.04):
    """条件I_reg的同世界邻接TV，只作用于修正，保留冻结c0细节。"""
    index=torch.as_tensor(edges,device=model.delta_values.device,dtype=torch.long)
    distance=torch.as_tensor(distance_m,device=model.delta_values.device,dtype=torch.float64)
    if index.ndim!=2 or index.shape[1]!=2 or distance.shape!=(len(index),) or bool((distance<=0).any()):
        raise ValueError("米制邻接须为N×2索引和正距离")
    if not len(index):return model.delta_values.sum()*0
    delta=model.row_delta()
    return ((delta[index[:,0]]-delta[index[:,1]]).abs().to(distance.dtype)*(reference_length_m/distance)[:,None]).mean()


def fixed_coverage_teacher_l1(premult_rgb,fixed_alpha,target_srgb,mask,q):
    """相同固定2D教师损失接口；不使用F0教师、不除以alpha。"""
    alpha=fixed_alpha.detach()
    if alpha.ndim==3:alpha=alpha[0]
    weight=mask.to(premult_rgb.dtype)*q
    return ((premult_rgb-alpha[None]*target_srgb).abs()*weight[None]).sum()/(3*weight.sum().clamp_min(1))


def separated_real_teacher_loss(premult_rgb,fixed_alpha,target_pack,weights):
    """A/C/S分域独立归一化；S权重必须显式给出，candidate不自动获许可。"""
    if set(weights)!={"A","C","S"}:raise ValueError("必须显式登记A/C/S全部权重，不能默认打开shadow loss")
    keys={"A":("A_mask","q_A","target_A_srgb"),"C":("C_hole_photo","q_C_hole","target_C_srgb"),
        "S":("S_target_shadow_photo","q_S","target_S_srgb")}
    components={name:fixed_coverage_teacher_l1(premult_rgb,fixed_alpha,target_pack[rgb],target_pack[mask],target_pack[q])
        for name,(mask,q,rgb) in keys.items()}
    return sum(float(weights[name])*value for name,value in components.items()),components


def color_only_native_packet(frozen_packet,model):
    """未来经用户授权后可接实际光栅；本函数不导入/调用renderer。"""
    if model.plan.source_role not in ("synthetic_unit_test","qualified_common_actual_alphaT"):
        raise ValueError("当前scope仅静态预览，不能接成新target的渲染训练许可")
    rgb=model()
    if len(rgb)!=len(frozen_packet["means3D"]):raise ValueError("固定native行与世界颜色行数不同")
    return dict(frozen_packet,colors_precomp=rgb)


def save_parameterization(model,directory,*,metadata=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model.plan.save(directory/"basis_plan.npz")
    np.savez_compressed(directory/"color_parameters.npz",delta_values=model.delta_values.detach().cpu().numpy(),mode=np.asarray(model.mode))
    manifest=dict(schema="road_color_parameterization.v1",mode=model.mode,spacing_m=model.plan.spacing_m,
        field_role="frozen baseline c0 plus learned world-XY bilinear correction" if model.mode=="shared_world" else "frozen c0 plus independent row correction",
        shader="reference camera sRGB appearance, not physical reflectance",coordinate_units="meters",UV="integer texel centers",
        interpolation="float64 metric weights and accumulation, float32 shader RGB",row_override="ineligible and protected rows use exact frozen c0",
        learnable_parameter_scalars=model.delta_values.numel(),geometry_or_opacity_or_scale_parameters=0,
        scope_source=model.plan.scope_source,source_role=model.plan.source_role,
        GPU_executed=False,metadata=metadata or {})
    (directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")


def load_parameterization(directory):
    directory=Path(directory);plan=ColorBasisPlan.load(directory/"basis_plan.npz")
    manifest=json.loads((directory/"manifest.json").read_text())
    with np.load(directory/"color_parameters.npz",allow_pickle=False) as saved:
        mode=str(saved["mode"]);delta=saved["delta_values"].copy()
    model=RoadColorParameterization(plan,mode)
    if manifest["mode"]!=mode or delta.shape!=model.delta_values.shape or not np.isfinite(delta).all() or np.any(np.abs(delta)>1):
        raise ValueError("持久颜色参数shape/finite/bounds不一致")
    with torch.no_grad():model.delta_values.copy_(torch.from_numpy(delta))
    return model,manifest


def bake_road_colors(source_road,output_path,model):
    """仅写共同允许的reference_srgb；其余原始NPZ字段逐值保留。"""
    if model.plan.source_role not in ("synthetic_unit_test","qualified_common_actual_alphaT"):
        raise ValueError("静态预览scope不能写成新target的道路候选")
    output_path=Path(output_path)
    if output_path.exists():raise FileExistsError(output_path)
    with np.load(source_road,allow_pickle=False) as saved:road={key:saved[key].copy() for key in saved.files}
    ids=model.plan.world_row_ids
    if np.any(ids>=len(road["reference_srgb"])) or not np.array_equal(road["reference_srgb"][ids],model.plan.base_srgb):
        raise ValueError("源道路与共同初始化c0不一致")
    selected=model.plan.eligible
    road["reference_srgb"][ids[selected]]=model().detach().cpu().numpy()[selected]
    np.savez_compressed(output_path,**road)
    return dict(updated_rows=int(selected.sum()),only_field="reference_srgb",GPU_executed=False)
