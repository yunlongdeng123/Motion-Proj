"""模块2 CPU入口：表面支撑、材质资格、渲染激活是三个独立状态。

源资产 + 显式世界请求 + 已审材质资格 -> ActivationRequest（不激活新行）
候选packet的fresh native K/near证据 -> ActivationDecision -> 既有builder导出

本模块不加载Torch/native，不修改默认scene，不从旧active表推断新增行K安全。
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


def _mask(value,n,name):
    a=np.asarray(value)
    if a.dtype!=np.bool_ or a.shape!=(n,):raise ValueError(name+' 必须为完整world bool向量')
    a=a.copy();a.flags.writeable=False
    return a


def _ids(value,n,name):
    a=np.asarray(value)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or len(np.unique(a))!=len(a):raise ValueError(name+' 必须为唯一world IDs')
    if len(a) and (a.min()<0 or a.max()>=n):raise ValueError(name+' 越界')
    a=np.sort(a.astype(np.int64));a.flags.writeable=False
    return a


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ActivationRequest:
    geometry_support: np.ndarray
    material_qualified: np.ndarray
    parent_render_active: np.ndarray
    requested: np.ndarray
    legacy_protected: np.ndarray
    candidate_world_rows: np.ndarray
    material_qualification_source: str

    @property
    def render_active(self):
        """CPU准备时保持父资产；候选资格不等于已通过native激活门。"""
        return self.parent_render_active


def prepare_request(*,parent_render_active,geometry_support,material_qualified,
                    requested,material_qualification_source,legacy_protected=None):
    """requested是本次显式世界激活意图；不得传旧生成W/coverage_requested当表面存在门。"""
    n=len(parent_render_active)
    active=_mask(parent_render_active,n,'parent_render_active')
    geometry=_mask(geometry_support,n,'geometry_support')
    material=_mask(material_qualified,n,'material_qualified')
    request=_mask(requested,n,'requested')
    protected=_mask(np.zeros(n,bool) if legacy_protected is None else legacy_protected,n,'legacy_protected')
    if not isinstance(material_qualification_source,str) or not material_qualification_source.strip():
        raise ValueError('材质资格必须显式指向来源审查，不能凭有限RGB冒充已知材质')
    candidates=_ids(np.flatnonzero(request&geometry&material&~active&~protected),n,'candidate_world_rows')
    return ActivationRequest(geometry,material,active,request,protected,candidates,material_qualification_source)


@dataclass(frozen=True)
class FreshNativeGate:
    candidate_road_sha256: str
    candidate_world_rows: np.ndarray
    packet_active_world_rows: np.ndarray
    K_guard: np.ndarray
    near_guard: np.ndarray
    frames: tuple
    K_pixel_frames: int
    source_receipt: str


def load_fresh_native_gate(mask_path,receipt_path,*,candidate_road_path,candidate_scene_manifest_path):
    """拒绝旧packet、mask误配、缺帧及body/scene不符的保护表。

    此gate只覆盖当前road body的固定geometry/Σ/opacity，不授权可训练
    scale/opacity最大包络；参数变化后必须产生匹配的新native gate。
    """
    r=json.loads(Path(receipt_path).read_text())
    if r.get('schema')!='module2_fresh_native_gate.v1' or r.get('native_producer_GPU_executed',r.get('GPU_executed')) is not True:
        raise ValueError('需要显式fresh native证据')
    if r.get('candidate_road_sha256')!=_sha(candidate_road_path):raise ValueError('fresh gate绑定的候选road已变化')
    if r.get('mask_sha256')!=_sha(mask_path):raise ValueError('K/near mask文件与fresh gate绑定不符')
    if r.get('scene_manifest_sha256')!=_sha(candidate_scene_manifest_path):raise ValueError('相机/场景身份与fresh gate不符')
    if r.get('fixed_geometry_sigma_opacity') is not True:raise ValueError('该入口只消费与body一致的固定参数footprint')
    if r.get('GPU_executed') is not True and (not r.get('native_producer_receipt') or not r.get('native_producer_receipt_sha256')):
        raise ValueError('CPU桥接必须显式绑定真实native producer receipt及hash')
    if r.get('native_producer_receipt'):
        if r.get('native_producer_receipt_sha256')!=_sha(r['native_producer_receipt']):raise ValueError('原native producer证据发生变化')
        producer=json.loads(Path(r['native_producer_receipt']).read_text())
        if producer.get('GPU_executed') is not True:raise ValueError('CPU桥接不能伪造native producer')
    if r.get('method')!='actual_native_fixed_sigma_tile_footprint_and_near':raise ValueError('K gate不得用旧active-only表或CPU猜测替代')
    with np.load(mask_path,allow_pickle=False) as d:
        K=d['world_K_guard'].copy();near=d['world_near_guard'].copy()
        candidates=d['candidate_world_rows'].copy();packet=d['packet_active_world_rows'].copy()
    return FreshNativeGate(r['candidate_road_sha256'],candidates,packet,K,near,tuple(r['frames']),
        int(r['K_pixel_frames']),str(receipt_path))


@dataclass(frozen=True)
class ActivationDecision:
    geometry_support: np.ndarray
    material_qualified: np.ndarray
    render_active: np.ndarray
    accepted_new: np.ndarray
    K_rejected: np.ndarray
    near_rejected: np.ndarray
    native_gate_source: str


def apply_fresh_gate(request,gate,*,candidate_road_sha256,required_frames,required_K_pixel_frames):
    n=len(request.parent_render_active)
    if gate.candidate_road_sha256!=candidate_road_sha256:raise ValueError('候选road与native gate不同')
    expected=request.candidate_world_rows
    if np.any(~(request.requested&request.geometry_support&request.material_qualified&~request.parent_render_active&~request.legacy_protected)[expected]):
        raise ValueError('候选集合不再满足独立geometry/material/request资格')
    if not np.array_equal(_ids(gate.candidate_world_rows,n,'gate candidates'),expected):raise ValueError('native gate未覆盖同一新增候选集合')
    packet=np.flatnonzero(request.parent_render_active|np.isin(np.arange(n),expected))
    if not np.array_equal(_ids(gate.packet_active_world_rows,n,'gate packet active'),packet):
        raise ValueError('拒绝旧active packet的K表；新增节点必须实际进入本次packet')
    if tuple(gate.frames)!=tuple(required_frames) or gate.K_pixel_frames!=required_K_pixel_frames:
        raise ValueError('native gate帧/完整K分母不符')
    K=_mask(gate.K_guard,n,'K_guard');near=_mask(gate.near_guard,n,'near_guard')
    new=np.zeros(n,bool);new[expected]=True
    accepted=new&~K&~near
    active=request.parent_render_active|accepted
    return ActivationDecision(request.geometry_support,request.material_qualified,
        _mask(active,n,'render_active'),_mask(accepted,n,'accepted_new'),
        _mask(new&K,n,'K_rejected'),_mask(new&near,n,'near_rejected'),gate.source_receipt)


def save_cpu_request(request,output):
    """只保存CPU计划；不导出改变active位的road或公共scene。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'activation_request.npz',geometry_support=request.geometry_support,
        material_qualified=request.material_qualified,parent_render_active=request.parent_render_active,
        render_active=request.render_active,requested=request.requested,legacy_protected=request.legacy_protected,
        candidate_world_rows=request.candidate_world_rows)
    (output/'receipt.json').write_text(json.dumps(dict(schema='module2_activation_request_CPU.v1',
        GPU_executed=False,new_rows_activated=0,candidate_rows=len(request.candidate_world_rows),
        material_qualification_source=request.material_qualification_source,
        native_gate_required=True,default_scene_modified=False),indent=2)+'\n')
