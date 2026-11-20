"""Shared world material -> fixed physical-Sigma average -> true native GS RGB.

No context creation, training loop, implicit lease or private loss is implemented.
The pixel-world shader is deliberately outside this primary render interface.
"""
from pathlib import Path
import json,hashlib,importlib.util,sys
import numpy as np
from scipy import sparse
import torch

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_common_avg_model(field_path,permission_path,runtime_path):
    """Copy A200 PARAMS, inject explicit teacher gate, and retain stageable L/H."""
    saved=torch.load(field_path,map_location='cpu',weights_only=False);state=saved['state'];meta=saved['metadata']
    if meta.get('reader')!='avg'or meta.get('cursor')!=200 or bool((state['high']!=0).any()):
        raise ValueError('requires common avg A200 with exactly zero H, never pixel/M11 warm start')
    with np.load(permission_path,allow_pickle=False)as z:
        keys=['common_base_srgb','common_material_valid','mH_union','common_latent_mask','common_reg_support']
        p={k:z[k].copy()for k in keys}
    expected={'base':p['common_base_srgb'],'valid':p['common_material_valid'][...,None],
              'latent_mask':p['common_latent_mask'][...,None],'reg_support':p['common_reg_support'][...,None]}
    for k,v in expected.items():
        if not np.array_equal(state[k].numpy(),v):raise ValueError('common buffer differs: '+k)
    spec=importlib.util.spec_from_file_location('native_capacity_frozen_world_runtime',runtime_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=module.WorldTextureRuntime(p['common_base_srgb'],p['common_material_valid'],p['mH_union'],
        p['common_latent_mask'],p['common_reg_support'],low_mask=state['low_mask'][...,0].numpy(),**saved['settings'])
    with torch.no_grad():model.low.copy_(state['low']);model.high.zero_()
    model.set_stage('B') # H-only first; caller sets C for L+H. No loss/steps here.
    binding=dict(source_common_avg_A200=str(field_path),source_common_avg_A200_SHA256=file_sha(field_path),
        permission=str(permission_path),permission_SHA256=file_sha(permission_path),runtime=str(runtime_path),runtime_SHA256=file_sha(runtime_path),
        H_initial_exact_zero=True,low_initial_exact_common_A200=True,gate='explicit final teacher mH_union; never restored from A200 buffers',
        stages='B: H-only; C: L+H',extra_H_source_filter='none; nominal lattice Nyquist and frozen physical Sigma AA remain',
        private_regularizers_added=False,ordinary_GS_primary=True)
    return model,binding


class SharedAverageGSRead:
    """Fixed CSR B+KL L+KH H. KH already includes teacher mH; never gate twice."""
    def __init__(self,base,KL,KH,row_ids,active_world_rows,eligible_active,parent_rgb,*,low_shape,high_shape,source=None):
        self.row_ids=np.asarray(row_ids,np.int64).copy();self.active_world_rows=np.asarray(active_world_rows,np.int64).copy()
        eligible=np.asarray(eligible_active);parent=np.asarray(parent_rgb,np.float32)
        if eligible.dtype!=bool or eligible.shape!=self.active_world_rows.shape or parent.shape!=(len(eligible),3):
            raise ValueError('explicit active row/scope/parent shapes required')
        if np.any(np.diff(self.active_world_rows)<=0)or len(np.unique(self.row_ids))!=len(self.row_ids):raise ValueError('global world row identity is ambiguous')
        if not np.array_equal(np.sort(self.row_ids),self.active_world_rows[eligible]):
            raise ValueError('operator rows must equal explicit common scope; recompile for another scope, never silently shrink')
        positions=np.searchsorted(self.active_world_rows,self.row_ids)
        if not np.array_equal(self.active_world_rows[positions],self.row_ids):raise ValueError('operator row absent from active topology')
        self.base=np.asarray(base,np.float64).copy();self.KL=sparse.csr_matrix(KL,dtype=np.float64);self.KH=sparse.csr_matrix(KH,dtype=np.float64)
        self.low_shape=tuple(low_shape);self.high_shape=tuple(high_shape);self.source=dict(source or {})
        n=len(self.row_ids)
        if self.base.shape!=(n,3)or self.KL.shape!=(n,int(np.prod(self.low_shape)))or self.KH.shape!=(n,int(np.prod(self.high_shape))):raise ValueError('compiled matrix dimensions differ')
        if not all(np.isfinite(x).all()for x in [self.base,self.KL.data,self.KH.data,parent]):raise ValueError('nonfinite fixed read')
        self.parent_rgb=parent.copy();self.eligible=eligible.copy();self.positions=positions
        self._device_cache={}

    @classmethod
    def from_operator(cls,root,active_world_rows,eligible_active,parent_rgb,gate='teacher'):
        root=Path(root);d=json.loads((root/'read.json').read_text())
        if gate not in d['gates']:raise ValueError('requested gate not in frozen physical read')
        with np.load(root/'constants.npz',allow_pickle=False)as z:base=z['base'];ids=z['row_ids']
        paths=[root/'read.json',root/'constants.npz',root/'low.npz',root/f'high_{gate}.npz']
        return cls(base,sparse.load_npz(paths[2]),sparse.load_npz(paths[3]),ids,active_world_rows,eligible_active,parent_rgb,
            low_shape=d['low_shape'],high_shape=d['high_shape'],source={str(p):file_sha(p)for p in paths})

    def _tensors(self,device):
        key=str(torch.device(device))
        if key not in self._device_cache:
            def csr(a):
                # CSR avoids an additional full20M-row COO row-index allocation.
                return torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int64)),torch.from_numpy(a.indices.astype(np.int64)),
                    torch.from_numpy(a.data),size=a.shape,dtype=torch.float64,device=device,requires_grad=False,check_invariants=True)
            self._device_cache[key]=(torch.as_tensor(self.base,dtype=torch.float64,device=device),csr(self.KL),csr(self.KH),
                torch.as_tensor(self.parent_rgb,device=device),torch.as_tensor(self.positions,device=device))
        return self._device_cache[key]

    def colors(self,model):
        low,high=model.fields()
        if tuple(low.shape[:2])!=self.low_shape or tuple(high.shape[:2])!=self.high_shape:raise ValueError('field shape does not match physical read')
        b,l,h,parent,pos=self._tensors(high.device)
        values=(b+torch.sparse.mm(l,low.reshape(-1,3).double())+torch.sparse.mm(h,high.reshape(-1,3).double())).clamp(0,1).float()
        return parent.index_copy(0,pos,values)

    def render(self,ctx,model,raw,*,joint=False):
        """Actual native raster, not a pixel-world/basis composition."""
        if not np.array_equal(ctx.active_world_rows,self.active_world_rows):raise ValueError('native context topology differs')
        colors=self.colors(model)
        return ctx.render_joint(int(raw),colors)if joint else ctx.render_road(int(raw),colors)

    def initial_ci_numpy(self,model):
        with torch.no_grad():L,H=model.fields()
        if bool((H!=0).any()):raise ValueError('oracle initial ci must use exactly H0')
        values=np.clip(self.base+self.KL@L.detach().cpu().numpy().reshape(-1,3),0,1).astype(np.float32)
        colors=self.parent_rgb.copy();colors[self.positions]=values
        return colors

    def describe(self):
        return dict(renderer='ordinary_GS_physical_Sigma_average_then_native_raster',gate_in_KH=True,
            active_rows=len(self.active_world_rows),scope_rows=int(self.eligible.sum()),KL_nnz=self.KL.nnz,KH_nnz=self.KH.nnz,
            low_shape=list(self.low_shape),high_shape=list(self.high_shape),matrix_dtype='float64; ci cast float32 after one clamp',
            CSR_GPU_backward_qualification='pending real native smoke; CPU fixture is not native proof',source_SHA256=self.source,
            explicit_scope_SHA256=hashlib.sha256(self.active_world_rows[self.eligible].astype('<i8').tobytes()).hexdigest(),
            pixel_world_endpoint_only=True,private_loss_or_regularizer=False)
