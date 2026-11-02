"""Migrate the historical R2 contribution-based selection to native B30k.

Keeps the successful two-way repair: road colour/alpha AND opacity-only dark
residual rows. Selection is an edit heuristic, never semantic ownership truth.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from editor_common import Native,cpu,prepare_rasterization
from utils.sh_utils import eval_sh
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1/implementation_v1')
sys.path.insert(0,str(OLD))
from shadow_math import project_gaussians,replay,rotation_matrix
RUN=Path(__file__).resolve().parent.parent
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
META=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')

@torch.no_grad()
def main():
    started=time.time();n=Native();m=json.loads(META.read_text());scale=m['meter_to_model']
    plane=np.asarray(m['ground_plane_model']);plane/=np.linalg.norm(plane[:3])
    inv=np.linalg.inv(np.asarray(m['world_to_model'])@np.asarray(m['coordinate_anchor']))
    cloud=np.load(FULL/'full199_ground_pool.npz')['local_xyz'];tree=cKDTree(cloud)
    selected={};records=[]
    for alias,raw,box in [('5',101,(20,205,205,305)),('37',105,(195,235,340,320))]:
        c=n.camera(raw);kw=prepare_rasterization(c,n.static,n.args)
        dirs=torch.nn.functional.normalize(kw['means3D']-c.camera_center,dim=1)
        rgb=kw['colors_precomp'] if kw['colors_precomp'] is not None else (eval_sh(n.static.active_sh_degree,kw['shs'].transpose(1,2),dirs)+.5).clamp_min(0)
        state=dict(xyz=cpu(kw['means3D']),colors=cpu(rgb),scales=cpu(kw['scales']),rotation=cpu(kw['rotations']),
            opacity=cpu(kw['opacities']).ravel(),mask=cpu(kw['mask']))
        camera=dict(world_view_transform=cpu(c.world_view_transform),full_proj_transform=cpu(c.full_proj_transform),
            camera_center=cpu(c.camera_center),FoVx=c.FoVx,FoVy=c.FoVy,width=480,height=320)
        pr=project_gaussians(state,camera,n.args.neg_fov)
        x0,y0,x1,y1=box;lo=np.array([x0//16,y0//16]);hi=np.array([(x1-1)//16,(y1-1)//16])
        keep=(pr['rmax']>lo).all(1)&(pr['rmin']<=hi).all(1)
        pr={k:v[keep] for k,v in pr.items()}
        # Dense reference subset includes both the original artifact area and
        # surrounding road. Every retained row has actual pre-edit alpha*T.
        yy,xx=np.mgrid[y0:y1:2,x0:x1:2];pixels=np.c_[xx.ravel(),yy.ravel()]
        rendered,alpha,mass,probes=replay(pr,pixels,batch=32)
        native=cpu(n.render(c)['rgb']).transpose(1,2,0)[pixels[:,1],pixels[:,0]]
        env=cpu(n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)).transpose(1,2,0)[pixels[:,1],pixels[:,0]]
        mismatch=np.abs(rendered+(1-alpha[:,None])*env-native)
        assert mismatch.mean()<1e-4,dict(alias=alias,mean=float(mismatch.mean()),maximum=float(mismatch.max()))
        height=(pr['xyz']@plane[:3]+plane[3])/scale
        rot=rotation_matrix(pr['rotation']);sigma=np.linalg.norm(np.einsum('nji,j->ni',rot,plane[:3])*pr['scales'],axis=1)/scale
        local=pr['xyz']@inv[:3,:3].T+inv[:3,3];distance,_=tree.query(local)
        visible=mass>1e-9
        road=visible&(np.abs(height)<=.20)&(3*sigma<=.30)&(distance<.45)
        dark=visible&~road&(pr['colors'].max(1)<.05)
        # Receiver priority resolves overlap across the two local anchors.
        for kind,mask in [('artifact',dark),('receiver',road)]:
            for j in np.flatnonzero(mask):
                identity=int(pr['ids'][j])
                if identity not in selected or kind=='receiver' and selected[identity]['kind']!='receiver':
                    selected[identity]=dict(kind=kind,xyz=pr['xyz'][j],rgb=pr['colors'][j],alias=alias,raw=raw)
        row=dict(alias=alias,anchor_raw=raw,box=list(box),projected_rows=len(pr['ids']),receiver_rows=int(road.sum()),
            dark_artifact_rows=int(dark.sum()),receiver_alpha_T_fraction=float(mass[road].sum()/mass.sum()),
            dark_artifact_alpha_T_fraction=float(mass[dark].sum()/mass.sum()),native_replay_MAE=float(mismatch.mean()),seconds=time.time()-started)
        records.append(row);print(json.dumps(row),flush=True)
    ri=np.array(sorted(k for k,v in selected.items() if v['kind']=='receiver'),np.int64)
    ai=np.array(sorted(k for k,v in selected.items() if v['kind']=='artifact'),np.int64)
    assert len(ri)>100 and len(ai)>10
    selection=dict(receiver_ids=ri,artifact_ids=ai,receiver_xyz_anchor=np.array([selected[k]['xyz'] for k in ri]),
        artifact_xyz_anchor=np.array([selected[k]['xyz'] for k in ai]),receiver_rgb_anchor=np.array([selected[k]['rgb'] for k in ri]),
        receiver_base_opacity=cpu(n.static.get_opacity).ravel()[ri],artifact_base_opacity=cpu(n.static.get_opacity).ravel()[ai])
    dest=RUN/'selection';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'selection.npz',**selection)
    receipt=dict(anchors=records,receiver_rows=len(ri),artifact_rows=len(ai),parent_unchanged=n.unchanged(),
        selection_is_heuristic_not_owner_truth=True,source_rule='Historical R2: contribution + road geometry/LiDAR; remaining near-black contributors get opacity gates',
        evidence_frames=199,optimizer_updates=0)
    (dest/'selection_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
