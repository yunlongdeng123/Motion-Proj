"""V1 原始静态初始化与真实谱系的地面诊断；仅派生显示文件，零优化。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, gc, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from backend import Adapter, clean
from cloud_sampling import Sampler, V1, pack
from export_inspection import js, prepare_depth, native_raster
from ray_reference import candidates, composite


def ply_points(path):
    v=PlyData.read(str(path))['vertex']
    return np.c_[v['x'],v['y'],v['z']].astype(np.float64), np.asarray(v['nx'],np.float32)


def ground_mask(world,inv_anchor,plane,band):
    p=world@inv_anchor[:3,:3].T+inv_anchor[:3,3]
    return (abs(p@plane[:3]+plane[3])<=band)&(p[:,0]>=3)&(p[:,0]<=90)&(abs(p[:,1])<=8)


def build_history(a,b,root):
    import torch
    cfg=json.loads((V1/'effective_config.json').read_text())
    first=V1.parent/'scene0008_089_137_hybrid_30000_seed0_GN11'
    binding=json.loads((first/'actual_sgcm_binding.json').read_text())
    # input.ply is the exact historical full input, before native dynamic-neighbour exclusion.
    world,t=ply_points(V1/'input.ply');old,_=ply_points(first/'input.ply')
    if not np.array_equal(world,old):raise ValueError('V1 original and resumed input clouds differ')
    dynamic=np.concatenate([ply_points(p)[0] for p in sorted(Path(cfg['source_path']).glob('points3d_dynamic_*.ply'))])
    if not cfg['including_dynamic'] or cfg['random_init_point']!=0:raise ValueError('unsupported V1 native initialization policy')
    nearest=cKDTree(dynamic).query(world,k=1,distance_upper_bound=.1,workers=1)[0]
    keep=nearest>=.1;world=world[keep];t=t[keep];input_rows=np.flatnonzero(keep).astype(np.int32)
    if len(world)!=binding['static_points']:raise ValueError('native V1 initialization count mismatch')
    if not np.allclose(binding['metadata']['world_to_model'],a.A,atol=1e-10):raise ValueError('V1/V3 coordinate systems are not aligned')
    payload,step=torch.load(str(V1/'chkpnt30000.pth'),map_location='cpu',weights_only=False,mmap=True)
    lineage=payload[14]['sgcm_lineage_v1']['lineage_ids'].numpy()
    if step!=30000 or lineage.min()<0 or lineage.max()>=len(world) or len(lineage)!=len(b['xyz']):raise ValueError('V1 saved lineage does not index original static initialization')
    ids=lineage.astype(np.int32).copy();del payload;gc.collect()
    anchor=np.array(a.meta['coordinate_anchor']);inv=np.linalg.inv(anchor)
    ready=json.loads(Path(a.meta['original_readiness']).read_text());plane=np.array(ready['SGCM_geometry']['ground_fit']['plane_anchor'])
    init_ground=ground_mask(world,inv,plane,.15)
    # 原生 fused_point_cloud 转 float32 后使用三近邻均方距离初始化各向同性尺度。
    initial=(world@a.A[:3,:3].T+a.A[:3,3]).astype(np.float32)
    dist=cKDTree(initial).query(initial,k=4,workers=1)[0][:,1:]
    sigma=np.sqrt(np.maximum((dist**2).mean(1),1e-7)).astype(np.float32)
    descendants=np.bincount(ids,minlength=len(initial)).astype(np.int32)
    chosen=init_ground[ids].copy()
    # 固定候选集合：地面初始来源的全部后代，或49帧任一时刻中心落在实测平面±0.30m。
    omega=2*np.pi/b['cycle']
    for raw in range(89,138):
        delta=(raw-113)*.02-b['t'];xyz=b['xyz']+b['vel']*(np.sin(delta*omega)/omega)[:,None]
        w=xyz@a.invA[:3,:3].T+a.invA[:3,3]
        chosen|=ground_mask(w,inv,plane,.30)
    rows=np.flatnonzero(chosen);roots=ids[rows];ground_rows=np.flatnonzero(init_ground)
    local=world@inv[:3,:3].T+inv[:3,3];tree=cKDTree(local[init_ground,:2]);support=tree.query_ball_point(local[:,:2],.8,return_length=True).astype(np.int32)
    arrays=dict(init_xyz=initial,init_xy=local[:,:2].astype(np.float32),init_t=t,init_sigma=sigma,init_ground=init_ground.astype(np.uint8),input_row=input_rows,
        descendants=descendants,support_count=support,ground_xyz=initial[init_ground],ground_xy=local[init_ground,:2].astype(np.float32),
        row=rows.astype(np.int32),root=roots,root_ground=init_ground[roots].astype(np.uint8),
        xyz=b['xyz'][rows],sigma=b['scale'][rows],q=b['q'][rows],velocity=b['vel'][rows],t=b['t'][rows],sigma_t=b['sigma_t'][rows],
        opacity=b['opacity'][rows],color=np.maximum(.5+.28209479177387814*b['dc'][rows,0],0))
    metadata=dict(version=1,arm='gn11_v1',frames=49,raw_start=89,raw_end=137,checkpoint_sha256=b['sha'],cycle=b['cycle'],
        initial_static_count=len(initial),initial_ground_count=len(ground_rows),trained_candidate_count=len(rows),trained_static_count=len(b['xyz']),
        initial_ground_with_descendants=int(np.count_nonzero(descendants[init_ground])),initial_opacity=.01,initial_sigma_t=float(np.sqrt((cfg['time_duration'][1]-cfg['time_duration'][0])*cfg['t_init'])),
        support_radius_m=.8,initial_ground_band_m=.15,current_ground_band_m=.30,plane_anchor=plane.tolist(),coordinate_anchor=anchor.tolist(),
        input_ply=str(V1/'input.ply'),dynamic_exclusion_m=.1,
        initialization='native original input.ply + original all-five dynamic nearest<0.1m exclusion; count matches actual initial SGCM binding; sigma CPU reconstructed native 3NN formula, not saved CUDA step0',
        lineage='saved optimizer.sgcm_lineage_v1, inherited by clone/split; root-to-descendant offset is not one-to-one Gaussian displacement',
        candidate_definition='all descendants of initial static centers in ground plane band0.15m, UNION static centers in plane band0.30m at any of49 timestamps; local x3..90m,y±8m; fixed diagnostic subset, not a native road branch or semantic GT',
        ground_source='V1 actual static initialization input near shared measured ground plane, before Gaussian creation; not the V3 10cm-dedup pool',
        time_values=sorted(float(v) for v in np.unique(t)),gpu_used=False,optimizer_updates=0)
    js(root,'data/v1_ground_history.js','v1GroundHistory',dict(**pack(arrays),metadata=metadata))
    np.save(root/'v1_candidate_mask.npy',chosen)
    (root/'v1_ground_build.json').write_text(json.dumps(clean(metadata),indent=2))
    return chosen,metadata


def export(root,raws=range(89,138),resume=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);started=time.time();a=Adapter();s=Sampler(a);b=s.get_model('gn11_v1')['branches'][0]
    if resume and (root/'v1_ground_build.json').exists() and (root/'data/v1_ground_history.js').exists():
        metadata=json.loads((root/'v1_ground_build.json').read_text());chosen=np.load(root/'v1_candidate_mask.npy')
        if metadata['checkpoint_sha256']!=b['sha'] or len(chosen)!=len(b['xyz']):raise ValueError('cached ground group checkpoint mismatch')
    else:chosen,metadata=build_history(a,b,root)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)
    receipt=dict(status='V1_GROUND_EXPORT_RUNNING',metadata=metadata,checks=[],gpu_used=False,optimizer_updates=0)
    for raw in raws:
        path=f'data/inspection/gn11_v1_{raw}.js'
        if resume and (root/path).exists():continue
        t0=time.time();p=prepare_depth(a,b,raw,root,include_rows=True)
        which=chosen[p['source_row']];ground={k:v[which] for k,v in p.items()};op={k:v[~which] for k,v in p.items()}
        ga,gd=native_raster(root,ground);oa,od=native_raster(root,op);errs=[]
        for x,y in [(90,255),(255,240),(130,231),(267,297),(365,280),(240,100),(30,30)]:
            for pp,aa,dd in [(ground,ga,gd),(op,oa,od)]:
                ix,al=candidates(pp,x,y);c=composite(al,np.zeros((len(ix),3),np.float32),pp['depth'][ix]);errs.extend([abs(c['alpha']-aa[y,x]),abs(c['depth_sum']-dd[y,x])])
        if max(errs)>.002:raise ValueError(('independent pixel reference failed',raw,max(errs)))
        arrays=dict(ground_alpha=ga,ground_depth_m=np.where(ga>1e-8,gd/np.maximum(ga,1e-8)/a.scale,0).astype(np.float32),
            ordinary_alpha=oa,ordinary_depth_m=np.where(oa>1e-8,od/np.maximum(oa,1e-8)/a.scale,0).astype(np.float32))
        js(root,path,f'inspection/gn11_v1/{raw}',pack(arrays));receipt['checks'].append(dict(raw=raw,python_reference_max_error=max(errs),seconds=round(time.time()-t0,2)))
        print('V1 ground',raw,'seconds',round(time.time()-t0,2),flush=True)
        (root/'v1_ground_receipt.json').write_text(json.dumps(clean(receipt),indent=2))
        del p,ground,op;gc.collect()
    receipt.update(status='V1_GROUND_EXPORT_COMPLETE',seconds=round(time.time()-started,2));(root/'v1_ground_receipt.json').write_text(json.dumps(clean(receipt),indent=2));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--raws',nargs='+',type=int,default=list(range(89,138)));p.add_argument('--resume',action='store_true');q=p.parse_args();export(q.output,q.raws,q.resume)
