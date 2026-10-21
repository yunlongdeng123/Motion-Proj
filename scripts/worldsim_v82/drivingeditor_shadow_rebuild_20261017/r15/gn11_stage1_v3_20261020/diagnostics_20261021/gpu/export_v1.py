"""V1冻结30K原生全帧重放 + 相同射线归因；原模型与PCA不写。"""
import contextlib
import gc
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def ranked_sample(rows, percent):
    # 与CPU查看器使用相同uint32嵌套顺序；本导出器保持独立部署。
    x=np.asarray(rows,dtype=np.uint32).copy()
    x^=x>>np.uint32(16);x*=np.uint32(0x7feb352d);x^=x>>np.uint32(15)
    x*=np.uint32(0x846ca68b);x^=x>>np.uint32(16)
    return rows[np.lexsort((rows,x))[:int(np.ceil(len(rows)*percent/100))]]

from gpu_backend import GPUAdapter,V1_RUN,CAMPAIGN,HERE,sha,clean,global_slot

def main():
    import torch
    from unittest.mock import patch
    from export_cpu import packed
    out=HERE/'v1_export_r3';out.mkdir(exist_ok=False)
    def js(path,key,v):
        path=out/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(clean(v),separators=(',',':'),allow_nan=False)+';')
    binding=dict(run=str(V1_RUN),contract='historical native GN11 20K->30K recovery; NOT V3 fresh method control',checkpoints={})
    for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')]:
        p=V1_RUN/f'{prefix}30000.pth';binding['checkpoints'][name]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    (HERE/'v1_binding.json').write_text(json.dumps(binding,indent=2))
    with global_slot(),patch.object(torch.optim.Adam,'step',side_effect=RuntimeError('no optimization')),torch.no_grad():
        adapter=GPUAdapter();adapter.model('gn11_v1');records=[];queries=[];cloud={};branches=[]
        for name,model in zip(('static','dynamic'),adapter.models[:2]):
            n=len(model._xyz);ids=np.sort(ranked_sample(np.arange(n),2));ix=torch.tensor(ids,device='cuda');prefix=name+'_'
            attrs=dict(row=ids.astype(np.int32),xyz=model._xyz[ix],sigma=torch.exp(model._scaling[ix]),q=model._rotation[ix],velocity=model._velocity[ix],
                t=model._t[ix,0],sigma_t=torch.exp(model._scaling_t[ix,0]),opacity=torch.sigmoid(model._opacity[ix,0]),
                color=(.5+.28209479177387814*model._features_dc[ix,0]).clamp_min(0),role=np.full(len(ids),0 if name=='static' else 2,np.uint8),
                road_id=np.full(len(ids),-1,np.int32),cycle=np.array([float(adapter.config.cycle)],np.float32))
            for k,v in attrs.items():cloud[prefix+k]=v.cpu().numpy() if torch.is_tensor(v) else v
            branches.append(dict(branch=name,total=n,displayed=len(ids),checkpoint_sha256=binding['checkpoints'][name]['sha256']))
        js('data/gn11_v1/cloud.js','gn11_v1/cloud',dict(**packed(cloud),branches=branches,sampling=dict(static_rate=2,dynamic_rate=2,vehicle='none',vehicle_rate=10,vehicle_scope='projection',raw=102,roads_all_retained=True,gpu_used=True,optimizer_updates=0,stats={b['branch']:dict(total=b['total'],ordinary_total=b['total'],road_total=0,base_sampled=b['displayed'],sampled=b['displayed'],vehicle_candidates=0,vehicle_sampled=0) for b in branches})))
        for raw in range(89,138):
            packs={};m={};oldroot=V1_RUN/'eval/train_30000_render'
            for mode,channel,old in [('deleted_five','deleted_five','static'),('joint','joint','composite'),('dynamic','dynamic_raw','dynamic')]:
                p,k,_,_=adapter.branch('gn11_v1',raw,mode);native=adapter.native_render(k,p['settings']);alpha=native[3][-1:];depth=native[3][-2:-1]/alpha.clamp_min(1e-8)/adapter.scale
                rgb=native[2]+(1-alpha)*p['env'] if mode!='dynamic' else native[2]
                packs[channel]=rgb.cpu().numpy();prefix={'deleted_five':'static','joint':'joint','dynamic_raw':'dynamic'}[channel]
                m[prefix+'_alpha']=np.round(alpha[0].cpu().numpy().clip(0,1)*65535).astype(np.uint16)
                d=depth[0].cpu().numpy();m[prefix+'_depth_m']=np.where((alpha[0].cpu().numpy()>1e-8)&np.isfinite(d)&(d>0),np.round(np.clip(d*100,0,65534)),65535).astype(np.uint16)
                arr=np.round(np.clip(packs[channel].transpose(1,2,0),0,1)*255).astype(np.uint8)
                path=out/f'assets/gn11_v1/{channel}/{raw:03d}.webp';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(path,lossless=True,method=4)
                # V1 eval is_training=False: dynamic branch is raw (no env).
                # Static/joint have env; compare each original PNG's actual semantics.
                comparison=arr.astype(np.int16)
                oldpath=oldroot/old/f'image_0_0008{raw:03d}.png';oldim=np.asarray(Image.open(oldpath).convert('RGB')).astype(np.int16)
                err=int(np.abs(comparison-oldim).max())
                if err>1:raise RuntimeError(f'V1 native PNG replay {raw}/{mode}: {err}')
                records.append(dict(raw=raw,mode=mode,max_8bit_error=err,source=str(oldpath),source_sha256=sha(oldpath)))
            with np.load(CAMPAIGN/f'final49/full_r2/0008{raw:03d}/render_float.npz') as z:
                gt=z['gt'];d=z['lidar_depth_m'][0];m['lidar_depth_m']=np.where(d>0,np.round(np.clip(d*100,0,65534)),65535).astype(np.uint16);m['body_mask']=z['body_mask'][0].astype(np.uint8)
            m['residual']=np.round(np.abs(packs['joint']-gt).mean(0).clip(0,1)*65535).astype(np.uint16)
            js(f'data/gn11_v1/{raw}.js',f'gn11_v1/{raw}',packed(m))
            if raw in (89,102,131,137):
                for x,y in ((90,255),(255,240),(365,280)):
                    for mode in ('deleted_five','joint'):
                        v=adapter.trace('gn11_v1',raw,x,y,mode);key=f'gpu/gn11_v1/{raw}/{x}/{y}/{mode}';path=f'data/gpu/gn11_v1_{raw}_{x}_{y}_{mode}.js';js(path,key,v);queries.append(dict(key=key,path=path))
            print('V1 '+str(raw),file=sys.stderr,flush=True)
        adapter.check_protected()
        receipt=dict(status='V1_NATIVE_30K_DIAGNOSTIC_REPLAY_COMPLETE',binding=binding,whole_frame_replays=records,queries=queries,
            native_camera_world_alignment=True,optimizer_updates=0,road_rows_injected=0,quality_accepted=False,human_verdict=None,
            adapter='V3 NativeGaussianModel-compatible no-road forward; every49 frame three original PNG branches independently replayed',
            comparison_limit='historical recovery and method/config differ; frame-aligned diagnostic, not isolated causal V3-vsV1 treatment')
        (out/'v1_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],replays=len(records),queries=len(queries),output=str(out))))

if __name__=='__main__':main()
