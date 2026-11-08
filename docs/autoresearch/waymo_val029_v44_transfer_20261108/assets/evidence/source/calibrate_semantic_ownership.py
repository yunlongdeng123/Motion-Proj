"""Freeze majority visible-mass identity across BUILD, including originally static rows."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from native_runtime import ROOT,load,camera,native_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--raws',type=int,nargs='*');a=p.parse_args();a.output.mkdir(exist_ok=False)
    import torch
    pipe,models,cfg=load(a.model,a.iteration);static,dynamic,env=models
    from utils.v4_road_render import merge_rows,raster
    identity=json.loads((ROOT/'identity_contract.json').read_text());actors=identity['dynamic_actor_ids'];ns=len(static.get_xyz);nd=len(dynamic.get_xyz);n=ns+nd
    masses=torch.zeros((len(actors),n),device='cuda');total=torch.zeros(n,device='cuda');started=time.monotonic();records=[]
    for frame_name in cfg['followup_scene_spec']['frame_ids']:
        raw=int(frame_name[-3:]);cam=camera(a.model,cfg,raw)
        if a.raws and raw not in a.raws:continue
        s=native_rows(cam,static,pipe);d=native_rows(cam,dynamic,pipe,True);rows=merge_rows(s,d)
        assert len(rows['means3D'])==n
        feature=torch.zeros((n,24),device='cuda');feature[:,:2]=rows['features'][:,:2];feature.requires_grad_(True)
        rows=dict(rows,features=feature);out=raster(cam,pipe,0,rows,torch.zeros(3,device='cuda'));tag=out['extra'][0]
        labels=torch.tensor(np.array(Image.open(Path(cfg['source_path'])/'mask/mask_0/Annotations'/f'{frame_name}.png').resize((480,320),Image.Resampling.NEAREST)),device='cuda')
        present=[actor for actor in actors if bool((labels==actor).any())]
        total+=torch.autograd.grad(tag.sum(),feature,retain_graph=bool(present))[0][:,2].detach()
        for i,actor in enumerate(present):
            value=(tag*(labels==actor)).sum();gradient=torch.autograd.grad(value,feature,retain_graph=i<len(present)-1)[0][:,2].detach()
            masses[actors.index(actor)]+=gradient
        records.append(dict(raw=raw,visible_classes=present));print(raw,flush=True)
        del out,rows,s,d,feature,tag
    best,which=masses.max(0);ratio=best/total.clamp_min(1e-20);ids=torch.tensor(actors,device='cuda')[which];ids=torch.where((total>0)&(ratio>.5),ids,torch.zeros_like(ids))
    cpu=lambda x:x.detach().cpu().numpy()
    np.savez_compressed(a.output/'semantic_ownership.npz',static_actor_ids=cpu(ids[:ns]),dynamic_actor_ids=cpu(ids[ns:]),static_ratio=cpu(ratio[:ns]),dynamic_ratio=cpu(ratio[ns:]),static_mass=cpu(total[:ns]),dynamic_mass=cpu(total[ns:]))
    report=dict(status='PASS_FROZEN_SHARED_VISIBLE_MASS_OWNERSHIP',threshold=.5,strict_greater=True,BUILD_frames=records,static_rows=ns,dynamic_rows=nd,counts={str(actor):dict(static=int((ids[:ns]==actor).sum()),dynamic=int((ids[ns:]==actor).sum())) for actor in actors},model=str(a.model),iteration=a.iteration,gaussian_parameter_updates=0,seconds=time.monotonic()-started)
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['counts']))
if __name__=='__main__':main()
