"""四帧只读来源诊断；RGB逐值对齐独立原生评价，不使用共享语义归属。"""
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from native_runtime import ROOT,load,camera,native_rows
def main():
    import torch
    from drivingeditor_dynamic_edit import actor_ids
    from utils.v4_road_render import merge_rows,raster
    out=ROOT/'native_source_attribution';out.mkdir(exist_ok=False);pipe,models,cfg=load(ROOT/'formal10k',10000);static,dynamic,env=models;owners=actor_ids(dynamic);records=[];start=time.monotonic()
    versions={(b,k):v._version for b,m in enumerate(models[:2]) for k,v in vars(m).items() if torch.is_tensor(v)}
    with torch.no_grad():
        for raw in [30,60,150,198]:
            cam=camera(ROOT/'formal10k',cfg,raw);s=native_rows(cam,static,pipe);d=native_rows(cam,dynamic,pipe,True)
            for rows,is_static in [(s,True),(d,False)]:
                f=rows['features'].new_zeros((len(rows['means3D']),24));f[:,:2]=rows['features'][:,:2]
                if is_static:f[:,3]=1
                else:f[:,2]=(owners==10);f[:,4]=(owners!=10)
                rows['features']=f
            rows=merge_rows(s,d);mask=torch.cat([torch.ones(len(static.get_xyz),device='cuda',dtype=torch.bool),owners!=10]);folder=out/f'{raw:03d}';folder.mkdir();labels=np.array(Image.open(Path(cfg['source_path'])/'mask/mask_0/Annotations'/f'0065{raw:03d}.png').resize((480,320),Image.Resampling.NEAREST));region=labels==10
            for state in ['keep','delete']:
                values=dict(rows,mask=rows['mask']&(mask if state=='delete' else torch.ones_like(mask)))
                p=raster(cam,pipe,0,values,torch.zeros(3,device='cuda'));rgb=p['rgb']+(1-p['alpha'])*env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                a=rgb.cpu().numpy();reference=np.load(ROOT/'formal10k_eval'/f'{raw:03d}/{state}.npz')['rgb'];error=float(abs(a-reference).max());assert error<2e-5,(raw,state,error)
                group=p['extra'][:4];conservation=float(abs(group.sum(0)-p['alpha'][0]).max());assert conservation<3e-6
                np.savez_compressed(folder/f'{state}.npz',rgb=a,alpha=p['alpha'].cpu().numpy(),depth_sum=p['depth_sum'].cpu().numpy(),group_alpha=group.cpu().numpy())
                records.append(dict(raw=raw,state=state,original_RGB_maxabs=error,group_conservation_maxabs=conservation,target_mask_pixels=int(region.sum()),mean_groups_on_target_mask=group[:,region].mean(1).cpu().tolist()))
    assert all(vars(models[b])[k]._version==v for (b,k),v in versions.items())
    (out/'summary.json').write_text(json.dumps(dict(status='PASS_NATIVE_SOURCE_BRANCH_DIAGNOSTIC_PARITY',group_order=['target_source_dynamic','all_static','other_source_dynamic','road_zero'],semantic_rebinding=False,updates=0,optimizer_created=False,frames=records,seconds=time.monotonic()-start),indent=2))
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
