"""只读同场景KEEP/DELETE，静态与其他动态独立归因。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from editable_scene import Scene,ROOT,BANK
def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['semantic','final'],required=True);p.add_argument('--output',required=True);p.add_argument('--raws',type=int,nargs='*');p.add_argument('--heldout',action='store_true');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(exist_ok=False);scene=Scene(with_road=a.mode=='final');started=time.monotonic();records=[]
    if a.mode=='final' and (ROOT/'road_gates/checkpoint001000.pt').exists():
        ids=np.load(ROOT/'road_gates/contributors.npz')['ids'];state=torch.load(ROOT/'road_gates/checkpoint001000.pt',map_location='cuda',weights_only=False)
        scene.gate_ids=torch.tensor(ids,device='cuda');scene.gates=state['gate'].sigmoid()
    raws=list(range(79,119)) if a.heldout else (a.raws or scene.raws)
    if a.heldout:
        from heldout_camera import held_camera
    with torch.no_grad():
        for raw in raws:
            cam=held_camera(scene,raw) if a.heldout else scene.camera(raw);values=scene.rows(raw,cam);folder=out/f'{raw:03d}';folder.mkdir();outputs={};rec=dict(raw=raw,regions={})
            source=ROOT/'heldout/rgb'/f'{raw:03d}.png' if a.heldout else BANK/'image/image_0'/f'0065{raw:03d}.png'
            observed=np.array(Image.open(source).resize((480,320),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/255
            Image.fromarray(np.uint8(observed.transpose(1,2,0)*255+.5)).save(folder/'observed.png')
            for name,delete in [('keep',False),('delete',True),('keep_again',False)]:
                pack=scene.render(raw,delete,rows=values);rgb=pack['rgb'].cpu().numpy();groups=pack['extra'][:4]
                conservation=float((groups.sum(0)-pack['alpha'][0]).abs().max());assert conservation<3e-6,conservation
                assert np.isfinite(rgb).all()
                if delete:assert float(groups[0].abs().max())<1e-6
                if name=='keep_again':assert np.array_equal(rgb,outputs['keep']);continue
                outputs[name]=rgb;Image.fromarray(np.uint8(rgb.transpose(1,2,0).clip(0,1)*255+.5)).save(folder/f'{name}.png')
                np.savez_compressed(folder/f'{name}.npz',rgb=rgb,alpha=pack['alpha'].cpu().numpy(),depth_sum=pack['depth_sum'].cpu().numpy(),group_alpha=groups.cpu().numpy())
                rec[name+'_group_conservation_maxabs']=conservation
            rec.update(keep_gt_mae=float(abs(outputs['keep']-observed).mean()),keep_delete_changed_pixels=int((abs(outputs['keep']-outputs['delete']).max(0)>1/255).sum()),target_delete_contribution_max=0.,keep_again_exact=True)
            if not a.heldout:
                labels=np.array(Image.open(BANK.parent/'masks_all_resolved_r2'/f'{raw:03d}/labels.png'));base=np.load(ROOT/'formal10k_eval'/f'{raw:03d}/keep.npz')['rgb']
                rec['native_keep_parity_maxabs']=float(abs(base-outputs['keep']).max())
                if a.mode=='semantic':assert rec['native_keep_parity_maxabs']<2e-5,rec
                for label,role in [(10,'target_white'),(21,'front_pickup'),(8,'preserved_pickup'),(-1,'all_other_vehicles')]:
                    mask=((labels>0)&(labels!=10)) if label==-1 else labels==label;n=int(mask.sum())
                    rec['regions'][role]=dict(pixels=n,native_keep_gt_mae=float(abs(base-observed)[:,mask].mean()) if n else None,keep_gt_mae=float(abs(outputs['keep']-observed)[:,mask].mean()) if n else None,delete_vs_keep_mae=float(abs(outputs['delete']-outputs['keep'])[:,mask].mean()) if n else None)
                if a.mode=='final':
                    d=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');pred=outputs['delete'];lp=np.where(pred<=.04045,pred/12.92,((np.clip(pred,0,1)+.055)/1.055)**2.4)
                    for key in ['real_domain','inferred_domain']:
                        m=d[key];rec['regions'][key]=dict(pixels=int(m.sum()),linear_proxy_mae=float(abs(lp-d['reference_linear'])[:,m].mean()) if m.any() else None)
            records.append(rec)
            if raw%20==0:print(json.dumps(rec),flush=True)
    assert scene.unchanged()
    (out/'summary.json').write_text(json.dumps(dict(status='PASS_PAIRED_EDIT_MECHANICAL_CHECKS',mode=a.mode,frames=records,heldout=a.heldout,group_order=['target_all','other_static','other_dynamic','road'],vehicle_asset_inserted=False,optimizer_updates=0,native_parent_unchanged=True,quality_accepted=None,seconds=time.monotonic()-started),indent=2))
if __name__=='__main__':main()
