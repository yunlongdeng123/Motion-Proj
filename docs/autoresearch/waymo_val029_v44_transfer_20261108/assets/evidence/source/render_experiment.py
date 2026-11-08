"""Paired edits and conservation checks on fixed assets, no image-space routing."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from editable_scene import Scene,ROOT,BANK
def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['semantic','asset','final'],required=True);p.add_argument('--output',required=True);p.add_argument('--raws',type=int,nargs='*');p.add_argument('--heldout',action='store_true');p.add_argument('--gate-dir',default='road_gates');a=p.parse_args();out=ROOT/a.output;out.mkdir(exist_ok=False);scene=Scene(with_asset=a.mode!='semantic',with_road=a.mode=='final');started=time.monotonic();records=[]
    if a.mode=='final':
        ids=np.load(ROOT/a.gate_dir/'contributors.npz')['ids'];state=torch.load(str(ROOT/a.gate_dir/'checkpoint001000.pt'),map_location='cuda',weights_only=False);scene.gate_ids=torch.tensor(ids,device='cuda');scene.gates=state['gate'].sigmoid()
    raws=list(range(79,119)) if a.heldout else (a.raws or scene.raws)
    if a.heldout:
        from heldout_camera import held_camera
    with torch.no_grad():
        for raw in raws:
            cam=held_camera(scene,raw) if a.heldout else scene.camera(raw);values=scene.rows(raw,cam);folder=out/f'{raw:03d}';folder.mkdir();outputs={};rec=dict(raw=raw,regions={})
            source=ROOT/'heldout/rgb'/f'{raw:03d}.png' if a.heldout else BANK/'image/image_0'/f'0029{raw:03d}.png';observed=np.array(Image.open(source).resize((480,320),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/255;Image.fromarray(np.uint8(observed.transpose(1,2,0)*255+.5)).save(folder/'observed.png')
            for name,delete in [('keep',False),('delete',True),('keep_again',False)]:
                pack=scene.render(raw,delete,rows=values);rgb=pack['rgb'].cpu().numpy();conservation=float((pack['extra'][:4].sum(0)-pack['alpha'][0]).abs().max());assert conservation<3e-6,conservation
                if delete:assert float(pack['extra'][0].abs().max())<1e-6
                if name=='keep_again':assert np.array_equal(rgb,outputs['keep']);continue
                outputs[name]=rgb;Image.fromarray(np.uint8(rgb.transpose(1,2,0).clip(0,1)*255+.5)).save(folder/f'{name}.png');np.savez_compressed(folder/f'{name}.npz',rgb=rgb,alpha=pack['alpha'].cpu().numpy(),depth_sum=pack['depth_sum'].cpu().numpy(),group_alpha=pack['extra'][:4].cpu().numpy());rec[name+'_group_conservation_maxabs']=conservation
            rec.update(keep_gt_mae=float(abs(outputs['keep']-observed).mean()),keep_delete_changed_pixels=int((abs(outputs['keep']-outputs['delete']).max(0)>1/255).sum()),target_delete_contribution_max=0.,keep_again_exact=True)
            if not a.heldout:
                labels=np.array(Image.open(BANK.parent/'masks_all_resolved_r2'/f'{raw:03d}/labels.png'));base=np.load(ROOT/'formal10k_eval'/f'{raw:03d}/keep.npz')['rgb']
                for label,role in [(70,'target_red'),(25,'revealed_white_suv'),(20,'preserved_gray'),(-1,'all_other_vehicles')]:
                    mask=((labels>0)&(labels!=70)&(labels!=25)) if label==-1 else labels==label;n=int(mask.sum());rec['regions'][role]=dict(pixels=n,native_keep_gt_mae=float(abs(base-observed)[:,mask].mean()) if n else None,keep_gt_mae=float(abs(outputs['keep']-observed)[:,mask].mean()) if n else None,delete_vs_keep_mae=float(abs(outputs['delete']-outputs['keep'])[:,mask].mean()) if n else None)
                if a.mode=='final':
                    d=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');pred=outputs['delete'];lp=np.where(pred<=.04045,pred/12.92,((np.clip(pred,0,1)+.055)/1.055)**2.4)
                    for key in ['real_domain','inferred_domain']:
                        m=d[key];rec['regions'][key]=dict(pixels=int(m.sum()),linear_proxy_mae=float(abs(lp-d['reference_linear'])[:,m].mean()) if m.any() else None)
            records.append(rec)
            if raw%10==0 or raw in [130,155,160,190]:print(json.dumps(rec),flush=True)
    assert scene.unchanged();report=dict(status='PASS_PAIRED_EDIT_MECHANICAL_CHECKS',mode=a.mode,frames=records,heldout=a.heldout,optimizer_updates=0,native_parent_unchanged=True,quality_accepted=None,seconds=time.monotonic()-started);(out/'summary.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
