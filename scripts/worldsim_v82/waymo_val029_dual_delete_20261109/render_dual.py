"""同一冻结场景的两个查询：删除红车或白色SUV；不新增训练。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from PIL import Image
LEGACY=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1')
sys.path.insert(0,str(LEGACY/'source'))
from editable_scene import Scene
from heldout_camera import held_camera
KEYS=[130,155,160,190]
def main():
    import torch
    from utils.v4_road_render import raster
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--raws',type=int,nargs='+',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);scene=Scene(with_asset=True,with_road=True)
    idx=np.load(LEGACY/'road_gates/contributors.npz')['ids'];state=torch.load(LEGACY/'road_gates/checkpoint001000.pt',map_location='cuda',weights_only=False)
    scene.gate_ids=torch.tensor(idx,device='cuda');scene.gates=state['gate'].sigmoid()
    scale=float(np.linalg.det(scene.T[:3,:3])**(1/3));start=time.monotonic();records=[]
    with torch.no_grad():
        for raw in a.raws:
            assert 109<=raw<=197
            cam=held_camera(scene,raw) if raw<119 else scene.camera(raw)
            values,ids,cam=scene.rows(raw,cam);folder=a.output/f'{raw:03d}';folder.mkdir();images={};rec=dict(raw=raw,targets={},keep_reference_maxabs=None)
            for name,target in [('keep',None),('delete_red',70),('delete_white',25),('keep_again',None)]:
                mask=values['mask'] if target is None else values['mask']&(ids!=target)
                if target is not None:
                    other=25 if target==70 else 70
                    assert torch.equal(mask[ids==other],values['mask'][ids==other])
                pack=raster(cam,scene.pipe,0,dict(values,mask=mask),torch.zeros(3,device='cuda'))
                rgb=pack['rgb']+(1-pack['alpha'])*scene.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                rgb=rgb.cpu().numpy();groups=pack['extra'][:4];assert np.isfinite(rgb).all()
                conservation=float(abs(groups.sum(0)-pack['alpha'][0]).max());assert conservation<3e-6
                if name=='keep_again':assert np.array_equal(rgb,images['keep']);continue
                images[name]=rgb
                Image.fromarray(np.uint8(rgb.transpose(1,2,0).clip(0,1)*255+.5)).save(folder/f'{name}.png')
                if raw in KEYS:
                    np.savez_compressed(folder/f'{name}.npz',rgb=rgb,group_alpha=groups.cpu().numpy(),alpha=pack['alpha'].cpu().numpy(),depth_sum=pack['depth_sum'].cpu().numpy())
                if target is not None:
                    group=0 if target==70 else 1;removed=float(groups[group].abs().max());assert removed<1e-6
                    rec['targets'][name]=dict(native_actor_id=target,removed_group_max=removed,other_actor_rows_preserved=True,changed_pixels=int((abs(rgb-images['keep']).max(0)>1/255).sum()),group_conservation_maxabs=conservation)
            ref=LEGACY/('final_heldout_eval' if raw<119 else 'final_all_eval')/f'{raw:03d}'
            rec['keep_reference_maxabs']=float(abs(images['keep']-np.load(ref/'keep.npz')['rgb']).max())
            red_error=float(abs(images['delete_red']-np.load(ref/'delete.npz')['rgb']).max())
            assert rec['keep_reference_maxabs']<2e-5 and red_error<2e-5
            rec['red_delete_reference_maxabs']=red_error;records.append(rec);print(json.dumps(rec),flush=True)
    assert scene.unchanged()
    receipt=dict(status='PASS_FROZEN_SCENE_TWO_DISTINCT_DELETION_QUERIES',raws=a.raws,rows=records,
        model=str(LEGACY/'formal10k'),iteration=10000,updates=0,meter_to_model=scale,
        target_red=dict(actor_id=70,alias='a69',track='lvxXvuSB3kueTyP2ignh7w'),
        target_white=dict(actor_id=25,alias='a24',track='DwjZr_bYUSZOr-XDGregnA'),
        group_order=['red','white_suv','other','road'],same_scene_for_all_queries=True,
        existing_road_and_gates_frozen=True,new_background_fit=False,
        hidden_background_truth_available=False,human_verdict=None,automatic_shutdown=False,
        seconds=time.monotonic()-start)
    (a.output/'summary.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
