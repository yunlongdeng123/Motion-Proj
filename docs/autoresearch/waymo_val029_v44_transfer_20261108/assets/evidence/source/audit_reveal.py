"""Locate the reveal-depth limitation without changing geometry or choosing a new fit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from editable_scene import ROOT,BANK,Scene
def main():
    import torch
    from utils.v4_road_render import raster
    scene=Scene(True,False);cam=scene.camera(155);obs=scene.observation(155);body=next(b for b in obs['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');pose=torch.tensor(scene.T@np.array(body['world_from_actor']).reshape(4,4),device='cuda',dtype=torch.float32);points=np.array([v['actor_local_xyz'] for v in json.loads((BANK.parent/'verified_correspondences.json').read_text())['pairs']]);complete=np.load(ROOT/'revealed_material_r2/complete_asset.npz');distance=cKDTree(complete['xyz']).query(points)[0]
    world=points@np.array(body['world_from_actor']).reshape(4,4)[:3,:3].T+np.array(body['world_from_actor']).reshape(4,4)[:3,3];cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (BANK/'calib/0029155.txt').read_text().splitlines()};C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);C=C@np.linalg.inv(np.array(obs['ego_world']).reshape(4,4));cp=world@C[:3,:3].T+C[:3,3];K=cal['P0'].reshape(3,4)[:,:3]/4;K[2,2]=1;uv=(cp@K.T)[:,:2]/cp[:,2:];pixel=np.floor(uv).astype(int);scale=np.linalg.det(scene.T[:3,:3])**(1/3)
    with torch.no_grad():
        rows=scene.asset.rows(pose,cam,scene.pipe,1,precomputed=True);out=raster(cam,scene.pipe,0,rows,torch.zeros(3,device='cuda'));alpha=out['alpha'][0].cpu().numpy()[pixel[:,1],pixel[:,0]];depth=out['depth_sum'][0].cpu().numpy()[pixel[:,1],pixel[:,0]]/np.maximum(alpha,1e-5)/scale;rgb=out['rgb'].cpu().numpy().transpose(1,2,0);Image.fromarray(np.uint8(rgb.clip(0,1)*255+.5)).save(ROOT/'analysis/155_white_asset_only.png')
    packs={k:np.load(ROOT/v/'155/delete.npz') for k,v in [('semantic','semantic_all_eval'),('asset','semantic_asset_all_r2'),('final','final_all_eval')]};groups={k:pack['group_alpha'][:,pixel[:,1],pixel[:,0]].mean(1).tolist() for k,pack in packs.items()};report=dict(status='READ_ONLY_FAILURE_LOCALIZATION',anchors=len(points),asset_nearest_surface_distance_m=dict(mean=float(distance.mean()),median=float(np.median(distance)),max=float(distance.max())),asset_only_alpha_mean=float(alpha.mean()),asset_only_depth_mae_m=float(abs(depth-cp[:,2]).mean()),asset_only_supported_alpha_over_half=int((alpha>.5).sum()),joint_group_mean_alpha_red_white_other_road=groups,claim='Sparse BUILD surface and mixed-render diagnostics only. No tuning or acceptance follows from this audit.',updates=0);(ROOT/'analysis/reveal_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
