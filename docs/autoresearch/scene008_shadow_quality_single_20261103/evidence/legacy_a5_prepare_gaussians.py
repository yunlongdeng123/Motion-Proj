import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from shadow_math import project_gaussians,replay,rotation_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');torch.set_num_threads(2)
    spec=json.loads((root/'rigid_specs/scene0008_089_137.json').read_text())
    cp_path=root/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'
    cp=torch.load(cp_path,map_location='cpu',weights_only=False);cap=cp['background']
    with np.load(Path(a.data)/'training_data.npz') as f:
        plane=f['plane_model'];scale=float(f['world_scale']);q=f['xyz_model'];pixels=f['pixels'];cloud=f['local_lidar'];time=float(f['times'][12])
    cam=json.loads((Path(a.data)/'camera_metadata.json').read_text())[12]['camera']
    ref=json.loads(Path(spec['reference_config']).read_text())
    sys.path.insert(0,str(root/'infra'));from renderer import evaluate_sh_rgb
    omega=2*math.pi/float(cap[16]);xyz=cap[1]+cap[9]*torch.sin((time-cap[7])*omega)/omega
    marginal=torch.exp(-.5*((cap[7]-time)/cap[8].exp())**2).reshape(-1)
    rgb=evaluate_sh_rgb(torch.cat((cap[2],cap[3]),1),torch.nn.functional.normalize(xyz-torch.tensor(cam['camera_center']),dim=-1),int(cap[0]))
    arr=lambda t:t.detach().numpy().astype('float32',copy=False)
    state=dict(xyz=arr(xyz),colors=arr(rgb),scales=arr(cap[4].exp()),rotation=arr(cap[5]),opacity=arr(cap[6].sigmoid()).ravel()*arr(marginal),mask=arr(marginal)>.05)
    projected=project_gaussians(state,cam,ref['neg_fov'])
    footprint=np.array(Image.open(root/'black_shadow_audit_20261016_v1/focus_cpu/reviewed_artifact_footprint.png'))>0
    fy,fx=np.where(footprint);fpix=np.c_[fx,fy]
    replay_rgb,_,mass,_=replay(projected,fpix)
    eroot=root/'rigid_post30k/scene0008_089_137/evaluation'
    mf=json.loads((eroot/'scene_evaluation_manifest.json').read_text());row=mf['streams']['observed']['frames'][12]
    with np.load(eroot/row['npz']) as f:expected=f['static_raw_rgb'][:,fy,fx].T
    assert abs(replay_rgb-expected).mean()<1e-4
    _,_,coarse_mass,_=replay(projected,pixels.reshape(60,140,2)[::4,::4].reshape(-1,2))
    ids=projected['ids'];height=(projected['xyz']@plane[:3]+plane[3])/scale
    rot=rotation_matrix(projected['rotation']);sig=np.linalg.norm(np.einsum('nji,j->ni',rot,plane[:3])*projected['scales'],axis=1)/scale
    distance,_=cKDTree(cloud/scale).query(projected['xyz']/scale,k=1,workers=2)
    uv=projected['uv'];region=(uv[:,0]>35)&(uv[:,0]<185)&(uv[:,1]>220)&(uv[:,1]<290)
    visible=(mass>1e-9)|(coarse_mass>1e-9)
    road=visible&region&(abs(height)<=.20)&(3*sig<=.30)&(distance<.45)
    bad=visible&~road&(projected['colors'].max(1)<.05)
    selection={'schema':'local_road_Gaussian_selection_v1','receiver_points':int(road.sum()),'artifact_opacity_only_points':int(bad.sum()),
      'original_footprint_alpha_T_receiver_fraction':float(mass[road].sum()/mass.sum()),
      'original_footprint_alpha_T_artifact_fraction':float(mass[bad].sum()/mass.sum()),
      'CPU_baseline_replay_MAE':float(abs(replay_rgb-expected).mean()),
      'receiver_criteria':'real contribution + local view region + abs plane height<=0.20m + normal3sigma<=0.30m + local LiDAR nearest<=0.45m',
      'artifact_criteria':'remaining real contributing near-black SH points; opacity-only optimization, no blanket vehicle-box pruning',
      'parent_checkpoint':{'path':str(cp_path),'bytes':cp_path.stat().st_size,'mtime_ns':cp_path.stat().st_mtime_ns},
      'selection_is_geometry_and_contribution_heuristic_not_semantic_GT':True}
    assert road.sum()>100 and bad.sum()>10,selection
    np.savez_compressed(out/'selection.npz',receiver_ids=ids[road],artifact_ids=ids[bad],
      receiver_xyz_anchor=projected['xyz'][road],artifact_xyz_anchor=projected['xyz'][bad],
      receiver_rgb_anchor=projected['colors'][road],receiver_base_opacity=arr(cap[6].sigmoid()).ravel()[ids[road]],
      artifact_base_opacity=arr(cap[6].sigmoid()).ravel()[ids[bad]])
    (out/'selection_receipt.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection))
if __name__=='__main__':main()
