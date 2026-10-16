import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from shadow_math import project_gaussians, replay, rotation_matrix, ray_plane

torch.set_num_threads(2)
base=Path(__file__).parent.parent
plan=json.loads((Path(__file__).parent/'experiment_plan.json').read_text())
root=Path(plan['remote_root']); case=plan['primary_case']
out=base/'focus_cpu'; out.mkdir(exist_ok=False)
cp=torch.load(root/case['checkpoint_relative'],map_location='cpu',weights_only=False)
cap=cp['background']; spec=cp['spec']
sys.path.insert(0,str(root/'infra'))
from renderer import evaluate_sh_rgb
evaluation=root/case['evaluation_relative']
manifest=json.loads((evaluation/'scene_evaluation_manifest.json').read_text())
row=next(r for r in manifest['streams']['observed']['frames'] if r['frame_id']==case['frame_id'])
cam=row['camera']; time=row['model_time']
ref=json.loads(Path(spec['reference_config']).read_text())
readiness=json.loads((root/'metadata'/case['scene_id']/'CPU_readiness.json').read_text())
plane=np.asarray(readiness['SGCM_geometry']['plane_model'])
scale=float(np.linalg.norm(np.asarray(manifest['normalization'])[:3,:3],axis=0).mean())
arr=lambda t:t.detach().numpy().astype(np.float32,copy=False)
omega=2*math.pi/float(cap[16])
xyz=cap[1]+cap[9]*torch.sin((time-cap[7])*omega)/omega
marginal=torch.exp(-.5*((cap[7]-time)/cap[8].exp())**2).reshape(-1)
colors=evaluate_sh_rgb(torch.cat((cap[2],cap[3]),1),torch.nn.functional.normalize(xyz-torch.tensor(cam['camera_center']),dim=-1),int(cap[0]))
state=dict(xyz=arr(xyz),scales=arr(cap[4].exp()),rotation=arr(cap[5]),opacity=arr(cap[6].sigmoid()).ravel()*arr(marginal),colors=arr(colors),mask=arr(marginal)>.05)
projected=project_gaussians(state,cam,ref['neg_fov'])
# Manually traced visible artifact silhouette after actual image review. This is
# a diagnostic footprint, never a cast-shadow GT annotation or training mask.
polygon=[(54,257),(59,253),(63,248),(65,251),(71,251),(74,247),(75,243),(82,245),(94,243),(108,243),(118,244),(123,246),(130,247),(135,247),(138,241),(145,236),(151,232),(158,232),(155,236),(148,241),(143,246),(138,251),(130,256),(115,260),(85,261),(64,260)]
mask_image=Image.new('L',(480,320)); ImageDraw.Draw(mask_image).polygon(polygon,fill=255)
mask=np.asarray(mask_image)>0
mask_image.save(out/'reviewed_artifact_footprint.png')
yy,xx=np.where(mask); pixels=np.c_[xx,yy].astype(np.float32)
rgb,alpha,mass,_=replay(projected,pixels)
with np.load(evaluation/row['npz']) as f:
    expected=f['static_raw_rgb'][:,yy,xx].T; gt=f['gt']; bodies=f['observed_masks'].any(0)
err=np.abs(rgb-expected)
assert err.mean()<1e-4 and err.max()<.002
height=(projected['xyz']@plane[:3]+plane[3])/np.linalg.norm(plane[:3])/scale
rot=rotation_matrix(projected['rotation']); n=plane[:3]/np.linalg.norm(plane[:3])
sigma_n=np.linalg.norm(np.einsum('nji,j->ni',rot,n)*projected['scales'],axis=1)/scale
dark=projected['colors'].max(1)<.05
ids=projected['ids']; groups={
 'all_visible_points':np.ones(len(ids),bool),
 'near_black_SH_max_lt_005':dark,
 'near_black_center_abs_height_le_015m':dark&(abs(height)<=.15),
 'near_black_center_abs_height_gt_050m':dark&(abs(height)>.5),
 'near_black_normal_3sigma_gt_050m':dark&(3*sigma_n>.5),
 'near_black_max_scale_gt_100m':dark&(projected['scales'].max(1)/scale>1),
 'near_black_center_abs_height_015_to_050m':dark&(abs(height)>.15)&(abs(height)<=.5),
}
summary={}; groupids={}
for name,sel in groups.items():
    summary[name]={'footprint_mean_alpha_T_CPU':float(mass[sel].sum()/len(pixels)), 'contributing_point_count':int((sel&(mass>0)).sum())}
    groupids[name]=ids[sel].tolist()
rank=np.argsort(np.where(dark,mass,0))[::-1]
rank=rank[(mass[rank]>0)&dark[rank]][:100]
times=np.asarray([r['model_time'] for r in manifest['streams']['observed']['frames']])
original,_=torch.load(spec['background_checkpoint'],map_location='cpu',weights_only=False)
entries=[]
for k in rank:
    pid=int(ids[k]); velocity=arr(cap[9][pid]); sigma=float(cap[8][pid].exp()); tcenter=float(cap[7][pid])
    trajectory=arr(cap[1][pid])+np.sin((times-tcenter)*omega)[:,None]*velocity/omega
    entries.append(dict(static_id=pid,mean_alpha_T=float(mass[k]/len(pixels)),signed_height_m=float(height[k]),normal_3sigma_m=float(3*sigma_n[k]),scales_m=(projected['scales'][k]/scale).tolist(),depth_m=float(projected['depth'][k]/scale),RGB=projected['colors'][k].tolist(),position_span_m=float(np.linalg.norm(trajectory.max(0)-trajectory.min(0))/scale),raw_velocity=velocity.tolist(),regularized_velocity=(velocity*np.exp(-sigma/float(cap[16])/2*float(cap[17]))).tolist(),marginal_range=[float(np.exp(-.5*((times-tcenter)/sigma)**2).min()),float(np.exp(-.5*((times-tcenter)/sigma)**2).max())],original_GN11_xyz=arr(original[1][pid]).tolist(),current_base_xyz=arr(cap[1][pid]).tolist(),original_GN11_scales_m=(arr(original[4][pid].exp())/scale).tolist()))
probe_results=[]
for xy in case['probe_pixels_xy']:
    _,pa,pm,traces=replay(projected,np.asarray([xy],np.float32))
    plane_xyz=ray_plane(xy,cam,plane)
    ground_z=float((plane_xyz@np.asarray(cam['world_view_transform'])[:3,:3]+np.asarray(cam['world_view_transform'])[3,:3])[2]/scale)
    probe_results.append({'xy':xy,'GT_RGB':gt[:,xy[1],xy[0]].tolist(),'body_mask':bool(bodies[xy[1],xy[0]]),'static_RGB':expected[np.argmin(((pixels-np.array(xy))**2).sum(1))].tolist(),'alpha':float(pa[0]),'plane_hypothesis_camera_depth_m':ground_z,'group_alpha_T':{name:float(pm[sel].sum()) for name,sel in groups.items()},'top20':traces[0]['contributors']})
result=dict(schema='focused_artifact_CPU_v1',polygon=polygon,footprint_pixels=int(mask.sum()),footprint_is_shadow_GT=False,body_mask_overlap_pixels=int((mask&bodies).sum()),CPU_replay_MAE=float(err.mean()),CPU_replay_max=float(err.max()),group_summary=summary,near_black_threshold='max evaluated SH RGB <0.05; appearance grouping only, not shadow semantics',group_ids=groupids,top100_near_black=entries,probes=probe_results,scale=scale)
(out/'focus_attribution.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['group_ids','top100_near_black','probes']}))
