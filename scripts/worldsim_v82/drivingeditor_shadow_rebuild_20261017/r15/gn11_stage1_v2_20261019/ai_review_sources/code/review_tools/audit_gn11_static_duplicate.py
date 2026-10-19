"""Read-only CPU audit of actual matched-five GN11 static/dynamic duplication.

No rasterizer, CUDA, training updates, or changes to A/B inputs/checkpoints.
Standalone static alpha is not a semantic ownership test.
"""
from pathlib import Path
import os,sys,json,hashlib,time,math,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import torch,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from plyfile import PlyData
E=Path(__file__).resolve().parent
N=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018')
S=N/'training_source/scene0008_089_137'
R=N/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
D=E/'gn11_static_duplicate_20261019'
IDS=[5,26,18,6,37]
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def ply(p):
 v=PlyData.read(str(p))['vertex'].data
 return np.stack([v[k] for k in ['x','y','z']],-1).astype(np.float64)
def measure(x):
 a=np.asarray(x).ravel()
 return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(count=0)
def main():
 if (D/'receipt.json').exists():print((D/'receipt.json').read_text());return
 D.mkdir(exist_ok=True);(D/'frames').mkdir(exist_ok=True);(D/'actual_source').mkdir(exist_ok=True)
 torch.set_num_threads(2);start=time.monotonic()
 files=['train.py','gaussian_renderer/__init__.py','scene/waymo_loader2.py','scene/gaussian_model.py','utils/hybrid30k_losses.py','utils/paper_losses.py','utils/paper_masks.py']
 hashes={}
 for f in files:
  p=S/f;hashes[f]=hashlib.sha256(p.read_bytes()).hexdigest();q=D/'actual_source'/f;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 state=torch.load(str(R/'training_state30000.pth'),map_location='cpu',weights_only=False);cfg=state['config'];A=Path(cfg['source_path']);del state
 selected=read(A/'dynamic_multi_actor_preparation.json');event=read(R/'checkpoint_events/iteration_030000.json')
 # Invoke exactly the actual native CPU mask decoder and loss implementations.
 sys.path.insert(0,str(S));from utils.paper_masks import _read_mask
 from utils.hybrid30k_losses import static_complement_rgb_loss,full_dynamic_opacity_mse
 from utils.loss_utils import ssim
 frame_rows=[];all_labels=[]
 for raw in range(89,138):
  fid=f'0008{raw:03d}';labels=np.array(Image.open(A/'matched_training_labels'/f'{fid}.png'));all_labels.append(labels)
  decoded=_read_mask(str(A/'dynamic_masks/dynamic_mask_0'/f'{fid}.png'),480,320).numpy()[0]
  assert np.array_equal(decoded,labels>0)
  ev=next(r for r in event['frames'] if r['image_name']==fid);m=read(Path(ev['hybrid_metrics']))
  with np.load(m['arrays'],allow_pickle=False) as z:
   assert np.array_equal(z['actor_mask'][0],labels>0)
   rows=[]
   for slot,actor in enumerate(IDS,1):
    take=labels==slot
    if not take.any():continue
    rows.append(dict(actor=actor,body_pixels=int(take.sum()),canonical_mask_coverage=1.,
      dynamic_alpha_mean=float(z['alpha_d'][0][take].mean()),static_alpha_mean=float(z['alpha_s'][0][take].mean()),
      both_standalone_alpha_over_0p9=float(((z['alpha_d'][0]>.9)&(z['alpha_s'][0]>.9))[take].mean()),
      dynamic_body_RGB_MAE_255=float(np.abs(np.clip(z['dynamic'][:,take],0,1)-z['gt'][:,take]).mean()*255),
      composite_body_RGB_MAE_255=float(np.abs(np.clip(z['composite'][:,take],0,1)-z['gt'][:,take]).mean()*255)))
   frame_rows.append(dict(raw=raw,actors=rows))
   if raw==129:
    saved={k:z[k].copy() for k in z.files}
    for key in ['gt','composite','static','dynamic']:
     arr=np.rint(np.clip(z[key],0,1).transpose(1,2,0)*255).astype(np.uint8);Image.fromarray(arr).save(D/'frames'/f'raw129_{key}.png')
    overlay=np.array(Image.open(D/'frames/raw129_gt.png'))
    for slot,color in [(4,np.array([255,140,0])),(5,np.array([0,230,230]))]:
     take=labels==slot;overlay[take]=np.rint(overlay[take]*.45+color*.55)
    Image.fromarray(overlay).save(D/'frames/raw129_mask_overlay.png')
 # Exact current image-space gradient: excludes all masked pixels, even SSIM boundary effects.
 mask=torch.from_numpy(saved['actor_mask']);gt=torch.from_numpy(saved['gt']);static=torch.from_numpy(saved['static']).requires_grad_()
 loss=static_complement_rgb_loss(static,gt,mask,ssim,cfg['lambda_dssim']);g,=torch.autograd.grad(loss,static)
 assert torch.count_nonzero(g[:,mask[0]>0])==0
 da=torch.from_numpy(saved['alpha_d']).requires_grad_();sa=torch.from_numpy(saved['alpha_s']).requires_grad_()
 op=full_dynamic_opacity_mse(da,mask);og=torch.autograd.grad(op,(da,sa),allow_unused=True)
 assert og[1] is None
 gradient=dict(actual_functions_invoked=True,device='cpu',raw=129,static_RGB_loss=float(loss),
   static_RGB_gradient_inside_all_actor_pixels_absmax=float(g[:,mask[0]>0].abs().max()),
   static_RGB_gradient_outside_actor_absmax=float(g[:,mask[0]==0].abs().max()),
   dynamic_opacity_gradient_to_static_alpha_connected=og[1] is not None,
   scope='Saved real30K image tensors and real mask; image-space autograd only, not Gaussian-parameter gradients or new rasterization')
 # Re-run the actual initial static subtraction in sensor/world meters.
 original=ply(A/'points3d.ply');dynamic=np.concatenate([ply(Path(p)) for p in cfg['followup_scene_spec']['actor_ply_paths']])
 near,_=cKDTree(dynamic).query(original,k=1,distance_upper_bound=.1,workers=2);keep=near>=.1
 static_init=original[keep];tree=cKDTree(static_init);del original,near
 # Measured source points inside body masks AND true contemporaneous3D vehicle boxes.
 # Counts are candidate vehicle-surface leakage, not render contribution or owner ground truth.
 leakage=[]
 for raw in [89,107,128,129,137]:
  fid=f'0008{raw:03d}';points=np.fromfile(A/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(float);pose=np.loadtxt(A/'pose'/f'{fid}.txt')
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (A/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
  cam=np.c_[points,np.ones(len(points))]@cal['Tr_velo_to_cam_0'].reshape(3,4).T;uvw=np.c_[cam,np.ones(len(cam))]@cal['P0'].reshape(3,4).T
  uv=np.floor(uvw[:,:2]/np.maximum(uvw[:,2:3],1e-12)+.5).astype(np.int64);inside=(cam[:,2]>.01)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280)
  ann=read(A/'annotations'/f'{fid}.json');world=(np.c_[points,np.ones(len(points))]@pose.T)[:,:3]
  for slot,actor in [(3,6),(4,37)]:
   t=selected['actors'][slot]['raw_track_id'];b=next(a['box'] for a in ann['laser_labels'] if a['id']==t)
   c=np.array([b['center_x'],b['center_y'],b['center_z']]);h=b['heading'];rot=np.array([[math.cos(h),-math.sin(h),0],[math.sin(h),math.cos(h),0],[0,0,1.]])
   local=(points-c)@rot;obb=(np.abs(local)<=np.array([b['length'],b['width'],b['height']])/2).all(1)
   body=np.array(Image.open(A/f'dynamic_actor_assets/actor_{slot}/body_masks'/f'{fid}.png'))>0
   take=np.flatnonzero(inside&obb);take=take[body[uv[take,1],uv[take,0]]]
   dist,_=tree.query(world[take],workers=2,distance_upper_bound=.001)
   prior=selected['actors'][slot]['frames'][raw-89]
   leakage.append(dict(raw=raw,actor=actor,actual_body_and_OBB_LiDAR_points=len(take),points_with_initial_static_neighbor_within_1mm=int(np.isfinite(dist).sum()),
     retained_dynamic_prior_in_this_frame=prior['retained_real_points'],whole_frame_prior_rejection=prior['whole_frame_prior_rejection']))
 del tree,static_init
 # Existing point-space static temporal freedom, plus spatial OBB diagnostics (not owner).
 from scene.gaussian_model import GaussianModel
 captured,step=torch.load(str(R/'chkpnt30000.pth'),map_location='cpu',weights_only=False,mmap=True);assert step==30000
 xyz=captured[1].detach().numpy();vel=captured[9].detach().numpy();t=captured[7].detach().numpy();T=float(captured[16]);scale=.1
 temporal=dict(static_rows=len(xyz),static_base_velocity_m_per_model_time=measure(np.linalg.norm(vel,axis=1)/scale),
   static_velocity_nonzero_over_1e_6=int((np.linalg.norm(vel,axis=1)>1e-6).sum()),
   renderer_static_path='get_xyz_SHM(camera.timestamp); opacity *= get_marginal_t(camera.timestamp)',
   no_semantic_owner_field_in_native_stage1=True)
 del captured
 assert not torch.cuda.is_initialized()
 result=dict(status='COMPLETE_READONLY_MATCHED_FIVE_GN11_DUPLICATE_AUDIT',source=str(S),run=str(R),
   actual_source_sha256=hashes,frames=49,raw_anchor=129,mask_union_exact_all49=True,canonical_mask_matches_saved_real_eval_all49=True,
   raw129=next(x for x in frame_rows if x['raw']==129),per_frame=frame_rows,actual_CPU_gradient=gradient,
   initialization=dict(rule='Euclidean world/sensor-coordinate distance <0.1m to any selected dynamic prior point; not all body/OBB/swept-volume removal',dynamic_seed_points=len(dynamic),initial_static_points=int(keep.sum()),source_body_candidates=leakage,
     caveat='1mm nearest static source point is initialization evidence, not proof that those exact points cause the final image. OBB intersect body is not perfect surface GT.'),
   static_temporal_freedom=temporal,
   active_weights={k:cfg[k] for k in cfg if k.startswith('hybrid_lambda') or k in ['hybrid_nps_enabled','paper_lambda_nps','lambda_v_reg','lambda_t_reg']},
   GPU_used=False,training_updates=0,training_inputs_modified=False,checkpoints_modified=False,seconds=time.monotonic()-start,human_verdict=None,
   limitations=['Standalone alpha near1 can be valid road behind vehicle; not proof of ownership or wrong geometry.',
    'Exact offending Gaussian contributions and clean-road correctness require separate point-contribution/depth evidence; not established by this CPU audit.',
    'No new losses or hard2D static-alpha zero constraints added. Background behind a vehicle should still be opaque road, not empty.'])
 write(D/'receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','actual_source_sha256']},indent=2))
if __name__=='__main__':main()
