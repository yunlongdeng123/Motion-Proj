"""独立核对源RGB、相机重投影和证据来源；不把合同测试当质量通过。"""
import sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from video_review import scene_frame
BASE=ROOT.parent;spec=next(s for s in read(ROOT/'registration.json')['scenes'] if s['name']=='scene_0255');data=Path(spec['data']);inst=read(data/'instances/instances_info.json');rows=[]
for run in ['r12','r13']:
 folder=BASE/run/'scene_0255';index=[r for r in read(folder/'source_index.json') if 'source_id' in r];raw=np.load(folder/'sources.npz');error=0.;n=0
 for j,r in enumerate(index):
  assert r['source_id']==j
  im=np.array(Image.open(data/'images'/f'{r["frame"]:03}_{r["camera"]}.jpg').convert('RGB').resize((r['depth_hw'][1],r['depth_hw'][0]),Image.Resampling.BILINEAR));uv=raw[f'{j}_uv'];assert np.array_equal(im[uv[:,1],uv[:,0]],raw[f'{j}_rgb'])
  fr=scene_frame(spec['spec'],r['frame'],inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');world=transform(raw[f'{j}_local'],np.array(b['pose']));c,k=camera(fr,r['camera'],r['depth_hw']);cp=transform(world,np.linalg.inv(c));q=cp@k.T;projected=q[:,:2]/q[:,2:];err=np.max(np.abs(projected-uv));assert err<.01;error=max(error,float(err));n+=len(uv)
 count=0
 for i in [0,7,15,29]:
  accepted=mask(folder/f'accepted_{i:05}.png');p=np.load(folder/f'provenance_{i:05}.npz');yy,xx=np.where(accepted)
  assert not (accepted&mask(ROOT/'scene_0255/protect'/f'{i:05}.png')).any()
  image=rgb(folder/f'proposal_{i:05}.png')
  for y,x in zip(yy,xx):
   sid=int(p['source_id'][y,x]);pid=int(p['source_point'][y,x]);second=int(p['second_source'][y,x]);assert min(sid,second)>=0
   assert abs(index[sid]['frame']-index[second]['frame'])>=5
   assert np.array_equal(image[y,x],raw[f'{sid}_rgb'][pid])
  count+=len(yy)
 rows.append(dict(run=run,source_views=len(index),source_points=n,raw_RGB_exact=True,self_reprojection_max_px=error,accepted_provenance_pixels=count,source_time_separation_verified=True,protect_excluded=True))
 raw.close()
cal=read(BASE/'r13/scene_0255/calibration.json')['records'];admitted=[r for r in cal if r['admitted']]
assert len(admitted)==5
for r in admitted:
 assert min(r['fit_anchors'],r['heldout_anchors'])>=8 and r['heldout_corrected_median_m']<=.25 and r['heldout_corrected_q90_m']<=.5
 assert r['heldout_corrected_median_m']<r['heldout_raw_median_m']
v=read(BASE/'identity_review/video_validation.json');assert len(v['videos'])==8 and v['decoded_frames']==240
dump(BASE/'identity_review/evidence_validation.json',dict(runs=rows,calibration_admitted_views=len(admitted),video_count=8,actual_decoded_frames=240,scope='source/provenance and numerical contracts only, not visual acceptance',human_verdict=None))
print('EVIDENCE VALIDATION PASSED',rows)
