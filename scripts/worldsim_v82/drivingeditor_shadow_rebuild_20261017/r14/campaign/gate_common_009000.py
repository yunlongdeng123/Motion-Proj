"""Once-only evidence intervention. All correspondence inputs are frozen BUILD data."""
from pathlib import Path
import sys,json,numpy as np,torch
from scipy.spatial import cKDTree
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');ROOT=R/'ownership_rebuild_r14';TARGETS=[5,6,12,18,25,26,37]
sys.path.insert(0,str(R/'implementation_r14_final/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,save,write
from shadow_rebuild.math3d import covariance,quaternion_matrix
from shadow_rebuild.prepare import transform
torch.set_num_threads(2)
def load(relative):
 saved=torch.load(ROOT/relative,map_location='cpu',weights_only=False)
 if saved['global_step']!=9000 or saved['stage']!='geometry':raise ValueError('C intervention only once at the pre-registered9000 gate')
 cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate();d=Dataset(ROOT/'idsplat_frontend/data_IDSplat','cpu',True)
 if saved['input_contract']!=d.manifest:raise ValueError('Frozen B/C input contract differs')
 model=SceneModel(seed_from_state(saved['state']),cfg);model.load_state_dict(saved['state']);opt=model.optimizer('geometry');opt.load_state_dict(saved['optimizer'])
 return saved,cfg,d,model,opt
def body_bank(actor):return dict(np.load(ROOT/f'body_bank_a{actor:02d}.npz'))
def road_bank(actor):
 raw=dict(np.load(ROOT/f'source_candidate_evidence_a{actor:02d}.npz'));take=raw['road_positive'];return {k:v[take] for k,v in raw.items()}
def build_only(seed,d):
 frames=np.asarray(seed['source_frame']);raw=np.asarray(seed['source_raw_row'])
 if np.any(raw%5==0) or np.any(raw<0) or any(d.rows[int(i)]['split']!='train' for i in np.unique(frames)):raise ValueError('QUERY or unknown measurement entered BUILD intervention')
def volume_points(points,tree,mean,precision,radius,sigma):
 ids=np.asarray(tree.query_ball_point(mean,radius),np.int64)
 if not len(ids):return ids
 delta=points[ids]-mean;return ids[np.einsum('ni,ij,nj->n',delta,precision,delta)<sigma*sigma]
def source_visibility(d):
 """Actual source-frame mask/depth evidence; no QUERY RGB or all-return zbuffer."""
 records={};sup=ROOT/'shared_masks_r14'
 for i,row in enumerate(d.rows):
  if row['split']!='train':continue
  raw=np.load(ROOT/'raw_measurements'/(row['frame_id']+'.npz'));pixel=raw['pixel'].astype(np.int64);ids=np.flatnonzero(raw['BUILD']&raw['in_image']);bins=pixel[ids,1]*480+pixel[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,raw['camera_z'][ids]);front=ids[raw['camera_z'][ids]<=depth[bins]+.1];masks=np.load(sup/(row['frame_id']+'.npz'));body=[]
  for a in TARGETS:
   good=front[(raw['candidate_owner'][front]==a)&bool(masks['mask_known'][a])];good=good[masks['masks'][pixel[good,1],pixel[good,0],a]];body.append(raw['raw_row'][good])
  records[i]={'front_raw_row':raw['raw_row'][front],'body_raw_rows':dict(zip(TARGETS,body))}
 return records
def motion_source_keys():
 tr=np.load(ROOT/'independent_tracks/actual_tracks.npz');result={}
 for a in TARGETS:
  take=(tr['candidate_actor']==a)&(tr['probability_static_actor_unknown'][:,1]>.7);fi=tr['frame_index'][take].reshape(-1);rr=tr['raw_lidar_row'][take].reshape(-1);result[a]=set((fi*1000000+rr).tolist())
 return result
