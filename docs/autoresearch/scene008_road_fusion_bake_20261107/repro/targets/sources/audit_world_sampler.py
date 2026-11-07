"""Replay all49 stored target pixels from the one world atlas; CPU only."""
from pathlib import Path
import json,hashlib,time
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1/targets')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();m=json.loads((B/'target_manifest.json').read_text());a=np.load(B/'world_atlas_targets.npz');keys=a['keys'];lo=keys.min(0);shape=keys.max(0)-lo+1
 grid=np.full(tuple(shape),-1,int);ij=keys-lo;grid[ij[:,0],ij[:,1]]=np.arange(len(keys));offset=np.median(a['local_xyz'][:,:2]-keys*.08,axis=0);records=[]
 assert sha(B/'world_atlas_targets.npz')==m['world_atlas_sha256']
 for row in m['rows']:
  path=B/row['file'];assert sha(path)==row['sha256'];z=np.load(path);mask=z['fusion_replace_mask'];xy=z['atlas_local_xy_m'][mask].astype(float);uv=(xy-offset)/.08-lo
  ix=np.floor(uv[:,0]).astype(int);iy=np.floor(uv[:,1]).astype(int);dx=uv[:,0]-ix;dy=uv[:,1]-iy
  ix=np.clip(ix,0,grid.shape[0]-2);iy=np.clip(iy,0,grid.shape[1]-2)
  ids=np.maximum(np.stack([grid[ix,iy],grid[ix+1,iy],grid[ix,iy+1],grid[ix+1,iy+1]],1),0)
  w=np.stack([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy],1);errors={}
  for key,texture in [('direct_rgb','source_chart_rgb'),('fusion_rgb','fusion_chart_rgb')]:
   reconstructed=np.sum(a[texture][ids]*w[:,:,None],axis=1);diff=abs(reconstructed-z[key][mask]);maximum=float(diff.max()) if diff.size else 0.
   assert maximum<5e-5,(row['raw'],key,maximum)
   errors[key]=dict(maxabs=maximum,meanabs=float(diff.mean()) if diff.size else None)
  records.append(dict(raw=row['raw'],role=row['role'],replayed_pixels=int(mask.sum()),errors=errors))
 out=dict(schema='WORLD_ATLAS_ALL49_SAMPLER_REPLAY_V1',status='PASS_ONE_WORLD_TEXTURE_REPLAY_ALL49',
   target_manifest_sha256=sha(B/'target_manifest.json'),world_atlas_sha256=sha(B/'world_atlas_targets.npz'),
   maximum_over_all49=max(v['maxabs'] for r in records for v in r['errors'].values()),rows=records,seconds=time.time()-start,
   interpretation='All target replacement pixels replay from the same serialized world texture and localXY, within float32-coordinate tolerance. No view/state colour variables. This is representation consistency, not independent photometric accuracy.',
   state_axis=False,targets_modified=False,GPU_forwards=0)
 (B/'target_consistency_receipt.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:out[k] for k in ['status','maximum_over_all49','seconds']}))
if __name__=='__main__':main()
