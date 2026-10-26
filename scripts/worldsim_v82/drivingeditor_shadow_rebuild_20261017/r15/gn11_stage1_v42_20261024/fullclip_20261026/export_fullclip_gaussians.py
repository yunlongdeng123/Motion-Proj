"""Export source-confirmed, untrained road Gaussians in the baseline model frame."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);a=dict(np.load(p/'full199_initialization.npz'))
m=json.loads(Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json').read_text())
ids=np.flatnonzero(a['surface_supported']&a['eligible']);A=np.asarray(m['world_to_model']);scale=m['meter_to_model']
n=a['normal_world'][ids]@A[:3,:3].T/scale;n/=np.linalg.norm(n,axis=1)[:,None]
q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];q/=np.linalg.norm(q,axis=1)[:,None]
xyz=a['model_xyz'][ids];sigma=a['gaussian_sigma_m'][ids]*scale;rgb=a['rgb'][ids]
assert len(ids)==7848 and np.isfinite(xyz).all() and (sigma>0).all() and np.allclose(np.linalg.norm(q,axis=1),1)
np.savez_compressed(p/'full199_gaussians_model.npz',unit_ids=ids,xyz=xyz.astype(np.float32),scaling=sigma.astype(np.float32),rotation_wxyz=q.astype(np.float32),
    rgb=rgb,opacity=np.full(len(ids),.5,np.float32),world_to_model=A,coordinate_anchor=np.asarray(m['coordinate_anchor']),meter_to_model=np.array(scale),
    first_source=a['first_source'][ids],second_source=a['second_source'][ids],source_count=a['source_count'][ids])
names=['x','y','z','nx','ny','nz']+['f_dc_'+str(i) for i in range(3)]+['opacity']+['scale_'+str(i) for i in range(3)]+['rot_'+str(i) for i in range(4)]
v=np.zeros(len(ids),dtype=[(k,'<f4') for k in names]);values=np.c_[xyz,n,(rgb-.5)/.28209479177387814,np.zeros(len(ids)),np.log(sigma),q]
for i,k in enumerate(names):v[k]=values[:,i]
with (p/'full199_gaussians_model.ply').open('xb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(ids))+'\n'+''.join('property float '+k+'\n' for k in names)+'end_header\n').encode());v.tofile(f)
(p/'gaussian_export.json').write_text(json.dumps(dict(count=len(ids),coordinates='fixed V41-B model coordinates',meter_to_model=scale,SH_degree=0,opacity_storage='PLY logit; NPZ activated',scale_storage='PLY natural log; NPZ activated',training_updates=0,not_a_full_native_training_checkpoint=True),indent=2))
print(len(ids))
