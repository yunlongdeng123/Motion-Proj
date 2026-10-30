"""Meaningful gauge/row-column equivalence check, no simulator or GPU needed."""
import json
from pathlib import Path
import numpy as np
from shared_gauge import transform_cameras

root=Path(__file__).resolve().parent
gauge=json.loads((root/'baseline_gauge.json').read_text(encoding='utf-8-sig'))
T=np.asarray(gauge['world_to_model']); rng=np.random.default_rng(431030)
poses=[]
for _ in range(16):
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    c=np.eye(4); c[:3,:3]=q; c[:3,3]=rng.normal(size=3)*7
    poses.append(c[:3])
poses=np.asarray(poses)
converted,shared,scale=transform_cameras(poses,T)
errors=[]
for old,new in zip(poses,converted):
    camera_points=np.column_stack([rng.normal(size=(50,2)),rng.uniform(.5,30,size=50)])
    world=camera_points@old[:3,:3].T+old[:3,3]
    model=world@T[:3,:3].T+T[:3,3]
    recovered=(model-new[:3,3])@new[:3,:3]
    errors.append(float(np.abs(recovered/scale-camera_points).max()))
assert max(errors)<1e-9
assert np.allclose(shared,T) and abs(scale-.1)<1e-12
report=dict(status='PASS_SHARED_GAUGE_MATH',camera_count=16,max_camera_coordinate_world_m_error=max(errors),scale=scale,baseline_native_spatial_lr_scale=gauge['native_spatial_lr_scale'],native_camera_extent=gauge['native_camera_extent'],GPU_or_training_executed=False)
(root/'shared_gauge_math_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
