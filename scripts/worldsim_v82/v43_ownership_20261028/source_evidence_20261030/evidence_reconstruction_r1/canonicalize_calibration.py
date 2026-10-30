"""Fix only inverse metadata precision of the new camera-control acquisition.

No RGB, LiDAR, camera_to_world, masks, world state or old dataset is modified.
"""
import json
from pathlib import Path
import shutil
import numpy as np

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/evidence_reconstruction_r1/dataset_hybrid8')
paths=[root/'observations/manifest.json',root/'observations/cameras.json',root/'privileged/reference_manifest.json',root/'privileged/heldout/manifest.json']
changes=[]
def visit(value,path):
    if isinstance(value,dict):
        if 'camera_to_world' in value and 'world_to_camera' in value:
            c=np.asarray(value['camera_to_world'],dtype=np.float64); old=np.asarray(value['world_to_camera'])
            inv=np.linalg.inv(c)
            changes.append(dict(path=str(path),frame=value.get('frame_id',value.get('id')),old_inverse_closure=float(np.abs(c@old-np.eye(4)).max()),new_inverse_closure=float(np.abs(c@inv-np.eye(4)).max()),camera_to_world_unchanged=True))
            value['world_to_camera']=inv.tolist()
        for x in value.values():visit(x,path)
    elif isinstance(value,list):
        for x in value:visit(x,path)
for path in paths:
    backup=path.with_suffix('.before_inverse_precision.json')
    if backup.exists():raise FileExistsError(backup)
    shutil.copy2(path,backup)
    data=json.loads(path.read_text());visit(data,path);path.write_text(json.dumps(data,indent=2))
(root/'calibration_precision_receipt.json').write_text(json.dumps(dict(reason='float32 Blender Matrix.inverted closure exceeds original validator1e-6 for richer trajectory; recompute only inverse export in float64',camera_or_RGB_or_LiDAR_changed=False,rows=changes),indent=2))
