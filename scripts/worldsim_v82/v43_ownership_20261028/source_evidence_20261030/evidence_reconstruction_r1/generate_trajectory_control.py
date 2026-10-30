"""Same independent world/time/sensors; replace only 16 camera extrinsics.

Run under the same Blender 3.6.23 executable as the original generator.
No acquisition runs by importing this file. The trajectory must pass a separate preflight.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from mathutils import Matrix
import bpy


def main():
    argv=sys.argv[sys.argv.index('--')+1:]
    def take(flag):
        index=argv.index(flag); value=argv[index+1]; del argv[index:index+2]; return value
    trajectory=Path(take('--trajectory')).resolve()
    base=Path(take('--base-generator')).resolve()
    preflight=Path(take('--preflight')).resolve()
    plan=json.loads(trajectory.read_text())
    check=json.loads(preflight.read_text())
    trajectory_sha=hashlib.sha256(trajectory.read_bytes()).hexdigest()
    if check.get('status')!='PASS_TRAJECTORY_PREFLIGHT' or check.get('trajectory_sha256')!=trajectory_sha:
        raise ValueError('A passed visibility/sensor preflight for this exact trajectory is required')
    rows={int(r.get('frame_id',r.get('id'))):r for r in plan['frames']}
    if set(rows)!=set(range(16)):
        raise ValueError('Control must preserve exactly the same 16 times')
    for row in rows.values():
        c=np.asarray(row['camera_to_world'],dtype=float)
        if c.shape!=(4,4) or not np.isfinite(c).all() or not np.allclose(c[3],[0,0,0,1]) or not np.allclose(c[:3,:3].T@c[:3,:3],np.eye(3),atol=1e-6):
            raise ValueError('Finite rigid OpenCV camera extrinsics required')
    spec=importlib.util.spec_from_file_location('same_world_generator',base)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original=module.set_frame
    cv_to_blender=np.diag([1.,-1.,-1.,1.])
    def set_frame(frame,config,camera,car,edit='none',shadow=True):
        original(frame,config,camera,car,edit,shadow)
        camera.matrix_world=Matrix((np.asarray(rows[int(frame)]['camera_to_world'])@cv_to_blender).tolist())
        bpy.context.view_layer.update()
    module.set_frame=set_frame
    original_metadata=module.camera_metadata
    def metadata(*args,**kwargs):
        result=original_metadata(*args,**kwargs)
        # Blender's float32 Matrix.inverted loses >1e-6 closure at larger camera translations.
        # Keep the rendered camera c2w exactly, export its inverse in float64.
        result['world_to_camera']=np.linalg.inv(np.asarray(result['camera_to_world'],dtype=np.float64)).tolist()
        return result
    module.camera_metadata=metadata
    sys.argv=sys.argv[:sys.argv.index('--')+1]+argv
    output=Path(argv[argv.index('--output')+1]).resolve()
    module.main()
    receipt=dict(status='SAME_WORLD_CAMERA_TRAJECTORY_CONTROL_ACQUIRED',trajectory=str(trajectory),trajectory_sha256=trajectory_sha,
        base_generator=str(base),base_generator_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        changed='camera extrinsics only at the same 16 simulator times; legal RGB/LiDAR/masks change as their physical consequence',
        unchanged='world meshes/material/light, actor motion, intrinsics, renderer seeds/samples, LiDAR ray pattern, time axis',
        GT_isolation='observations only may enter adapter/training; privileged surface/depth/edited reference remain evaluation/preflight-only')
    (output/'trajectory_control_receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
