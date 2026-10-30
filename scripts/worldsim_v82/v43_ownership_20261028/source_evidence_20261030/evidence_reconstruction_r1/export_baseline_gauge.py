import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path); args=p.parse_args()
    if args.output and args.output.exists(): raise FileExistsError(args.output)
    transform=args.root/'native_scene_r3/transform_matrix.txt'
    camera_file=args.root/'formal_r3_10k/cameras.json'
    cameras=json.loads(camera_file.read_text())
    centers=[]
    for row in cameras:
        c2w=np.eye(4); c2w[:3,:3]=row['rotation']; c2w[:3,3]=row['position']
        # Match scene_utils.getNerfppNorm / getWorld2View2's float32 W2C path.
        centers.append(np.linalg.inv(np.linalg.inv(c2w).astype(np.float32))[:3,3])
    centers=np.asarray(centers,dtype=np.float32)
    calibration_spatial=float(1/(1.1*np.linalg.norm(centers-centers.mean(0),axis=1).max()))
    payload,_=torch.load(str(args.root/'formal_r3_10k/chkpnt10000.pth'),map_location='cpu',mmap=True,weights_only=False)
    if len(payload)!=18: raise ValueError('Unexpected native capture layout')
    actual=float(payload[15])
    if actual!=1.: raise ValueError('Expected current Scene hard-coded native spatial LR scale1')
    result=dict(world_to_model=np.loadtxt(transform).tolist(),native_spatial_lr_scale=actual,native_camera_extent=calibration_spatial,
        scale_provenance='Current Scene initializes both models with spatial_lr_scale=1; camera extent is a separate calibration-derived native field',
        provenance='baseline legal calibration + saved native optimizer-scale metadata, no GT geometry or semantic labels',
        source_transform_sha256=hashlib.sha256(transform.read_bytes()).hexdigest(),source_cameras_sha256=hashlib.sha256(camera_file.read_bytes()).hexdigest())
    if args.output: args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
