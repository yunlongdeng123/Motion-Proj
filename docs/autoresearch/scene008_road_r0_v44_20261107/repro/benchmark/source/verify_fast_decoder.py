"""CPU parity of existing fast decoder against immutable reference and smoke."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import json
from pathlib import Path
import time
import sys
import numpy as np
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    reader=b.raw_reader();pb=reader.protobuf();crc=reader.crc_function()
    extraction=b.read(b.FULL/"extraction.json");t=time.time()
    with Path(extraction["raw_path"]).open("rb") as stream:
        payload=b.record_at(stream,extraction["frames"][137],crc)
    frame=pb.Frame.FromString(payload)
    values=[frame.lasers[0].ri_return1.range_image_pose_compressed]
    values.extend(getattr(l,"ri_return"+str(k)).range_image_compressed for l in frame.lasers for k in (1,2))
    checks=[]
    for i,value in enumerate(values):
        if not value:
            continue
        a=reader.reference_matrix(value,pb);c=reader.matrix(value,pb)
        if a.shape!=c.shape or a.dtype!=c.dtype or not np.array_equal(a,c):
            raise AssertionError("Fast/reference MatrixFloat mismatch")
        checks.append(dict(matrix=i,shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=True))
    fast_start=time.time();cloud,details=reader.lidar(frame,pb);fast_seconds=time.time()-fast_start
    anchor=np.asarray(b.read(args.output/"public/patch_manifest.json")["coordinate_anchor"],float)
    transform=np.linalg.inv(anchor)@np.asarray(frame.pose.transform,float).reshape(4,4)
    xyz=(cloud[:,:3].astype(float)@transform[:3,:3].T+transform[:3,3]).astype(np.float32)
    with np.load(args.output/"public/inputs/smoke_0_137_198/appearance_blind/geometry/137.npz",allow_pickle=False) as source:
        if not np.array_equal(xyz,source["local_xyz"]):
            raise AssertionError("Full fast raw cloud differs from slow original smoke")
    b.write(args.output/"audit/fast_decoder_parity.json",dict(status="PASS_CPU_DECODER_BITWISE_PARITY",raw=137,
            decoder=reader.fast_matrix_dependency,matrices=checks,matrix_count=len(checks),full_cloud_rows=len(xyz),
            full_anchor_local_XYZ_matches_original_slow_smoke_exactly=True,fast_decode_seconds=fast_seconds,
            seconds=time.time()-t,GPU_used=False,training_updates=0))
    print(json.dumps(dict(status="PASS_CPU_DECODER_BITWISE_PARITY",matrices=len(checks),fast_decode_seconds=fast_seconds,full_cloud_rows=len(xyz))),flush=True)


if __name__=="__main__":
    main()
