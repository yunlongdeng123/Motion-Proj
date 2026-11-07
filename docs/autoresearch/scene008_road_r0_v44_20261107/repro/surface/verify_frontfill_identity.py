"""Read-only identity and CPU coordinate checks for the v2 domain repair."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--surface",type=Path,required=True);args=p.parse_args()
    root=args.surface;out=root/"frontfill_v2/identity_verification.json"
    if out.exists():raise FileExistsError("Preserve prior verification")
    p1=root/"road_reference_199.npz";p2=root/"frontfill_v2/road_reference_199.npz"
    with np.load(p1,allow_pickle=False) as old,np.load(p2,allow_pickle=False) as new:
        n=len(old["keys"]);m=len(new["keys"])
        checks={}
        for k in ("keys","local_xyz","world_xyz","model_xyz","normal_local","normal_world",
                  "gaussian_sigma_m","active","reference_linear","reference_srgb","agree_count",
                  "geometry_measured_support","actual_return_support_count"):
            checks["original_"+k+"_exact"]=bool(np.array_equal(old[k],new[k][:n]))
        for k in ("source_valid","source_agree"):
            checks["original_"+k+"_exact"]=bool(np.array_equal(old[k],new[k][:,:n]))
        checks["original_accepted_source_rgb_exact"]=bool(np.array_equal(old["source_rgb"],new["source_rgb"][:,:n],equal_nan=True))
        scale=float(new["meter_to_model"]);R=new["world_to_model"][:3,:3]
        normals=new["normal_world"]@R.T/scale;normals/=np.linalg.norm(normals,axis=1)[:,None]
        q=np.c_[1+normals[:,2],-normals[:,1],normals[:,0],np.zeros(m)]
        q/=np.linalg.norm(q,axis=1)[:,None]
        w,x,y,z=q.T
        rotated=np.c_[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]
        checks["quaternion_z_to_model_normal_maxabs"]=float(np.abs(rotated-normals).max())
        checks["physical_sigma_exact"]=bool(np.max(np.abs(new["gaussian_sigma_m"]-[.052,.052,.015]))<3e-9)
        added=np.arange(m)>=n
        checks["new_origin_flags_exact"]=bool(np.array_equal(new["candidate_origin"],added.astype(np.uint8)))
        checks["added_domain_flags_do_not_claim_legacy_membership"]=bool(~new["legacy_domain"][added].any())
        stats=dict(original_candidates=n,total_candidates=m,added_candidates=m-n,
                   original_active=int(old["active"].sum()),new_active=int(new["active"].sum()),
                   added_active=int((added&new["active"]).sum()),added_unknown=int((added&~new["active"]).sum()))
    assert all(x for x in checks.values() if isinstance(x,bool))
    assert checks["quaternion_z_to_model_normal_maxabs"]<1e-6
    report=dict(schema="R0_FRONTFILL_V2_IDENTITY_VERIFICATION",status="PASS",CPU_only=True,
                v1_sha256=sha(p1),v2_sha256=sha(p2),checks=checks,counts=stats,
                GPU_used=False,checkpoint_loads=0,optimizer_updates=0)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":main()
