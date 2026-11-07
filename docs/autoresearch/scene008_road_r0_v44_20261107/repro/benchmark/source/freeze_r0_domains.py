"""Freeze V44's original evaluation domains for R0; CPU and no predictions.

This integration-only data is intentionally outside strict benchmark public/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
BASE=Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OBS=BASE/"scene008_instances_20261105_r1/observations/frames"
FIXED=BASE/"scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(a):
    a=np.asarray(a,bool).squeeze()
    if a.shape!=(320,480):
        raise ValueError("Original mask shape changed")
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    dest=a.output/"integration_domains"
    if dest.exists():
        raise FileExistsError("Preserve earlier fixed domains")
    dest.mkdir(parents=True)
    with np.load(FIXED,allow_pickle=False) as fixed:
        np.testing.assert_array_equal(fixed["raw_ids"],np.arange(89,138))
        if fixed["layer_names"].tolist()[3]!="ground_beneath_a37_proxy":
            raise ValueError("Proxy layer identity changed")
        ground=fixed["depth_m"][:,3]>0
    totals={};rows=[]
    for raw in range(89,138):
        source=OBS/f"{raw:03d}.npz"
        with np.load(source,allow_pickle=False) as obs:
            a6,a37=mask(obs["visible"][0]),mask(obs["visible"][1])
            domains=dict(a6_visible=a6,a37_visible_keep=a37,
                         a6_measured_depth=mask(obs["depth_valid"][0]),
                         road_keep_original=mask(obs["road"]),
                         road_delete_proxy=mask(obs["road_edit"])&mask(ground[raw-89]),
                         protected_original=mask(obs["protect"])&~a6&~a37)
            if len(obs["depth_valid"])>1:
                domains["a37_measured_depth_keep"]=mask(obs["depth_valid"][1])
            roi=np.zeros((320,480),bool)
            if raw==137:
                roi[204:222,230:270]=True
            domains["raw137_fixed720_ROI"]=roi
            np.savez_compressed(dest/f"{raw:03d}.npz",**domains,
                                K=obs["K"],model_to_camera=obs["model_to_camera"],
                                model_time=obs["model_time"],model_from_local=obs["model_from_local"])
            counts={k:int(v.sum()) for k,v in domains.items()}
            for k,v in counts.items():
                totals[k]=totals.get(k,0)+v
            rows.append(dict(raw=raw,source_observation_sha256=sha(source),domain_sha256=sha(dest/f"{raw:03d}.npz"),counts=counts))
    expected={"a6_visible":39308,"a6_measured_depth":8930,"a37_visible_keep":536128,
              "road_keep_original":1075281,"road_delete_proxy":287368,"protected_original":6312448,
              "raw137_fixed720_ROI":720}
    for k,v in expected.items():
        if totals[k]!=v:
            raise ValueError(f"V44 denominator changed: {k} {totals[k]} != {v}")
    result=dict(schema="scene008.R0.V44_fixed_domains.v1",status="PASS_CPU_FROZEN_ORIGINAL_DOMAINS",raws=list(range(89,138)),
                image_hw=[320,480],domain_origin="Original observations; no current or candidate predictions",
                same_domains_in_keep_delete=True,source_reference_sha256=sha(FIXED),totals=totals,rows=rows,
                RGB_target_source="Original obs.rgb; road_delete_proxy only uses obs.road_edit_rgb",
                depth_target_source="Original obs.depth_m[0 or1], camera metres",
                proxy_is_hidden_GT=False,R1_estimator_access_allowed=False,GPU_used=False,optimizer_updates=0)
    (dest/"manifest.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=result["status"],totals=totals)),flush=True)


if __name__=="__main__":
    main()
