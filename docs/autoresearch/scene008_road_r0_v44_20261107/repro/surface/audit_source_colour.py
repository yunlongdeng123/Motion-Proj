"""Summarize observed source sRGB versus the uncalibrated clean-road proxy."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("--asset",type=Path,required=True);args=p.parse_args()
    output=args.asset.parent/"source_colour_audit.json"
    if output.exists():raise FileExistsError("Preserve previous colour audit")
    with np.load(args.asset,allow_pickle=False) as a:
        ref=a["reference_srgb"];src=a["source_rgb"];valid=a["source_valid"]
        active=a["active"];front=active&(a["local_xyz"][:,0]>19.75)
        rows=[]
        for lo,hi in ((0,198),(89,137),(138,198)):
            mask=valid[lo:hi+1]&front[None]
            delta=np.broadcast_to(ref,(hi-lo+1,*ref.shape))[mask]-src[lo:hi+1][mask]
            rows.append(dict(raw_range=[lo,hi],accepted_front_point_frame_records=int(mask.sum()),
                ref_minus_observed_mean_srgb=delta.mean(0).tolist() if len(delta) else None,
                ref_observed_MAE_srgb=float(np.abs(delta).mean()) if len(delta) else None,
                abs_channel_difference_p95=float(np.quantile(np.abs(delta),.95)) if len(delta) else None))
        encoded=np.where(ref<=.04045,ref/12.92,((ref+.055)/1.055)**2.4)
        check=float(np.abs(encoded[active]-a["reference_linear"][active]).max())
        report=dict(schema="R0_SRGB_ENCODING_AND_SOURCE_PROXY_AUDIT_V1",asset=str(args.asset),
            source_encoding="PNG integer channels/255 are encoded sRGB, bilinear sampled before fusion.",
            fusion="Exactly one sRGB->linear conversion, fixed q75 clean-radiance cluster; reference_srgb is exactly one inverse conversion.",
            rendering="MeasuredExtension passes reference_srgb directly as colors_precomp; no extra linear transform in road_kwargs.",
            accepted_source_srgb_minmax=[float(src[valid].min()),float(src[valid].max())],
            inverse_encoding_maxabs_linear=check,front_active_cells=int(front.sum()),comparisons=rows,
            interpretation="Differences measure real-view illumination/photometric variation against the clean-view proxy, not hidden clean GT error. Current49 build images are not held-out.",
            no_colour_fit=True,optimizer_updates=0,GPU_used=False)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":main()
