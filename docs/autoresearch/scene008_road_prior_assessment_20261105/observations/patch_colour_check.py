"""Check actual cached RGB only at the small patch's geometric witnesses."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def project(xyz, K, W):
    cam = xyz@W[:3,:3].T+W[:3,3]
    uvh = cam@K.T
    return uvh[:,:2]/np.maximum(cam[:,2:],1e-8)-.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--full", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026"))
    args = p.parse_args()
    out = args.run/"observations"
    dest = out/"patch137_colour_check.json"
    if dest.exists():
        raise FileExistsError("Preserve earlier colour check")
    with np.load(out/"patch137_observation_ledger.npz") as a:
        xyz, valid = a["local_xyz"], a["geometric_RGB_qualified"]
        members = {k:a["member_"+k] for k in ("fixed_roi","under_tail_band")}
    frames = np.flatnonzero(valid.any(1))
    observed = np.zeros((199,len(xyz),3),np.float32)
    for raw in frames:
        with np.load(args.full/"frames"/f"{raw:03d}.npz") as a:
            uv = project(xyz,a["K"],a["w2l"])
        image = np.asarray(Image.open(args.full/"rgb"/f"{raw:03d}.png"),np.float32)/255
        observed[raw] = np.stack([map_coordinates(image[:,:,ch],[uv[:,1],uv[:,0]],order=1,mode="constant",cval=0) for ch in range(3)],1)
    linear = np.where(observed<=.04045,observed/12.92,((observed+.055)/1.055)**2.4)
    luma = linear@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        q75 = np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
        upper = valid & (luma>=q75[None]-.025) & (luma<=q75[None]+.06)
        reference = np.nanmedian(np.where(upper[:,:,None],linear,np.nan),axis=0)
    reference = np.nan_to_num(reference,nan=0)
    agree = valid & (np.abs(linear-reference[None]).max(2)<.035)
    count = agree.sum(0)
    first = np.where(agree,np.arange(199)[:,None],999).min(0)
    last = np.where(agree,np.arange(199)[:,None],-1).max(0)
    measured = (count>=3) & (reference.min(1)>.012) & (reference.max(1)<.88)
    strict_span = measured & (last-first>=6)
    results = {k:dict(query_points=int(m.sum()),
        geometrically_witnessed=int((m&valid.any(0)).sum()),
        stable_upper_cluster_ge3=int((m&measured).sum()),
        stable_upper_cluster_ge3_and_span6=int((m&strict_span).sum()),
        qualified_but_insufficient_or_unstable=int((m&valid.any(0)&~measured).sum()),
        qualified_source_records=int(valid[:,m].sum()),
        agreeing_source_records=int(agree[:,m].sum())) for k,m in members.items()}
    np.savez_compressed(out/"patch137_colour_check.npz",local_xyz=xyz,source_rgb_srgb=observed,
        source_valid=valid,source_agree=agree,reference_linear=reference,agree_count=count,
        proxy_measured=measured,strict_span6=strict_span)
    result = dict(schema="SMALL_PATCH_RGB_PROXY_CHECK_V1",CPU_only=True,source_camera="FRONT",
        source_frames_decoded=frames.tolist(),cached_PNGs_decoded=len(frames),raw_records_decoded=0,
        reference_rule="Existing shadow199 rule, fixed before this check: q75 upper linear-luma cluster[-.025,+.06]; max-channel RGB agreement<.035; >=3 agreeing views; each linear channel >0.012 and <0.88. Span>=6 also reported separately.",
        groups=results,source_budget="Same real199 FRONT cache, no external/generative pseudo-labels.",
        caution="Stable high-radiance clustering is a clean-road proxy. It does not certify true unshadowed albedo or independent clean GT. Query heights are inherited measured-support interpolation.",
        optimizer_updates=0,original_assets_modified=False,GPU_used=False)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

    # Diagnostic figure of source pixels and a common world-space patch.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [137,162,163,165]
    fig,axes = plt.subplots(1,4,figsize=(12,3.2),dpi=180)
    for ax,raw in zip(axes,shown):
        image = np.asarray(Image.open(args.full/"rgb"/f"{raw:03d}.png"))
        with np.load(args.full/"frames"/f"{raw:03d}.npz") as a:
            uv = project(xyz,a["K"],a["w2l"])
        ax.imshow(image)
        ax.scatter(uv[:,0],uv[:,1],s=4,c="#f0bc45",alpha=.35,linewidths=0)
        good = valid[raw]
        ax.scatter(uv[good,0],uv[good,1],s=4,c="#1be591",alpha=.8,linewidths=0)
        ax.set_xlim(max(0,uv[:,0].min()-20),min(479,uv[:,0].max()+20))
        ax.set_ylim(min(319,uv[:,1].max()+20),max(0,uv[:,1].min()-20))
        ax.set_title(f"FRONT raw{raw}: {int(good.sum())}/{len(xyz)} witnesses",fontsize=9)
        ax.axis("off")
    fig.suptitle("Same inferred road-surface query patch across real source images",fontsize=12)
    fig.text(.5,.025,"Yellow: projected query points. Green: measured-triangle / guard / box-clear geometric RGB witnesses.\nIndependent audit ROI; the user's 570-pixel patch was not available. Stable RGB is a proxy, not clean GT.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.01,right=.99,top=.85,bottom=.2,wspace=.035)
    fig.savefig(out/"patch137_real_rgb_witnesses.png",bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
