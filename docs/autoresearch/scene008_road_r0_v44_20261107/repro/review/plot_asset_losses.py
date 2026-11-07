"""Scientific native-image/loss-mask views and legal observation-state strata."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def count(x):return int(np.count_nonzero(x))
def box(mask,pad=8):
    y,x=np.where(mask);return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(480,int(x.max())+pad+1),min(320,int(y.max())+pad+1)]


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);args=p.parse_args();run=args.run
    out=run/"review/asset_ground_conflict";out.mkdir(parents=True,exist_ok=True)
    target=out/"observation_state_strata.json"
    if target.exists():raise FileExistsError("Preserve prior scientific views")
    records=[]
    for raw in (130,137):
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        folder=run/f"results/full49_corridor/{raw:03d}"
        b=load(folder/"baseline_delete_float.npz");a=load(folder/"calibrated_delete_float.npz")
        reviewed=obs["visible_reviewed"][0].astype(bool);candidate=obs["box_candidate"][0].astype(bool)
        domain=candidate&~obs["visible"][0].astype(bool);baseW=b["group_alpha_T"][0];newW=a["group_alpha_T"][0];loss=baseW-newW
        boundary=reviewed&~binary_erosion(reviewed);distance=distance_transform_edt(~boundary)
        selectors=dict(strong09_loss01=domain&(baseW>.9)&(loss>.01),strong09_loss10=domain&(baseW>.9)&(loss>.1),
            strong09_loss50=domain&(baseW>.9)&(loss>.5),moderate05_loss01=domain&(baseW>.5)&(loss>.01))
        flags=dict(a6_reviewed_margin=reviewed&~obs["visible"][0].astype(bool),a6_occluded=obs["occluded"][0].astype(bool),
            a6_free=obs["free"][0].astype(bool),a6_unknown=obs["unknown"][0].astype(bool),
            original_a37_visible=obs["visible"][1].astype(bool),original_a37_reviewed=obs["visible_reviewed"][1].astype(bool),
            original_observed_road=obs["road"].astype(bool),original_actor_label_a37=obs["actor_labels"]==5,
            other_original_actor=(obs["actor_labels"]>0)&(obs["actor_labels"]!=4)&(obs["actor_labels"]!=5))
        state=np.full((320,480),7,np.uint8)
        # Priority partition preserves a37 foreground rather than interpreting
        # its RGB as a hidden target-body or ground label.
        remaining=np.ones((320,480),bool)
        labels=[("original_a37_visible",flags["original_a37_visible"]),
            ("original_a37_reviewed_margin",flags["original_a37_reviewed"]&~flags["original_a37_visible"]),
            ("a6_measured_occluded_other",flags["a6_occluded"]),
            ("original_observed_road_and_a6_free",flags["original_observed_road"]&flags["a6_free"]),
            ("original_observed_road",flags["original_observed_road"]),
            ("a6_free_nonroad",flags["a6_free"]),("a6_unknown_other",flags["a6_unknown"])]
        for idx,(_,mask) in enumerate(labels):state[remaining&mask]=idx;remaining&=~mask
        names=[x[0] for x in labels]+["other_unresolved"]
        records.append(dict(raw=raw,selectors={name:dict(pixels=count(m),
            distance_to_reviewed_edge_px=dict(le1=count(m&(distance<=1)),gt1_le3=count(m&(distance>1)&(distance<=3)),gt3=count(m&(distance>3)),
                min=float(distance[m].min()) if m.any() else None,max=float(distance[m].max()) if m.any() else None),
            within_reviewed=count(m&reviewed),outside_reviewed=count(m&~reviewed),
            overlapping_observation_flags={key:count(m&flag) for key,flag in flags.items()},
            exclusive_original_RGB_state={label:count(m&(state==idx)) for idx,label in enumerate(names)}) for name,m in selectors.items()}))
        for name,m in selectors.items():Image.fromarray((m*255).astype(np.uint8)).save(out/f"raw{raw}_{name}_mask.png")
        np.savez_compressed(out/f"raw{raw}_states.npz",reviewed=reviewed,candidate=candidate,domain=domain,loss=loss,distance_to_reviewed_edge_px=distance,
            original_RGB_state=state,**selectors)
        images={label:np.asarray(Image.open(folder/file)) for label,file in (("Original RGB (a37 present)","observed.png"),
            ("Frozen V44 / delete a37","baseline_delete.png"),("Strip + wide field / delete a37","calibrated_delete.png"))}
        for file in ("observed.png","baseline_delete.png","calibrated_delete.png"):shutil.copyfile(folder/file,out/f"raw{raw}_{file}")
        crop=box(candidate|reviewed)
        def outlines(ax):
            ax.contour(reviewed.astype(float),levels=[.5],colors=["#00ff66"],linewidths=.8)
            ax.contour(candidate.astype(float),levels=[.5],colors=["#00d9ff"],linewidths=.8)
            ax.contour(obs["visible_reviewed"][1].astype(float),levels=[.5],colors=["#ff9e23"],linewidths=.8)
        fig,axs=plt.subplots(2,3,figsize=(15,8.6),constrained_layout=True)
        for ax,(label,img) in zip(axs[0],images.items()):
            ax.imshow(img,interpolation="nearest");ax.set_title(label);ax.set_xlim(-.5,479.5);ax.set_ylim(319.5,-.5)
            ax.add_patch(Rectangle((crop[0]-.5,crop[1]-.5),crop[2]-crop[0],crop[3]-crop[1],fill=False,edgecolor="#ee47ed",linewidth=1))
        axs[1,0].imshow(images["Original RGB (a37 present)"],interpolation="nearest");outlines(axs[1,0]);axs[1,0].set_title("Green: a6 reviewed; cyan: a6 box; orange: a37")
        im=axs[1,1].imshow(np.where(domain,np.maximum(loss,0),np.nan),vmin=0,vmax=1,cmap="magma",interpolation="nearest")
        outlines(axs[1,1]);axs[1,1].set_title("Asset alpha*T loss in unobserved a6 candidate")
        plt.colorbar(im,ax=axs[1,1],fraction=.04)
        axs[1,2].imshow(state,cmap="tab10",vmin=0,vmax=9,interpolation="nearest");outlines(axs[1,2]);axs[1,2].set_title("Original observation states (no hidden GT)")
        for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"raw{raw}: unchanged native renders and fixed observation/loss masks",fontsize=15)
        fig.savefig(out/f"raw{raw}_native_scientific_comparison.png",dpi=160);plt.close(fig)
        fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for ax,(label,img) in zip(axs[0],images.items()):ax.imshow(img,interpolation="nearest");outlines(ax);ax.set_title(label)
        for ax,data,title in ((axs[1,0],baseW,"Baseline actual asset alpha*T"),(axs[1,1],newW,"Strip actual asset alpha*T"),(axs[1,2],np.where(domain,np.maximum(loss,0),np.nan),"Positive asset alpha*T loss")):
            im=ax.imshow(data,vmin=0,vmax=1,cmap="magma",interpolation="nearest");outlines(ax);ax.set_title(title);plt.colorbar(im,ax=ax,fraction=.04)
            y,x=np.where(selectors["strong09_loss01"]);ax.scatter(x,y,s=6,facecolors="none",edgecolors="#5dff5d",linewidths=.5)
        for ax in axs.flat:ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"raw{raw}: a6 candidate crop; circles = baseline assetW>.9 and loss>.01",fontsize=14)
        fig.savefig(out/f"raw{raw}_a6_crop_and_loss.png",dpi=170);plt.close(fig)
    report=dict(schema="R0_A6_LOSS_NATIVE_SCIENTIFIC_VIEWS_AND_STATE_STRATA_V1",records=records,
        edge_distance="Euclidean pixel distance to original reviewed a6 inner boundary (reviewed XOR 1px erosion), including distance0 boundary pixels.",
        interpretation="Original a37 RGB remains an a37 observation, not a hidden a6/road target. Observed-road/free/unknown flags retain their source contract; overlaps and priority partition both reported.",
        no_hidden_GT=True,original_native_PNGs_copied_unchanged=True,GPU_used=False)
    target.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":main()
