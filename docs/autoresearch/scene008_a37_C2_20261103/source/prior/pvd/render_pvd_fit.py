import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);args=ap.parse_args()
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
original=np.load(root/'inputs/a6_prior_inputs.npz',allow_pickle=False)
real=original['observed_points_actor_native_m'];size=original['size_lwh_m']
data=np.load(args.candidate/'candidate_points_actor_native.npz',allow_pickle=False)
points=data['points_actor_native_m'];generated=data['generated_mask']
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,(i,j),title in zip(axes,[(0,2),(0,1),(1,2)],['Side: length / height','Top: length / width','Rear: width / height']):
    ax.scatter(real[::3,i],real[::3,j],s=.4,c='#e77722',alpha=.2,label='Original LiDAR observations')
    ax.scatter(points[generated,i],points[generated,j],s=2,c='#2878b5',alpha=.65,label='PVD completed points')
    ax.scatter(points[~generated,i],points[~generated,j],s=3,c='#3c9d40',alpha=.7,label='200 fixed original points')
    ax.plot([-size[i]/2,size[i]/2,size[i]/2,-size[i]/2,-size[i]/2],[-size[j]/2,-size[j]/2,size[j]/2,size[j]/2,-size[j]/2],c='#444444',ls='--',lw=.8)
    ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('meters');ax.set_ylabel('meters');ax.grid(alpha=.2)
axes[0].legend(markerscale=2,fontsize=7,loc='lower left')
fig.suptitle('a6: one frozen PVD car completion; seed0, 1000 DDPM steps, no target complete-shape GT',fontsize=11)
fig.savefig(args.candidate/'prior_observation_fit.png',dpi=180)
plt.close(fig)
print(args.candidate/'prior_observation_fit.png')
