"""BEV illustration of all metadata-selected moving Actors; no counterfactual GT."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
data=json.loads((args.run/'preview.json').read_text()); entries=data['actors']
source=np.load(Path(data['source_scene_folder'])/data['source_frame']['file'])
origins=source['origin_world_m']; directions=source['directions_world']
fig,axes=plt.subplots(2,len(entries),figsize=(4*len(entries),8.6),squeeze=False)
for column,entry in enumerate(entries):
    pose=np.asarray(entry['world_from_reference_actor']); centers=np.asarray(entry['canonical_centers']).reshape(-1,3)
    arrays=np.load(args.run/entry['file']); length,width,height=entry['size_lwh_m']; row_stats=entry['statistics']
    for row,variant in enumerate(['before','after']):
        ax=axes[row,column]; depth=arrays[variant+'_depth_m']; owner=arrays[variant+'_owner']; valid=np.isfinite(depth)
        world=origins[valid]+directions[valid]*depth[valid,None]; local=(world-pose[:3,3])@pose[:3,:3]
        near=(np.abs(local[:,:2])<=6).all(1); current=local[near]; selected_owner=owner[valid][near]==entry['owner_id']
        ax.scatter(current[~selected_owner,1],current[~selected_owner,0],s=3,c='#9aa5b1',alpha=.7,rasterized=True)
        color='#237c92' if row==0 else '#c05640'
        ax.scatter(current[selected_owner,1],current[selected_owner,0],s=32,c=color,marker='x',linewidths=1.3,zorder=4,rasterized=True)
        offset=0 if row==0 else data['lateral_m']
        ax.scatter(centers[:,1]+offset,centers[:,0],s=2,c='#c6cbd2',alpha=.6,rasterized=True)
        ax.add_patch(Rectangle((-width/2+offset,-length/2),width,length,fill=False,ls='--',lw=1,color=color))
        ax.plot([offset,offset],[0,min(length/2,1.5)],color=color,lw=2)
        ax.scatter([offset],[0],marker='+',s=50,c=color)
        ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_aspect('equal'); ax.grid(alpha=.12)
        ax.set_xlabel('Actor-local lateral y (m)',fontsize=9)
        if column==0: ax.set_ylabel(('Original\n' if row==0 else 'Edited +2 m\n')+'Actor-local forward x (m)',fontsize=9)
        ax.tick_params(labelsize=8)
        if row==0:
            ax.set_title(f'{entry["owner"][:8]}  |  {entry["speed_mps"]:.2f} m/s\n{entry["build_points"]} build LiDAR points',fontsize=10,pad=8)
            note=f'Actor first intersections: {row_stats["before_actor_first"]}'
        else:
            note=(f'Actor first: {row_stats["after_actor_first"]}; new occlusions: {row_stats["introduced_occlusions"]}\n'
                  f'Released: {row_stats["released_rays"]}; no stored surface: {row_stats["released_to_unknown"]}')
        ax.text(.5,-.20,note,transform=ax.transAxes,ha='center',va='top',fontsize=8)
fig.suptitle('Fixed canonical surface + edited rigid trajectory: all four moving Actors in one old AV2 window',fontsize=14,y=.99)
fig.text(.5,.948,'Each column edits one Actor only; original background, other Actors and per-return-time query beams stay fixed',ha='center',fontsize=10)
legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#9aa5b1',label='Other first intersections'),
        Line2D([0],[0],marker='x',color='#237c92',ls='none',label='Selected Actor first intersections'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='#c6cbd2',label='Surface centers at reference time'),
        Line2D([0],[0],ls='--',color='#555555',label='Read-only size at reference time')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False,fontsize=9)
fig.text(.5,.028,'Fixed LiDAR-only r9 surface; first registered heldout scan. Light dots: surface centers at reference time; colored crosses: actual per-return-time first intersections.',ha='center',fontsize=9)
fig.text(.5,.008,'No edited ground truth or accuracy claim. Original observed-return beam subset only; unknown disoccluded background is not filled. BEV clipping is for display only.',ha='center',fontsize=9)
fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.15,hspace=.48,wspace=.20)
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+extension),dpi=170)
