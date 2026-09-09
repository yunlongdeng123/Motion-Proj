"""画上层聚合器适配接口，明确区分原提案与已实现但未训练的接口。"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--implemented-interface',action='store_true')
args=parser.parse_args()
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(14,5.4))
ax.set(xlim=(0,16),ylim=(0,6)); ax.axis('off')

def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=.10',
                              facecolor=color,edgecolor='#40464c',linewidth=1.6))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color='#40464c'))

box(.3,3.6,3,1.1,'Frozen prefix\nImage encoder + AA 0-17','#e9edf0')
box(4,3.6,2.7,1.1,'Global state 17\nAll 24 views','#dceaf7')
box(7.4,3.6,3.3,1.1,'Upper AA 18-23\nFrame / global + LoRA','#ffe1b8')
box(11.5,3.6,3.9,1.1,'Select actor views + DPT\n4 / 11 / 17 + fresh 23','#dcefdc')
box(7.4,1.0,3.3,1.1,'Actor 3D queries\nLocal geometry read','#dcefdc')
box(11.5,1.0,3.9,1.1,'Explicit surface\nHard physical readout','#fff2ce')
arrow((3.36,4.15),(3.94,4.15)); arrow((6.76,4.15),(7.34,4.15)); arrow((10.76,4.15),(11.44,4.15))
ax.plot([1.8,1.8,13.45],[4.76,5.15,5.15],color='#40464c',linewidth=1.6)
arrow((13.45,5.15),(13.45,4.76))
ax.text(7.6,5.29,'Reuse frozen DPT inputs 4 / 11 / 17',ha='center',fontsize=10)
ax.plot([13.45,13.45,9.05],[3.54,2.7,2.7],color='#40464c',linewidth=1.6)
arrow((9.05,2.7),(9.05,2.16)); arrow((10.76,1.55),(11.44,1.55))
title=('Upper-tail interface implemented: CPU check only; sensor training pending'
       if args.implemented_interface else 'Proposed upper-tail adaptation: example boundary, not implemented')
ax.text(8,5.85,title,ha='center',fontsize=14)
ax.text(3.4,2.4,'Layer indices are zero-based.\nGlobal context stays window-wide.\nAdapted outputs are recomputed\nafter every optimizer update.',ha='center',va='center',fontsize=10,color='#444444')
footer=('LoRA: qkv, rank 8, groups 18-23. The running Q-v2 model does not use this interface.'
        if args.implemented_interface else 'This diagram describes a separate adaptation factor; it does not change R12 or preselect Q-v2 surface parameterization.')
ax.text(8,.35,footer,ha='center',fontsize=9)
fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.03)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output.with_suffix('.png'),dpi=180,facecolor='white')
fig.savefig(args.output.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
