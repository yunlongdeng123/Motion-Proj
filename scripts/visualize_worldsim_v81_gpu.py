"""只使用真实输出制作对比图；候选、混杂与合成干预显式分开。"""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch

RUN=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01');ATLAS=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2');OUT=RUN/'analysis';FIG=OUT/'figures'
REG={r['roi_id']:r for r in map(json.loads,(ATLAS/'v81_roi_registry.jsonl').read_text().splitlines())};ROWS=list(map(json.loads,(RUN/'evaluation/metrics.jsonl').read_text().splitlines()))
FULL={(r['method'],r['roi_id']):r for r in ROWS if r['role']=='DISCOVERY'}
CONTROL={r['roi_id']:r for r in map(json.loads,(ATLAS/'simple_control_results.jsonl').read_text().splitlines())}
COLORS={'dvgt':'#147b83','vggt':'#ce6241','ref':'#151f35'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,name):fig.savefig(FIG/name,dpi=140,bbox_inches='tight');plt.close(fig)
def data(rid,method,key='full6'):
    roi=REG[rid];out=RUN/method/roi['window_id']/key;r=json.loads((out/'result.json').read_text());view=next(v for v in r['views'] if v['camera']==roi['camera']);A=np.array(view['original_to_network_pixel_center']);depth=np.load(out/(roi['camera']+'_depth_z_m.npy'));ref=np.load(ATLAS/'reference_geometry'/f'{rid}.npz')
    uv=ref['uv'];nu=np.c_[uv,np.ones(len(uv))]@A.T;sample=map_coordinates(depth,[nu[:,1],nu[:,0]],order=1,mode='constant',cval=np.nan)
    x0,y0,x1,y1=roi['box'];y,x=np.mgrid[y0:y1,x0:x1];grid=np.stack([x,y,np.ones_like(x)],-1)@A.T;dense=map_coordinates(depth,[grid[...,1],grid[...,0]],order=1,mode='nearest')
    return dense,sample,ref

ids=list(dict.fromkeys([r['roi_id'] for r in json.loads((ATLAS/'case_selection.json').read_text())]+json.loads((OUT/'exploratory_review_index.json').read_text())+['scene-0626_9a9c05fe_CAM_BACK_LEFT_13','scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','scene-0800_a4354e58_CAM_BACK_LEFT_13']))
for rid in ids:
    if (FIG/('case_'+rid+'.png')).exists():continue
    roi=REG[rid];crop=Image.open(ATLAS/'crops'/f'{rid}.jpg');dv,dp,ref=data(rid,'dvgt');vg,vp,_=data(rid,'vggt');uv=ref['uv'];xy=uv-np.array(roi['box'][:2]);gt=ref['depth_z']
    hi=float(np.quantile(np.r_[gt,dp[np.isfinite(dp)&(dp>0)],vp[np.isfinite(vp)&(vp>0)]],.98));hi=max(hi,float(gt.max()));lo=0
    fig,axs=plt.subplots(2,4,figsize=(14,6.5));axs[0,0].imshow(crop);axs[0,0].set_title(f'RGB | {roi["cohort"]} | {roi["semantic"].replace("_candidate","")}')
    axs[0,1].imshow(crop);sc=axs[0,1].scatter(xy[:,0],xy[:,1],c=gt,s=6,cmap='viridis',vmin=lo,vmax=hi);axs[0,1].set_title(f'Held-out support: {len(gt)} points');fig.colorbar(sc,ax=axs[0,1],fraction=.05,label='Depth z (m)')
    for ax,depth,method in [(axs[0,2],dv,'dvgt'),(axs[0,3],vg,'vggt')]:
        sc=ax.imshow(depth,cmap='viridis',vmin=lo,vmax=hi);m=FULL[(method,rid)];ax.set_title(f'{method.upper()} depth | MAE {m["mae_m"]:.2f} m');fig.colorbar(sc,ax=ax,fraction=.05)
    m=json.loads((RUN/'input_manifests'/f'{roi["window_id"]}.json').read_text());v=next(v for v in m['views'] if v['camera']==roi['camera']);whole=Image.open(v['image']);axs[1,0].imshow(whole);x0,y0,x1,y1=roi['box'];axs[1,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#f44c38',lw=2));axs[1,0].set_title('Original camera + ROI')
    vmax=max(.25,float(np.nanquantile(np.r_[abs(dp-gt),abs(vp-gt)],.98)))
    for ax,pred,method in [(axs[1,1],dp,'dvgt'),(axs[1,2],vp,'vggt')]:
        sc=ax.scatter(xy[:,0],xy[:,1],c=abs(pred-gt),s=6,cmap='magma',vmin=0,vmax=vmax);fig.colorbar(sc,ax=ax,fraction=.05,label='Absolute error (m)');row=FULL[(method,rid)];ax.set_title(f'{method.upper()} error | normal {row["normal_error_deg"]:.1f} deg')
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(ref['K']).T
    for dep,name,col in [(gt,'Held-out',COLORS['ref']),(dp,'DVGT',COLORS['dvgt']),(vp,'VGGT',COLORS['vggt'])]:
        points=rays*dep[:,None];axs[1,3].scatter(points[:,2],points[:,0],s=3,alpha=.5,label=name,color=col)
    axs[1,3].set(xlabel='Camera z (m)',ylabel='Camera x (m)',title='Depth-derived side view');axs[1,3].legend(fontsize=7,markerscale=2);axs[1,3].axis('equal')
    for ax in list(axs[0])+[axs[1,1],axs[1,2]]:ax.set_xlim(0,crop.width);ax.set_ylim(crop.height,0);ax.set_xticks([]);ax.set_yticks([])
    axs[1,0].set_xticks([]);axs[1,0].set_yticks([])
    ctrl=CONTROL.get(rid,{}).get('mae_m');ct=f'{ctrl:.3f} m' if ctrl is not None else 'unavailable'
    fig.suptitle(rid+' | DISCOVERY: reference association must be reviewed',fontsize=12)
    fig.text(.02,.015,'VGGT: dataset-camera-baseline scale (no LiDAR fit). Input-LiDAR plane control MAE: '+ct+'. Colors shared per case; extremes clipped.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'case_'+rid+'.png')

anchors=json.loads((OUT/'anchor_diagnostics.json').read_text());cases=['scene-0626_9a9c05fe_CAM_BACK_LEFT_13','scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','scene-0139_7e27d5c0_CAM_FRONT_LEFT_10'];variants=['sparse2','sparse3','full6','temporal18'];labels=['2 cameras','3 cameras','6 cameras','3 x 6 temporal']
fig,axs=plt.subplots(2,3,figsize=(13,6))
for col,rid in enumerate(cases):
    for method in ['dvgt','vggt']:
        rr={r['variant']:r for r in anchors if r['roi_id']==rid and r['method']==method and not r['texture_case']}
        for row,key in [(0,'mae_m'),(1,'normal_error_deg')]:axs[row,col].plot(range(4),[rr[v][key] for v in variants],'o-',color=COLORS[method],label=method.upper())
    axs[0,col].set_title(['Textured sign (synthetic control)','Low-texture wall candidate','Gray wall panel candidate'][col]);axs[0,col].set_ylabel('Depth MAE (m)');axs[1,col].set_ylabel('Normal error (degrees)')
    for ax in axs[:,col]:ax.set_xticks(range(4),labels,rotation=15);ax.grid(alpha=.15);ax.legend()
fig.suptitle('Same scene, retained target camera | frozen before model inference');fig.tight_layout(rect=(0,0,1,.94));save(fig,'same_scene_views.png')

rid=cases[0];fig,axs=plt.subplots(1,3,figsize=(13,3.7))
for method in ['dvgt','vggt']:
    for texture,style in [(False,'-'),(True,'--')]:
        rr={r['variant']:r for r in anchors if r['roi_id']==rid and r['method']==method and bool(r['texture_case'])==texture}
        for ax,key in zip(axs,['mae_m','normal_error_deg','log_shape_rmse_diagnostic']):ax.plot(range(4),[rr[v][key] for v in variants],marker='o',ls=style,color=COLORS[method],label=method.upper()+(' attenuated' if texture else ' original'));ax.set_xticks(range(4),labels,rotation=15);ax.grid(alpha=.15)
for ax,label in zip(axs,['Depth MAE (m)','Normal error (deg)','Scale-centered log-depth RMSE']):ax.set_ylabel(label)
axs[0].legend(fontsize=7);fig.suptitle('One sign: texture attenuation | SYNTHETIC, held-out-plane-assisted mask');fig.tight_layout(rect=(0,0,1,.93));save(fig,'texture_diagnostic.png')

paired=json.loads((OUT/'paired_view_effects.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for col,key in enumerate(['delta_absrel','delta_log_shape']):
    y=0
    for method in ['dvgt','vggt']:
        for variant in ['sparse3','sparse2']:
            for sem in ['ground_plane','non_ground_plane_candidate']:
                s=paired[f'{method}/{variant}/{sem}'][key];mid=s['median_of_log_medians'];ci=s['ci95_log_bootstrap'];label=f'{method.upper()} / {variant} / '+('ground' if sem=='ground_plane' else 'wall candidates')+f' (logs={s["n_logs"]})'
                axs[col].plot(ci,[y,y],color=COLORS[method],lw=2);axs[col].plot(mid,y,'o',color=COLORS[method]);axs[col].text(axs[col].get_xlim()[0],y,'')
                axs[col].set_yticks(list(range(y+1)),[t.get_text() for t in axs[col].get_yticklabels()][:y]+[label]);y+=1
    axs[col].axvline(0,color='#777',lw=.8);axs[col].invert_yaxis();axs[col].grid(axis='x',alpha=.15);axs[col].set_title(['Sparse minus full6: AbsRel','Sparse minus full6: local log-shape'][col])
fig.suptitle('Common-camera pairs | log bootstrap 95% CI | candidates, not matched 2x2');fig.tight_layout(rect=(0,0,1,.93));save(fig,'paired_view_effects.png')

fig,ax=plt.subplots(figsize=(12,3.7));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,c='#e8eef4'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=c,ec='#687b8d',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#40566d'})
box(.1,2.55,2.1,.85,'Frozen nuScenes RGB\n68 windows / 27 logs');box(3,3,2,.7,'GPU 0: DVGT-1','#d7eeed');box(3,1.95,2,.7,'GPU 1: raw VGGT','#fae6dc');arrow((2.2,3),(3,3.35));arrow((2.2,2.8),(3,2.3));box(5.7,2.45,2.2,1,'Native tensors +\nexplicit unit / pose\nexport to camera z');arrow((5,3.35),(5.7,3.1));arrow((5,2.3),(5.7,2.8));box(8.8,2.45,3,.95,'Depth + shape + paired views\nBad / good candidate atlas');arrow((7.9,2.9),(8.8,2.9))
box(.1,.45,2.1,.8,'Current input LiDAR\nplane / scale controls');box(3,.45,2.3,.8,'Disjoint neighbor LiDAR\nstatic / occlusion audit');box(6,.45,2.1,.8,'Held-out reference\n+ RGB spot-check');arrow((5.3,.85),(6,.85));arrow((2.2,.85),(2.7,.85));arrow((2.7,.85),(2.7,1.5));arrow((2.7,1.5),(9.4,1.5));arrow((8.1,.85),(8.5,.85));arrow((8.5,.85),(8.5,2.7));arrow((8.5,2.7),(8.8,2.7));arrow((9.4,1.5),(9.4,2.45));box(9.1,.35,2.65,.8,'V8.2: NO_GO\n0 complete natural 2x2 blocks','#fff0ce')
fig.suptitle('V8.1 GPU discovery: two independent models; no training');save(fig,'architecture.png')
print('figures',len(list(FIG.glob('*.png'))),'case_figures',len(ids))
