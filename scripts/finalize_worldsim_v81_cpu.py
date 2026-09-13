"""补齐模型前合同、简单强控制、干预、可浏览真实证据图。"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,argparse,collections,html
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import pyarrow as pa,pyarrow.parquet as pq
from motion_proj.worldsim_v81.geometry import transform,apply,project,plane_fit,depth_metrics,roi_mask

def render_card(row,run,path):
    rid=row['roi_id'];ref=np.load(run/'reference_geometry'/f'{rid}.npz');rgb=np.array(Image.open(run/'crops'/f'{rid}.jpg'))
    x0,y0,x1,y1=row['box'];uv=ref['uv']-[x0,y0];pu=ref['prompt_uv']-[x0,y0];gt=ref['depth_z'];xyz=ref['xyz']
    fig,axes=plt.subplots(1,5,figsize=(18,4));fig.subplots_adjust(left=.025,right=.985,bottom=.23,top=.72,wspace=.48)
    axes[0].imshow(rgb);axes[0].set_title('Original RGB ROI')
    axes[1].imshow(rgb,alpha=.3);sc=axes[1].scatter(uv[:,0],uv[:,1],c=gt,s=4,cmap='viridis');axes[1].set_title('Held-out depth z (m)');fig.colorbar(sc,ax=axes[1],fraction=.045)
    axes[2].imshow(rgb,alpha=.5);axes[2].scatter(uv[:,0],uv[:,1],s=2,c='#19a799',label='held-out');axes[2].scatter(pu[:,0],pu[:,1],s=5,c='#ec6b45',label='input');axes[2].legend(fontsize=7);axes[2].set_title('Disjoint LiDAR support')
    err=row.get('plane_control_error_values')
    if err is not None:
        sc=axes[3].scatter(uv[:,0],uv[:,1],c=err,s=5,cmap='magma',vmin=0,vmax=max(.2,float(np.quantile(err,.95))));fig.colorbar(sc,ax=axes[3],fraction=.045)
    else:axes[3].text(.5,.5,'No valid input-plane fit',ha='center',transform=axes[3].transAxes)
    axes[3].set_title('Input-LiDAR plane |error| (m)')
    axes[4].scatter(xyz[:,2],xyz[:,0],s=3,c=gt,cmap='viridis');axes[4].set_xlabel('Camera z (m)');axes[4].set_ylabel('x (m)',labelpad=1);axes[4].set_title('Reference side view');axes[4].axis('equal')
    for ax in axes[:4]:ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f"{row['scene']} | {row['camera']} | {row.get('cohort') or 'middle'} | {row['semantic']}",fontsize=12,y=.96)
    ov=row.get('overlap_audited',row['overlap_frustum_upper_bound'])
    fig.text(.025,.04,f"DISCOVERY candidate; G={row['gradient_energy']:.5f}, H={row['gray_entropy']:.2f}, overlap={ov:.2f}, held-out n={row['reference_points']}, input n={row['lidar_prompt_points']}\nCPU geometric screen only. SOTA predictions NOT RUN; no model failure or human verdict assigned.",fontsize=9)
    fig.savefig(path,dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    idx=json.loads((run/'index.json').read_text());rows=[json.loads(x) for x in (run/'v81_roi_registry.jsonl').read_text().splitlines()];root=Path(idx['root'])
    manifests={}
    for path in sorted((run/'input_manifests').glob('*.json')):
        m=json.loads(path.read_text())
        for v in m['views']+m['context_views']:
            s=v['sample_token'];r=idx['sample_data'][s]['LIDAR_TOP'];v['world_from_ego']=transform(idx['poses'][r['ego_pose_token']]).tolist()
            v['ego_pose_source']='LIDAR_TOP sample timestamp; camera pose retains own timestamp'
        path.write_text(json.dumps(m));manifests[m['window_id']]=m
    controls=[]
    for row in rows:
        if row['reference_status']!='GEOMETRIC_SCREEN_PASS':continue
        rid=row['roi_id'];d=np.load(run/'reference_geometry'/f'{rid}.npz');plane=plane_fit(d['prompt_xyz']);pred=np.full(len(d['uv']),np.nan)
        if plane:
            rays=np.c_[d['uv'],np.ones(len(d['uv']))]@np.linalg.inv(d['K']).T
            denom=rays@plane['normal'];np.divide(plane['center']@plane['normal'],denom,out=pred,where=np.abs(denom)>1e-5);pred[pred<=0]=np.nan
        metrics=depth_metrics(pred,d['depth_z']);row['plane_control_mae_m']=metrics['mae_m'];row['plane_control_coverage']=metrics['coverage']
        controls.append({'roi_id':rid,'log':row['log'],'control':'INPUT_LIDAR_RANSAC_PLANE','scope':'CPU anchor sanity, not MVS/DriveMVS/SOTA','input_points':len(d['prompt_xyz']),**metrics})
        row['plane_control_error_values']=np.nan_to_num(abs(pred-d['depth_z']),nan=0).tolist() if metrics['valid']==metrics['support'] else None
        # 对相邻相机，以其独立多扫描 reference 做可见支撑确认；不把 frustum 当实际 overlap。
        ownT=d['world_from_camera'];world=apply(d['xyz'],ownT);m=manifests[row['window_id']];fractions=[]
        for view in m['views']:
            if view['camera']==row['camera']:continue
            ou,oz,_=project(world,np.array(view['world_from_camera']),np.array(view['K']));valid=roi_mask(ou,oz,[0,0,1600,900]);supported=np.zeros(len(world),bool)
            others=[r for r in rows if r['window_id']==row['window_id'] and r['camera']==view['camera'] and r['reference_status']=='GEOMETRIC_SCREEN_PASS']
            if others:
                from scipy.spatial import cKDTree
                ods=[np.load(run/'reference_geometry'/f"{r['roi_id']}.npz") for r in others]
                ouv=np.concatenate([r['uv'] for r in ods]);odz=np.concatenate([r['depth_z'] for r in ods]);dist,nn=cKDTree(ouv).query(ou)
                supported=valid&(dist<12)&(abs(oz-odz[nn])<.5+.01*oz)
            fractions.append(float(supported.mean()))
        row['overlap_audited']=max(fractions,default=0.)
        # 上下界共同确定；中间区不强判。
        ov=row['overlap_frustum_upper_bound'];old=row['cohort'];ol='low' if ov<=.1 else 'high' if row['overlap_audited']>=.4 and (row['parallax_deg'] or 0)>=1 else 'middle'
        row['overlap_label']=ol;row['cohort']=None
        if row['texture_label']!='middle' and ol!='middle' and row['dark_fraction']<.2 and row['saturated_fraction']<.2:
            row['cohort']='C'+('1' if row['texture_label']=='low' else '0')+('1' if ol=='low' else '0')
        row['overlap_status']='low uses frustum upper bound <=0.1; high requires heldout-visible lower bound >=0.4'
    chosen=[]
    # 每格按纹理分数中位附近选，最多同日志一个，完全不读预测/控制误差。
    for c in ['C00','C01','C10','C11']:
        group=[r for r in rows if r['cohort']==c and r['semantic']=='non_ground_plane_candidate']
        if not group:group=[r for r in rows if r['cohort']==c]
        if group:
            mid=float(np.median([r['gradient_energy'] for r in group]));group.sort(key=lambda r:(abs(r['gradient_energy']-mid),r['roi_id']));used=set()
            for r in group:
                if r['log'] in used:continue
                chosen.append(r);used.add(r['log'])
                if len(used)>=2:break
    figdir=run/'figures';figdir.mkdir(exist_ok=True)
    for r in chosen:render_card(r,run,figdir/(r['roi_id']+'.png'))
    # 核心二维图只展示测量过的输入和reference，空格保留数据缺口。
    fig,axs=plt.subplots(2,2,figsize=(12,7));fig.subplots_adjust(top=.87,bottom=.12,hspace=.35,wspace=.2)
    for ax,c in zip(axs.flat,['C00','C01','C10','C11']):
        group=[r for r in chosen if r['cohort']==c];n=sum(r['cohort']==c for r in rows);logs=len({r['log'] for r in rows if r['cohort']==c})
        ax.set_title(f'{c} | n={n} ROIs / {logs} logs',fontsize=12)
        if group:
            r=group[0];rgb=Image.open(run/'crops'/f"{r['roi_id']}.jpg");ax.imshow(rgb);d=np.load(run/'reference_geometry'/f"{r['roi_id']}.npz");u=d['uv']-r['box'][:2];ax.scatter(u[:,0],u[:,1],s=2,c=d['depth_z'],cmap='viridis');ax.set_xlabel(r['scene']+' / '+r['camera'])
        else:ax.text(.5,.5,'No eligible natural case\nDo not fill by synthetic images',ha='center',va='center')
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Evidence scarcity grid: real RGB + held-out reference',fontsize=16)
    fig.text(.05,.035,'Top: high texture; bottom: low texture. Left: high overlap; right: low overlap.\nCPU candidate selection only; SOTA inference and matched semantic confirmation are pending.',fontsize=10)
    fig.savefig(figdir/'evidence_grid.png',dpi=150);plt.close(fig)
    # 图谱支持：三种语义混杂量直接可见，按 log 计分母。
    eligible=[r for r in rows if r['reference_status']=='GEOMETRIC_SCREEN_PASS']
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for sem,col in [('ground_plane','#d28132'),('non_ground_plane_candidate','#197fa4')]:
        rs=[r for r in eligible if r['semantic']==sem]
        axs[0].scatter([r['overlap_frustum_upper_bound'] for r in rs],[r['gradient_energy'] for r in rs],s=10,alpha=.65,c=col,label=sem)
    axs[0].set(xlabel='Frustum overlap upper bound',ylabel='Gradient energy');axs[0].legend(fontsize=7)
    axs[1].scatter([r['lidar_coverage'] for r in eligible],[r['reference_coverage'] for r in eligible],s=10,c='#197fa4');axs[1].set(xlabel='Input prompt coverage',ylabel='Held-out support coverage')
    axs[2].hist([r['plane_control_mae_m'] for r in eligible if r.get('plane_control_mae_m') is not None],bins=25,color='#197fa4');axs[2].set(xlabel='Input-LiDAR plane MAE (m)',ylabel='ROI count')
    fig.suptitle('CPU factor and simple-control audit — no SOTA results');fig.tight_layout();fig.savefig(figdir/'factor_controls.png',dpi=150);plt.close(fig)
    # 同场景干预：空间对应平面上的高频衰减；输出完整原分辨率图片和 mask。
    intervention=[]
    # 首轮视觉复核确认的高纹理平面招牌；栅栏案例保留为confound，不再用于因果干预。
    clean_texture_id='scene-0626_9a9c05fe_CAM_BACK_LEFT_13'
    high=[r for r in rows if r['roi_id']==clean_texture_id and r['reference_status']=='GEOMETRIC_SCREEN_PASS']
    for r in high:
        m=manifests[r['window_id']];ref=np.load(run/'reference_geometry'/f"{r['roi_id']}.npz");n=ref['normal'];center=ref['center'];K=ref['K'];T=ref['world_from_camera'];x0,y0,x1,y1=r['box']
        corners=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]],float)@np.linalg.inv(K).T
        denom=corners@n
        if np.any(abs(denom)<1e-5):continue
        world=apply(corners*((center@n)/denom)[:,None],T);overrides={};txdir=run/'interventions'/r['roi_id'];txdir.mkdir(exist_ok=True)
        for v in m['views']+m['context_views']:
            uv,z,_=project(world,np.array(v['world_from_camera']),np.array(v['K']))
            if np.any(z<=1) or np.any(abs(uv)>10000):continue
            rgb=np.array(Image.open(v['image']).convert('RGB'));mask=np.zeros(rgb.shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.round(uv).astype(np.int32),255)
            if not mask.any():continue
            flat=cv2.bilateralFilter(rgb,15,100,15);blend=cv2.GaussianBlur(mask.astype(np.float32)/255,(15,15),3)[...,None]
            edited=np.uint8(np.clip(rgb*(1-blend)+flat*blend,0,255));fn=txdir/(v['sample_token'][:8]+'_'+v['camera']+'.png');Image.fromarray(edited).save(fn);overrides[v['image']]=str(fn)
        tx={'roi_id':r['roi_id'],'role':'SYNTHETIC_DIAGNOSTIC','image_overrides':overrides,'method':'bilateral high-frequency attenuation on reference-plane projected polygon','reference_use':'diagnostic mask definition only','calibration_modified':False,'multi_view_consistent_mask':True,'claim':'not natural low texture'}
        (txdir/'intervention.json').write_text(json.dumps(tx,indent=2));intervention.append(tx)
        orig=Image.open(next(v['image'] for v in m['views'] if v['camera']==r['camera'])).crop(r['box']);name=next(v['image'] for v in m['views'] if v['camera']==r['camera']);edited=Image.open(overrides[name]).crop(r['box'])
        fig,axes=plt.subplots(1,4,figsize=(13,3.7))
        for ax,im,title in zip(axes,[orig,orig,edited,edited],['Full views / natural','Sparse views / same RGB','Full / attenuated texture','Sparse + attenuation']):ax.imshow(im);ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle('Same-scene factorial input control (synthetic diagnostic only)');fig.text(.06,.025,'Geometry and calibration fixed; predicted depth/error will be added after GPU inference.',fontsize=9);fig.savefig(figdir/'factor_escalation_inputs.png',dpi=150);plt.close(fig)
    # 架构图：模块、少量箭头和标签。
    fig,ax=plt.subplots(figsize=(13,3.8));ax.set_xlim(0,13);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(.2,2.3,2.2,.95,'RGB + calibration\nCurrent LiDAR'),(3,2.3,2.6,.95,'Static ROI + factors\nFrozen 2 x 2 cohorts'),(6.3,2.3,2.5,.95,'Official frozen models\nDVGT / VGGT / DGGT'),(9.7,2.3,2.9,.95,'Depth / surface audit\nBadcase + goodcase atlas'),(3,.3,2.6,.95,'Neighbor LiDAR\nDisjoint held-out reference'),(9.7,.3,2.9,.95,'Controls + new logs\nV8.2 decision')]
    for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#edf4f8',edgecolor='#315974'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
    for p1,p2 in [((2.45,2.8),(2.9,2.8)),((5.7,2.8),(6.2,2.8)),((8.9,2.8),(9.6,2.8)),((5.7,.8),(11.1,2.15)),((11.1,2.2),(11.1,1.35))]:ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=15,color='#315974'))
    ax.text(6.6,3.65,'CPU preparation completed  |  model outputs require GPU',fontsize=12,ha='center');fig.savefig(figdir/'architecture.png',dpi=160,bbox_inches='tight');fig.savefig(figdir/'architecture.svg',bbox_inches='tight');plt.close(fig)
    selection=[{'roi_id':r['roi_id'],'cohort':r['cohort'],'log':r['log'],'selection':'within-cohort median texture distance; <=1 case/log per cell; nonground first; no model/control error selection','figure':str(figdir/(r['roi_id']+'.png'))} for r in chosen]
    (run/'case_selection.json').write_text(json.dumps(selection,indent=2));(run/'simple_control_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in controls))
    for r in rows:r.pop('plane_control_error_values',None)
    (run/'v81_roi_registry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));pq.write_table(pa.Table.from_pylist(rows),run/'texture_overlap_lidar_prior_stats.parquet')
    queue=[]
    for method in ['dvgt','vggt']:
        for window in sorted(manifests):
            for variant in ['full6','sparse3','sparse2']:
                queue.append({'method':method,'window_id':window,'variant':variant,'manifest':str(run/'input_manifests'/f'{window}.json'),'status':'WAIT_GPU','batch_size':1})
    (run/'gpu_queue.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in queue))
    reserve=Path('/root/autodl-tmp/data/worldsim_v74_h2/av2_final_reserve_r1')
    (run/'independent_confirmation.json').write_text(json.dumps({'dataset':'AV2','role':'SEALED_CONFIRMATION','logs':sorted(p.name for p in reserve.iterdir() if p.is_dir()),'selection':'inherited unscored 10-log reserve; do not mine or select by v81 model outcomes','payload_quality_access_this_run':False,'status':'WAIT_DISCOVERY_PATTERN_AND_RAW_RGB_CONTRACT'},indent=2))
    summary=json.loads((run/'atlas_summary.json').read_text());summary.update(cohorts={c:sum(r['cohort']==c for r in rows) for c in ['C00','C10','C01','C11']},cohort_logs={c:len({r['log'] for r in rows if r['cohort']==c}) for c in ['C00','C10','C01','C11']},selected_cards=len(chosen),gpu_jobs=len(queue),simple_control_count=len(controls),natural_model_badcases=0)
    (run/'atlas_summary.json').write_text(json.dumps(summary,indent=2))
    content='<html><meta charset="utf-8"><title>WorldSim V8.1 CPU Atlas</title><style>body{max-width:1500px;margin:32px auto;font:16px system-ui;background:#f3f6fa;color:#183246}img{width:100%;background:white;border-radius:8px}article{margin:25px 0}a{color:#126c9e}</style><h1>WorldSim V8.1 · CPU Evidence Atlas</h1><p>真实输入 / 留出参考 / 简单 LiDAR 平面控制。SOTA 尚未推理，所有卡片是 DISCOVERY 候选；没有已确认的模型 failure。</p><p>图中可能含栅栏前景、墙角或地面混杂；几何初筛通过不等于可作科学主证据。人工verdict未填写。</p>'
    for fn in ['architecture.png','evidence_grid.png','factor_controls.png','factor_escalation_inputs.png']:
        if (figdir/fn).exists():content+=f'<article><img src="figures/{fn}"></article>'
    for r in chosen:content+=f'<article><img src="figures/{r["roi_id"]}.png"></article>'
    (run/'index.html').write_text(content+'</html>');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
