"""GPU 结果的 CPU evaluator；只统计真实存在的模型输出，未知保持未知。"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from motion_proj.worldsim_v81.geometry import depth_metrics,interaction_interval,plane_fit

def matched_rows(rows,blocks):
    """仅纳入冻结四格中均有有效输出的完整block；拒绝混杂与不完整block。"""
    eligible={r['roi_id']:r for r in rows if r['coverage']>=.9 and r['error'] is not None and r.get('visual_screen')!='CONFOUND_EXCLUDE_MAIN'}
    selected=[]
    for block in blocks:
        ids=[block['cases'][c] for c in ['C00','C10','C01','C11']]
        if all(rid in eligible for rid in ids):selected.extend(eligible[rid] for rid in ids)
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',required=True);p.add_argument('--prediction-root',required=True);p.add_argument('--out',required=True);p.add_argument('--no-figures',action='store_true')
    a=p.parse_args();atlas=Path(a.atlas);results=[];registry=[json.loads(l) for l in (atlas/'v81_roi_registry.jsonl').read_text().splitlines()]
    selected={r['roi_id'] for r in json.loads((atlas/'case_selection.json').read_text())};panels={}
    blocks=json.loads((atlas/'matched_cohorts.json').read_text())
    for rp in sorted(Path(a.prediction_root).rglob('result.json')):
        r=json.loads(rp.read_text())
        if r.get('status')!='DONE':continue
        views={v['camera']:v for v in r['views']}
        for roi in registry:
            if roi['window_id']!=r['window'] or roi['reference_status']!='GEOMETRIC_SCREEN_PASS':continue
            if roi['camera'] not in views:continue  # 不把未输入/未预测的 camera 当模型 MISS。
            pred=np.load(rp.parent/(roi['camera']+'_depth_z_m.npy'));ref=np.load(atlas/'reference_geometry'/f"{roi['roi_id']}.npz")
            A=np.array(views[roi['camera']]['original_to_network_pixel_center']);uv=np.c_[ref['uv'],np.ones(len(ref['uv']))]@A.T
            sampled=map_coordinates(pred,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=np.nan)
            metrics=depth_metrics(sampled,ref['depth_z']);row={'roi_id':roi['roi_id'],'log':roi['log'],'scene':roi['scene'],'cohort':roi['cohort'],'semantic':roi['semantic'],'method':r['method'],'variant':r['variant'],'error':metrics['mae_m'],'role':r['role'],'reference_status':roi['reference_status'],**metrics}
            row.update(visual_screen=roi.get('visual_screen','UNREVIEWED'),scientific_acceptance=roi.get('scientific_acceptance','NOT_ESTABLISHED'),window=r['window'],output_key=rp.parent.name,result_path=str(rp),anchor_camera=r.get('anchor_camera'),texture_case=r.get('texture_case'),metric_scale=r['metric_scale'],scale_cv=r['camera_baseline_scale_cv'])
            valid=np.isfinite(sampled)&(sampled>0);rays=np.c_[ref['uv'],np.ones(len(ref['uv']))]@np.linalg.inv(ref['K']).T
            points=rays[valid]*sampled[valid,None];fit=plane_fit(points)
            row['surface_space']='depth unprojected with dataset intrinsics; not native Gaussian mesh'
            row['point_to_plane_m']=float(np.mean(abs((points-ref['center'])@ref['normal']))) if len(points) else None
            row['normal_error_deg']=float(np.degrees(np.arccos(np.clip(abs(fit['normal']@ref['normal']),0,1)))) if fit else None
            row['plane_bending_p95_m']=float(np.quantile(fit['residual'],.95)) if fit else None
            if valid.sum():
                # 仅用于区分局部形状误差与整体尺度偏移；不改变任何主深度指标。
                ld=np.log(sampled[valid]/ref['depth_z'][valid]);row['log_shape_rmse_diagnostic']=float(np.sqrt(np.mean((ld-ld.mean())**2)))
                row['median_depth_ratio_diagnostic']=float(np.median(sampled[valid]/ref['depth_z'][valid]))
            else:row.update(log_shape_rmse_diagnostic=None,median_depth_ratio_diagnostic=None)
            row['render_metrics']=None
            results.append(row)
            if roi['roi_id'] in selected:
                panels.setdefault(roi['roi_id'],[]).append((row,roi,ref['uv'].copy(),ref['depth_z'].copy(),sampled.copy(),points.copy()))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    strata={}
    for method,variant in sorted({(r['method'],r['variant']) for r in results}):
        for semantic in ['ground_plane','non_ground_plane_candidate']:
            subset=[r for r in results if r['method']==method and r['variant']==variant and r['semantic']==semantic and r['role']=='DISCOVERY' and r['coverage']>=.9]
            strata[f'{method}/{variant}/{semantic}']=interaction_interval(matched_rows(subset,blocks))
    summary={'rows':len(results),'status':'WAIT_MODEL_OUTPUTS' if not results else 'DISCOVERY_ONLY_PENDING_REFERENCE_SPOTCHECK_AND_MATCHING','frozen_matched_blocks':len(blocks),'log_interactions':strata,'promotion':'NO_GO','reason':'No automatic V8.2 promotion; requires confirmed reference, causal controls, independent logs and headroom'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    # 实际输出存在才生成模型对比图，选择规则沿用冻结case_selection。
    if panels and not a.no_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        figures=out/'figures';figures.mkdir(exist_ok=True)
        for rid,items in panels.items():
            fig,axs=plt.subplots(len(items),4,figsize=(14,3*len(items)),squeeze=False)
            for axes,(r,roi,uv,gt,pred,points) in zip(axs,items):
                crop=Image.open(atlas/'crops'/f'{rid}.jpg');xy=uv-np.array(roi['box'][:2]);lo=max(0,float(np.min(gt))-2);hi=float(np.max(gt))+2
                axes[0].imshow(crop);axes[0].scatter(xy[:,0],xy[:,1],c=gt,s=3,cmap='viridis',vmin=lo,vmax=hi);axes[0].set_title('RGB + held-out depth')
                sc=axes[1].scatter(xy[:,0],xy[:,1],c=pred,s=4,cmap='viridis',vmin=lo,vmax=hi);fig.colorbar(sc,ax=axes[1],fraction=.04);axes[1].set_title(f"{r['method']} / {r['variant']} depth (m)")
                sc=axes[2].scatter(xy[:,0],xy[:,1],c=abs(pred-gt),s=4,cmap='magma',vmin=0,vmax=2);fig.colorbar(sc,ax=axes[2],fraction=.04);axes[2].set_title(f"|error|; coverage={r['coverage']:.2f}")
                axes[3].scatter(points[:,2],points[:,0],s=3);axes[3].set(xlabel='Camera z (m)',ylabel='Camera x (m)',title='Depth-derived surface');axes[3].axis('equal')
                for ax in axes[:3]:ax.set_xlim(0,crop.width);ax.set_ylim(crop.height,0);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(rid+' | DISCOVERY: candidate, not a validated failure');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(figures/f'{rid}.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
