"""GPU 结果的 CPU evaluator；只统计真实存在的模型输出，未知保持未知。"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from motion_proj.worldsim_v81.geometry import depth_metrics,interaction_interval

def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',required=True);p.add_argument('--prediction-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();atlas=Path(a.atlas);results=[];registry=[json.loads(l) for l in (atlas/'v81_roi_registry.jsonl').read_text().splitlines()]
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
            results.append(row)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    strata={}
    for method,variant in sorted({(r['method'],r['variant']) for r in results}):
        for semantic in ['ground_plane','non_ground_plane_candidate']:
            subset=[r for r in results if r['method']==method and r['variant']==variant and r['semantic']==semantic and r['role']=='DISCOVERY' and r['coverage']>=.9]
            strata[f'{method}/{variant}/{semantic}']=interaction_interval(subset)
    summary={'rows':len(results),'status':'WAIT_MODEL_OUTPUTS' if not results else 'DISCOVERY_ONLY_PENDING_REFERENCE_SPOTCHECK_AND_MATCHING','log_interactions':strata,'promotion':'NO_GO','reason':'No automatic V8.2 promotion; requires confirmed reference, causal controls, independent logs and headroom'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
