"""实际短窗分布盘点；不同分母和未知对应保留，不作因果判定。"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from temporal_factory import T,read,dump
O=T/'r10'

def mask_stats(holes):
    fills=[];borders=[];areas=[]
    for h in holes:
        yy,xx=np.where(h);areas.append(float(h.mean()))
        fills.append(float(len(xx)/((np.ptp(xx)+1)*(np.ptp(yy)+1))) if len(xx) else None)
        borders.append(bool(h[0].any() or h[-1].any() or h[:,0].any() or h[:,-1].any()))
    return {'hole_canvas_fraction':float(np.mean(areas)),
            'hole_bbox_fill':float(np.mean([v for v in fills if v is not None])),
            'border_frame_fraction':float(np.mean(borders)),
            'area_max_min_ratio':max(areas)/min(areas) if min(areas)>0 else None}

def row(cid,scene,proc,holes):
    duration=proc['duration_s']
    return dict(case_id=cid,scene=scene,duration_s=duration,
        ego_path_speed_proxy_mps=proc['ego_camera_motion']['path_m']/duration,
        A_world_diameter_speed_proxy_mps=proc['A_world_motion']['diameter_m']/duration,
        hole_centroid_speed_proxy_pxps=proc['hole_centroid_path_px']/duration,
        static_A_moving_ego_image_change=proc['static_A_moving_ego_image_change'],
        sweep_B=proc.get('sweep_over_any_B',proc.get('GT_envelope_sweep_proxy')),
        sweep_semantics='SAM2 B pixels' if 'sweep_over_any_B' in proc else 'behind GT envelope only',
        **mask_stats(holes))

def main():
    old=read(T/'r9/temporal_audit.json');suites={}
    for name in ['r8','r10']:
        catalog=read(T/name/'dataset_catalog.json')['cases']
        lookup={r['case_id']:r for r in old['suites']['r8']['rows'] if r['split']=='train'}
        rows=[]
        for c in catalog:
            if c['split']!='train':continue
            holes=[np.asarray(Image.open(Path(c['folder'])/'model_hole'/f'{i:03}.png'))>0 for i in range(10)]
            rows.append(row(c['case_id'],c['receiver_scene'],c.get('process') or lookup[c['case_id']],holes))
        suites[name]=rows
    real=read(T/'r9/real_temporal_audit.json')['named_fixed_eval_windows'];rows=[]
    for p in real:
        holes=[np.asarray(Image.open(O/'evaluation'/p['eval_id']/'mask'/f'{i:05}.png'))>0 for i in range(10)]
        rows.append(row(p['eval_id'],p['scene'],p,holes))
    suites['real_fixed_DEV8']=rows
    keys=['duration_s','ego_path_speed_proxy_mps','A_world_diameter_speed_proxy_mps','hole_centroid_speed_proxy_pxps','hole_canvas_fraction','hole_bbox_fill','border_frame_fraction','area_max_min_ratio']
    summary={}
    for name,rs in suites.items():
        summary[name]={'cases':len(rs),'scenes':len({r['scene'] for r in rs}),
            'quantiles':{k:dict(zip(['min','q25','median','q75','max'],map(float,np.quantile([r[k] for r in rs if r[k] is not None],[0,.25,.5,.75,1])))) for k in keys},
            'mean_hole_canvas_fraction':float(np.mean([r['hole_canvas_fraction'] for r in rs])),
            'mean_border_frame_fraction':float(np.mean([r['border_frame_fraction'] for r in rs]))}
    dump(O/'process_distribution.json',{'summary':summary,'suites':suites,
        'limits':'world displacement/duration is a motion proxy, not instant speed or annotation accuracy. Real sweep only GT-envelope proxy; hidden texture evidence UNKNOWN. Comparison is same10 frames per case but source worlds/types differ; distribution differences are hypotheses, not proven causes.',
        'human_verdict':None})
    print(summary)

if __name__=='__main__':main()
