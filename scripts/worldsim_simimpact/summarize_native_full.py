"""完整预算、原生检测与有限控制的实测汇总，保留完整分母。"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
W=Path(__file__).resolve().parent;D=W/'evidence_native_detector';O=W/'evidence_native_full'
q=json.loads((O/'downstream_audit.json').read_text());v=json.loads((W/'native_full_validation_summary.json').read_text())
result={'replay':[],'closed_loop':q['closed_loop'],'sensor_validation':{},'detector':{},'feedforward':[]}
for c in sorted({r['condition'] for r in q['replay']}):
    rr=[r for r in q['replay'] if r['condition']==c]
    result['replay'].append({'condition':c,'frames':len(rr),'mean_ADE_delta_m':float(np.mean([r['ADE_delta_vs_real_m'] for r in rr])),
        'min_max_ADE_delta_m':[min(r['ADE_delta_vs_real_m'] for r in rr),max(r['ADE_delta_vs_real_m'] for r in rr)],
        'max_endpoint_change_m':max(r['execution_endpoint_change_vs_real_m'] for r in rr),
        'max_extra_braking_mps2':min(r['extra_minimum_acceleration_mps2'] for r in rr)})
result['sensor_validation']={'lidar_frames':len(v['lidars']),'camera_frames':len(v['cameras']),
    'mean_camera_PSNR_dB':float(np.mean([r['psnr_db'] for r in v['cameras']])),
    'pointwise_scope':'Official measured/imputed ray raster. Distance error uses GT-return positions, including predicted-drop positions; separate ray-drop counts retained. Mean per-frame metrics are not pooled quantiles.'}
for c in ['raw','median']:
    result['sensor_validation'][c]={k:float(np.mean([r[c][k] for r in v['lidars']])) for k in ['bias_m','mae_m','median_abs_m','p90_abs_m','rmse_m']}
for k in ['valid_rays','gt_returns','pred_returns','false_drop','false_return']:
    result['sensor_validation'][k]=sum(r[k] for r in v['lidars'])
motor=['car','truck','bus','trailer','construction_vehicle','motorcycle']
for name,file in [('real','summary.json'),('native','native_summary.json'),('constant13','constant13_summary.json'),('logged_pattern','logged_pattern_summary.json'),('local','local_summary.json')]:
    if not (D/file).exists():continue
    rows=json.loads((D/file).read_text());out={}
    for c in sorted({r.get('condition','real') for r in rows}):
        rr=[r for r in rows if r.get('condition','real')==c];gt=[g for r in rr for g in r['eligible_gt']]
        out[c]={'frames':len(rr),'eligible_all':len(gt),'unique_instances':len({g['instance'] for g in gt}),
                'eligible_motor_vehicles':sum(g['class'] in motor for g in gt),'unique_motor_vehicles':len({g['instance'] for g in gt if g['class'] in motor}),
                'eligible_by_class':dict(Counter(g['class'] for g in gt)),'thresholds':{}}
        for t in ['0.3','0.5']:
            out[c]['thresholds'][t]={}
            for m in ['center2m','bev_iou0p5']:
                matches=[x for r in rr for x in r['scores'][t][m]['matches']]
                out[c]['thresholds'][t][m]={'all':len(matches),'motor_vehicles':sum(x['class'] in motor for x in matches),'by_class':dict(Counter(x['class'] for x in matches))}
    result['detector'][name]=out
if (D/'ff_summary.json').exists():
    rows=json.loads((D/'ff_summary.json').read_text())
    for r in rows:
        result['feedforward'].append({k:r[k] for k in ['scene','condition','dataset_split']}|{'eligible':len(r['eligible_gt']),
            'scores':{t:{m:val['matched'] for m,val in r['scores'][t].items()} for t in ['0.3','0.5']}})
(O/'aggregate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result['sensor_validation'],indent=2))
if 'local' in result['detector']:
    print('LOCAL',json.dumps({c:v['thresholds']['0.5']['bev_iou0p5']['by_class'] for c,v in result['detector']['local'].items()}))
