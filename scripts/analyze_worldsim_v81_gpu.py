"""冻结候选的描述统计、公共相机配对与可复核的探索性选图。"""
import json,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01')
ATLAS=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2')
OUT=RUN/'analysis';OUT.mkdir(exist_ok=True);FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
REG={r['roi_id']:r for r in map(json.loads,(ATLAS/'v81_roi_registry.jsonl').read_text().splitlines())}
ROWS=list(map(json.loads,(RUN/'evaluation/metrics.jsonl').read_text().splitlines()))
CONTROL={r['roi_id']:r for r in map(json.loads,(ATLAS/'simple_control_results.jsonl').read_text().splitlines())}
def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2))
def med(v):return float(np.median(v)) if len(v) else None
def interval(rows,key):
    by=collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None:by[r['log']].append(r[key])
    values=np.array([np.median(v) for v in by.values()]);n=len(values)
    rng=np.random.default_rng(8101)
    return {'n_roi':len(rows),'n_logs':n,'median_of_log_medians':med(values),'ci95_log_bootstrap':np.quantile(np.median(rng.choice(values,(2000,n)),axis=1),[.025,.975]).tolist() if n>=3 else None}
def eligible(r):
    return r['cohort'] is not None and r['coverage']>=.9 and r['absrel'] is not None and r['visual_screen'] not in ['CONFOUND_EXCLUDE_MAIN','EXPOSURE_CONFOUND_CANDIDATE']

stats={};pairs=[];full={}
for r in ROWS:
    if r['role']=='DISCOVERY':full[(r['method'],r['roi_id'])]=r
for method in ['dvgt','vggt']:
    for semantic in ['ground_plane','non_ground_plane_candidate']:
        for cohort in ['C00','C10','C01','C11']:
            rr=[r for r in ROWS if r['method']==method and r['role']=='DISCOVERY' and r['semantic']==semantic and r['cohort']==cohort and eligible(r)]
            stats[f'{method}/{semantic}/{cohort}']={k:interval(rr,k) for k in ['absrel','mae_m','normal_error_deg','log_shape_rmse_diagnostic']}
    for variant in ['sparse3','sparse2']:
        for r in ROWS:
            if r['method']!=method or r['output_key']!=variant or not eligible(r):continue
            b=full.get((method,r['roi_id']))
            if b is None or not eligible(b):continue
            pairs.append({'method':method,'variant':variant,'roi_id':r['roi_id'],'log':r['log'],'cohort':r['cohort'],'semantic':r['semantic'],'delta_absrel':r['absrel']-b['absrel'],'delta_mae_m':r['mae_m']-b['mae_m'],'delta_normal_deg':r['normal_error_deg']-b['normal_error_deg'] if r['normal_error_deg'] is not None and b['normal_error_deg'] is not None else None,'delta_log_shape':r['log_shape_rmse_diagnostic']-b['log_shape_rmse_diagnostic'],'scale_ratio':r['metric_scale']/b['metric_scale']})
paired={}
for method in ['dvgt','vggt']:
    for variant in ['sparse3','sparse2']:
        for semantic in ['ground_plane','non_ground_plane_candidate']:
            pp=[r for r in pairs if r['method']==method and r['variant']==variant and r['semantic']==semantic]
            paired[f'{method}/{variant}/{semantic}']={k:interval(pp,k) for k in ['delta_absrel','delta_mae_m','delta_normal_deg','delta_log_shape']}
dump('cohort_descriptive.json',stats);dump('paired_view_effects.json',paired);dump('paired_rows.json',pairs)

selected=[]
for method in ['dvgt','vggt']:
    rr=[r for r in ROWS if r['method']==method and r['role']=='DISCOVERY' and r['cohort']=='C11' and r['semantic']=='non_ground_plane_candidate' and eligible(r)]
    for order,label in [(True,'bad_candidate'),(False,'good_candidate')]:
        logs=set()
        for r in sorted(rr,key=lambda r:(r['absrel'],r['roi_id']),reverse=order):
            if r['log'] in logs:continue
            logs.add(r['log']);selected.append({'roi_id':r['roi_id'],'selected_by_method':method,'selection':label,'absrel':r['absrel'],'mae_m':r['mae_m'],'log':r['log'],'stage':'POSTHOC_DISCOVERY_NOT_CONFIRMATION','visual_review':'PENDING'})
            if len(logs)==4:break
dump('exploratory_selection.json',selected)
unique=list(dict.fromkeys(r['roi_id'] for r in selected))
for page in range((len(unique)+11)//12):
    ids=unique[page*12:(page+1)*12];canvas=Image.new('RGB',(4*400,3*260),'#f1f4f7');d=ImageDraw.Draw(canvas)
    for i,rid in enumerate(ids):
        crop=Image.open(ATLAS/'crops'/f'{rid}.jpg').resize((384,216));x=(i%4)*400;y=(i//4)*260
        canvas.paste(crop,(x+8,y+36));d.text((x+8,y+3),rid,fill='black')
        labels=[s['selected_by_method']+':'+s['selection'] for s in selected if s['roi_id']==rid];d.text((x+8,y+18),' / '.join(labels),fill='black')
    canvas.save(FIG/f'exploratory_review_{page}.png')
dump('exploratory_review_index.json',unique)

anchor_ids=['scene-0626_9a9c05fe_CAM_BACK_LEFT_13','scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','scene-0139_7e27d5c0_CAM_FRONT_LEFT_10']
anchors=[r for r in ROWS if r['roi_id'] in anchor_ids and r['anchor_camera']==REG[r['roi_id']]['camera']]
dump('anchor_diagnostics.json',anchors)
runs=[json.loads(p.read_text()) for method in ['dvgt','vggt'] for p in (RUN/method).rglob('result.json')]
runtime={m:{'jobs':sum(r['method']==m for r in runs),'peak_allocated_gib':max(r['peak_gpu_allocated_gib'] for r in runs if r['method']==m),'forward_s_total':sum(r['elapsed_s'] for r in runs if r['method']==m)} for m in ['dvgt','vggt']}
scale=[r for r in runs if r['method']=='vggt' and r['role']=='DISCOVERY'];scale_summary={'full6_windows':len(scale),'scale_median':med([r['metric_scale'] for r in scale]),'scale_cv_median':med([r['camera_baseline_scale_cv'] for r in scale]),'scale_cv_range':np.quantile([r['camera_baseline_scale_cv'] for r in scale],[0,1]).tolist(),'warning':'sparse2 has one baseline ratio so CV=0 is not evidence of reliable scale'}
dump('runtime_summary.json',runtime);dump('scale_summary.json',scale_summary)
print(json.dumps({'rows':len(ROWS),'runtime':runtime,'scale':scale_summary,'exploratory_unique':len(unique),'paired':paired},indent=2))
