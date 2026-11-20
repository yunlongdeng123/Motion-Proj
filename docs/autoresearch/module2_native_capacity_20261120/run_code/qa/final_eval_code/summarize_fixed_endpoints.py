"""Arithmetic presentation of fixed native endpoint rows; no metric recomputation."""
from pathlib import Path
import argparse,csv,json,hashlib

FIELDS=['RGB_MAE','HF_cosine','HF_energy_ratio','HF_relative_RMSE','HF_MAE','flat_pred_RMS','flat_ref_RMS','flat_n','edge_precision','edge_recall','edge_F1','Sobel_RMSE']
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def csvwrite(p,rows):
    if not rows:return
    with Path(p).open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(path):
    path=Path(path);d=json.loads(path.read_text('utf-8'));out=path.parent
    if d.get('complete')is not True:raise ValueError('only completed bound endpoint evaluation can be summarized')
    by={(r['arm'],r['raw'],r['source'],r['split'],r['scale']):r for r in d['rows']}
    seen=[r for r in d['rows']if r['split']=='seen'];held=[r for r in d['rows']if r['split']=='held']
    csvwrite(out/'seen_primary_table.csv',seen);csvwrite(out/'held_prior_conditioned_table.csv',held)
    pairs=[]
    for base in seen:
        if base['arm']!='G_RGB':continue
        key=('G_DETAIL',base['raw'],base['source'],'seen',base['scale'])
        if key not in by:continue
        detail=by[key]
        if any(base[k]!=detail[k]for k in ['score_n','context_n','core_n','boundary_n','flat_n','edge_reference_n']):raise ValueError('paired main-effect masks/reference differ')
        r=dict(raw=base['raw'],source=base['source'],split='seen',scale=base['scale'],core_n=base['core_n'],world_blocks=base['world_blocks'])
        for field in FIELDS:
            a,b=base[field],detail[field]
            r[field+'_G_RGB']=a;r[field+'_G_DETAIL']=b;r[field+'_DETAIL_minus_RGB']=b-a if a is not None and b is not None else None
        pairs.append(r)
    write(out/'method_main_effects.json',dict(schema='fixed_native_loss_main_effect.v1',rows=pairs,
        contrast='G_DETAIL minus G_RGB at fixed registered1200 endpoints and identical masks; signed differences only, no total score',
        source_roles='A visible real appearance; C/S pseudo teacher. Held values are separate prior-conditioned diagnostics, never used to select step/hyperparameters.',
        oracle='Boxci control is not promoted and numerical convergence is not guaranteed by endpoint completion',
        source_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),GPU_executed=False,native_calls=0,optimizer_updates=0))
    csvwrite(out/'method_main_effects.csv',pairs)
    write(out/'ideal_mechanism.json',dict(rows=d['ideal_rows'],role='open synthetic12cm control, not real teacher ranking',production_promoted=False))
    # This is a complete vector table, not a pooled score or a selected best case.
    lines=['# 固定原生端点评价','',
        '普通GS主实验按固定最终端点评价。A为实拍可见外观，C/S为伪教师；held有原F0/A200预暴露，不是盲测。', '',
        '```mermaid','flowchart LR','  P[25份公共原生packet] --> M[冻结source与seen/held mask]',
        '  M --> V[RGB / HF对应与能量 / flatHF / edge / Sobel]',
        '  P --> C[独立覆盖 / K / 静态贡献]',
        '  V --> T[逐帧逐源逐尺度表]','  C --> T','```','',
        '| 帧 | 源 | 方法 | σ | core n | RGB MAE | HF cos | HF能量比 | flat预测RMS | edge P / R |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    def f(x,n=4):return 'N/A'if x is None else format(x,f'.{n}f')
    for r in seen:
        lines.append(f"| {r['raw']} | {r['source']} | {r['arm']} | {r['scale']:g} | {r['core_n']} | {f(r['RGB_MAE'],5)} | {f(r['HF_cosine'])} | {f(r['HF_energy_ratio'],2)} | {f(r['flat_pred_RMS'],5)} | {f(r['edge_precision'],3)} / {f(r['edge_recall'],3)} |")
    lines += ['', '全三尺度、所有逐帧/源域病例均保留；未用均值替代尾部和缺失项。详见metrics.csv、method_main_effects.csv以及各detail_file完整曲线。',
              '', 'S的held只有一个world块，域级泛化N/A；core0严格N/A。完整pre-alpha覆盖、无交点与静态alphaT见results.json.coverage_rows；K/alpha见K_rows。',
              '', '理想12cm contrast/edge/半高宽单列ideal_mechanism.json；这不是车下GT、真实方法排行榜或表示极限定理。没有自动视觉PASS或生产晋升。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n','utf-8')
    return dict(rows=len(d['rows']),seen=len(seen),held=len(held),main_effect_pairs=len(pairs),packets=len(d['packets']),final_endpoint_set_complete=d['final_endpoint_set_complete'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args();print(json.dumps(summarize(a.results)))
