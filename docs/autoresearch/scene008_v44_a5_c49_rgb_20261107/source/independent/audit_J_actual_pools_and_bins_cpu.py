"""Original-score real8F J pools and every registered bin; no solve/render."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get_bin(review,variant,r):
    row=next(v for v in review['records'] if v['variant']==variant and v['raw']==r['raw'] and v['state']==r['state'])
    field='primary128_seams' if r['kind']=='primary128' else 'world_seams'
    return next(v for v in row[field][r['edge']]['bins'] if v['tangent_bin']==r['tangent_bin'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--actual-review',type=Path,required=True);p.add_argument('--current-review',type=Path,required=True)
    p.add_argument('--candidate-parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    m,protocol,j,current=map(read,[a.manifest,a.protocol,a.actual_review,a.current_review]);op=a.manifest.parent/m['matrix_file']
    assert sha(op)==m['matrix_sha256']
    with np.load(op) as z:A=z['A'].copy();j0=z['baseline_J'].copy();c0=z['initial_parent_RGB'].copy();ids=z['eligible_parent_ids'].copy()
    with np.load(a.candidate_parent) as z:rgb=z['full_parent_RGB'].copy()
    predicted=j0+A@(rgb[ids].astype(np.float64)-c0.astype(np.float64));bins=[];actual=[]
    for i,r in enumerate(m['seam_records']):
        v=get_bin(j,'J',r);old=get_bin(current,'H',r);latest=get_bin(j,'baseline',r)
        actual.append(v['signed_residual_jump_rgb'])
        bins.append({k:r[k] for k in ['raw','state','kind','edge','tangent_bin','minus_pixels','plus_pixels']}|{
            'latest_score':latest['score'],'H2_C2000_score':old['score'],'J_actual_score':v['score'],
            'delta_vs_latest':v['score']-latest['score'],'delta_vs_H2_C2000':v['score']-old['score'],
            'actual_signed_residual_RGB':v['signed_residual_jump_rgb'],'predicted_signed_residual_RGB':predicted[i].tolist()})
    actual=np.asarray(actual,np.float64);assert actual.shape==(106,3)
    pools=[]
    for state in ['keep','delete']:
        for name,kind,edge,n in [('primary128','primary128','near_x',6),('world_near','world','near_x',24),('world_lateral','world','lateral_y',23)]:
            selected=[r for r in bins if r['state']==state and r['kind']==kind and r['edge']==edge];assert len(selected)==n
            ref=protocol['published_reference_metrics'][name][state];cur=ref['current_H2_C2000'];latest=ref['latest']
            target=min(cur,latest*.95,latest-1e-4) if name=='primary128' else min(cur,latest+1e-6)
            value=sum(v['J_actual_score'] for v in selected)/n
            pools.append(dict(name=name,state=state,bins=n,actual_original_score=value,original_target=target,tightened_reserve_target=target-1e-7,
                original_target_margin=target-value,tightened_reserve_margin=target-1e-7-value,original_pass=value<=target,
                tightened_reserve_pass_with_1e_9_tolerance=value<=target-1e-7+1e-9))
    summaries=[]
    for baseline in ['latest','H2_C2000']:
        delta='delta_vs_'+baseline
        for state in ['keep','delete']:
            selected=[r for r in bins if r['state']==state]
            strict=dict(improved=sum(v[delta]<0 for v in selected),regressed=sum(v[delta]>0 for v in selected),equal=sum(v[delta]==0 for v in selected))
            tol=dict(improved=sum(v[delta]<-1e-6 for v in selected),regressed=sum(v[delta]>1e-6 for v in selected),within_1e_6=sum(abs(v[delta])<=1e-6 for v in selected))
            worst=max(selected,key=lambda v:v[delta])
            summaries.append(dict(reference=baseline,state=state,bins=53,strict_sign_counts=strict,diagnostic_1e_6_counts=tol,
                worst_regression={k:worst[k] for k in ['raw','state','kind','edge','tangent_bin','minus_pixels','plus_pixels','latest_score','H2_C2000_score','J_actual_score',delta]}))
    fields=['raw','state','kind','edge','tangent_bin','minus_pixels','plus_pixels','latest_score','H2_C2000_score','J_actual_score','delta_vs_latest','delta_vs_H2_C2000']
    with (a.output/'all106_bin_directions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in bins)
    result={'schema':'J_REAL8F_ORIGINAL_POOLS_AND_TAILS_CPU_V1','status':'SIX_ORIGINAL_POOL_TARGETS_PASSED_OTHER_QUALITY_PENDING' if all(v['original_pass'] for v in pools) else 'ACTUAL_POOL_TARGET_FAILED',
        'actual_pools':pools,'per_bin_summary':summaries,'all106_bins':bins,'actual_vs_affine_prediction_max_abs':float(abs(actual-predicted).max()),
        'original4560_failed_gates':j['failed_gates'],'quality_accepted':False,'human_verdict':None,
        'new_solver_calls':0,'new_GPU_forwards':0,'new_optimizer_updates':0,
        'inputs':[dict(path=str(v),sha256=sha(v)) for v in [a.manifest,a.protocol,a.actual_review,a.current_review,a.candidate_parent,op,Path(__file__)]],
        'limits':['The1e-6 tail classification is diagnostic only; no per-bin no-regression constraint was added.',
          '53 metrics/state includes6 independent-primary plus47world metrics; some pixel domains overlap, not106 independent samples.',
          'Six pooled targets passing does not mean every bin improved or that visible seams/nonroad/whole image are accepted.',
          'The1e-7 LP reserve and original renderer thresholds are both shown; no post-output threshold change.']}
    q=a.output.parent/(a.output.name+'.json');q.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(q),'status':result['status'],'tails':summaries}))


if __name__=='__main__':main()
