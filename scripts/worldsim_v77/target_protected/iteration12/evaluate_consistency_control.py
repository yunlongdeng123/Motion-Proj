"""r36独立后验比较，Y只能在评价侧读取。"""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_observed_consistency import O,BASES,f


def main():
    assert f.read(O/'controller_state.json')['stage']=='complete_pending_paired_evaluation'
    from evaluate_qualified_state import main as quality
    from evaluate_actor_identity import main as identity
    quality(run_root=O,quality_root=f.T/'r32')
    identity('r36','r32')
    new={c['case_id']:c for c in f.read(O/'condition_evaluation.json')['cases']}
    results=[]
    for cid,base in BASES.items():
        before=next(c for c in f.read(f.T/base/'condition_evaluation.json')['cases'] if c['case_id']==cid)
        after=new[cid]
        for i in range(30):
            b=np.load(f.T/base/'state'/cid/f'{i:05}.npz');a=np.load(O/'state'/cid/f'{i:05}.npz')
            assert np.array_equal(b['N'],a['N']),'背景正证据不能被该actor点过滤改写'
            assert not (a['O']&a['N']).any() and np.array_equal(a['U'],~(a['O']|a['N']))
        cols=['rendered_O_precision_inside_H','hidden_B_coverage_by_O','unknown_H_fraction','N_H_fraction']
        results.append({'case_id':cid,'before_run':base,'before':{k:before[k] for k in cols},'after':{k:after[k] for k in cols},
            'precision_not_lower':after[cols[0]]>=before[cols[0]],
            'hidden_B_coverage_retention':after[cols[1]]/max(1e-9,before[cols[1]]),
            'N_unchanged_all_30_frames':True})
    old_actors=f.read(f.T/'r34/actor_identity_evaluation.json')['cases'][0]['actors']
    new_actors=next(c for c in f.read(O/'actor_identity_evaluation.json')['cases'] if c['case_id']=='Q046')['actors']
    edges=[]
    for old,a in zip(old_actors,new_actors):
        assert old['instance']==a['instance']
        edges.append({'instance':a['instance'],'slot':a['slot'],'primary':a['primary_reveal'],
            'before_precision':old['O_H_identity_precision'],'after_precision':a['O_H_identity_precision'],
            'before_coverage':old['hidden_B_coverage'],'after_coverage':a['hidden_B_coverage']})
    gate=all(r['precision_not_lower'] and r['hidden_B_coverage_retention']>=.90 for r in results)
    gate=gate and all(a['after_precision']>a['before_precision'] for a in edges if not a['primary'])
    result={'cases':results,'Q046_per_actor':edges,'predeclared_numeric_gate_pass':gate,
            'no_model_inference_or_training':True,'human_verdict':None,
            'decision':'pending independent condition QA, not model benefit' if gate else 'do_not_promote; keep originals; no threshold grid',
            'failure_ledger_refs':['V77-F02']}
    f.dump(O/'paired_comparison.json',result);state=f.read(O/'controller_state.json')
    state['stage']='complete_pending_independent_QA' if gate else 'complete_negative_control_do_not_promote'
    f.dump(O/'controller_state.json',state)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
