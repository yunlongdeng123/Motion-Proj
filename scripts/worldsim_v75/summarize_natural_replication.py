"""汇总固定seed43复核的三个对比；不添加事后阈值或删评价时刻。"""
import json
from pathlib import Path
import numpy as np
from run_natural_rollouts import OUT as SOURCE
from repeat_natural_rollouts import OUT,VARIANTS

def main():
    assert not (OUT/'replication_result.json').exists()
    rows=[];initials={}
    for seed,directory in [(42,SOURCE),(43,OUT)]:
        review=json.loads((directory/'review_result.json').read_text());assert review['status']=='complete'
        cases={row['variant']:row for row in review['variants'] if row['variant'] in VARIANTS}
        assert len(cases)==4 and all(cases[n]['valid_matches']==5 for n in VARIANTS)
        errors={name:np.array([r['to_real_px'] for r in cases[name]['pairs']]) for name in VARIANTS}
        contrasts=[]
        for name in ['reference_lidar','ordinary_bbox','gt_clean']:
            delta=errors['dvgt_metric']-errors[name]
            contrasts.append({'comparator':name,'dvgt_minus_comparator_mean_px':float(delta.mean()),
                              'per_frame_difference_px':delta.tolist(),'comparator_better_noninitial_frames':int((delta[1:]>0).sum()),
                              'noninitial_denominator':4,'sign':'positive means comparator closer to real RGB'})
        rows.append({'seed':seed,'means_px':{name:float(errors[name].mean()) for name in VARIANTS},'cases':cases,'contrasts':contrasts})
        arrays={name:np.load(directory/name/'clean.npy',mmap_mode='r') for name in VARIANTS}
        initial_equal=all(np.array_equal(arrays['gt_clean'][0],arrays[name][0]) for name in VARIANTS)
        initials[str(seed)]=initial_equal
    result={'status':'complete','seeds':[42,43],'source_logs':1,'target_count':1,'variants':VARIANTS,
            'rows':rows,'initial_frame_equal_within_seed':initials,
            'no_additional_thresholds':True,'no_new_reconstruction_or_conditions':True,
            'missing_matches':0,'human_verdict':None,'failure_ledger_delta':'none',
            'boundary':'two seeds are stochastic replication, not cross-scene confirmation; extra target LiDAR changes information budget; f=60 partly occluded'}
    (OUT/'replication_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','rows':[{k:v for k,v in r.items() if k!='cases'} for r in rows]}),flush=True)

if __name__=='__main__':main()
