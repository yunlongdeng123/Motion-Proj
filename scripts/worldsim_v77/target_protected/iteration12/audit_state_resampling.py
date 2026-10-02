"""检查稀疏条件在UNet空间缩放时是否真正到达Adapter；无Y接口。"""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).parent))
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')


def main():
    torch.set_num_threads(2);rows=[]
    for cid in ['L001','L007','L009']:
        for frame in range(30):
            state=dict(np.load(T/'r22/state'/cid/f'{frame:05}.npz'))
            h=torch.from_numpy(np.asarray(Image.open(T/'r16/synthetic'/cid/'model_hole'/f'{frame:03}.png'))>0)[None,None].float()
            for size in [(40,72),(20,36),(10,18),(5,9)]:
                active=F.adaptive_max_pool2d(h,size).bool()
                row={'case':cid,'frame':frame,'size':list(size)}
                for role in ['O','N']:
                    x=torch.from_numpy(state[role].astype('float32'))[None,None]
                    any_support=F.adaptive_max_pool2d(x,size)>0
                    old=F.interpolate(x,size,mode='nearest')>0
                    fraction=F.adaptive_avg_pool2d(x,size)
                    row[role]={'cells_with_observed_support':int((any_support&active).sum()),
                               'nearest_retained_cells':int((old&active).sum()),
                               'area_fraction_retained_cells':int(((fraction>0)&active).sum()),
                               'nearest_missing_supported_cells':int((any_support&~old&active).sum())}
                    assert torch.equal(fraction>0,any_support)
                rows.append(row)
    summary=[]
    for cid in ['L001','L007','L009']:
        for role in ['O','N']:
            rr=[r[role] for r in rows if r['case']==cid]
            summary.append({'case':cid,'role':role,**{k:sum(r[k] for r in rr) for k in rr[0]}})
    result={'phase':'engineering_condition_resampling_audit','source':'same legal r22 projected states; no hidden Y',
            'frames_per_case':30,'network_sizes':[[40,72],[20,36],[10,18],[5,9]],
            'interpretation':'counts are frame x scale feature cells, not independent examples or recovered area',
            'summary':summary,'rows':rows,'training_steps':0,'human_verdict':None}
    p=T/'r24/condition_resampling_audit.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
