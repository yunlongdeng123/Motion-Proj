"""CPU review triage; threshold statistics are not semantic recall or visual pass."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);records=[]
for raw in range(77,126):
    p=root/f'{raw:03d}';r=json.loads((p/'receipt.json').read_text())
    with np.load(p/'evidence.npz') as e:
        lum=(np.clip(e['white_rgb'],0,1)*np.array([.2126,.7152,.0722])[:,None,None]).sum(0)
        c=e['contact'];q=e['query'];a=e['alpha'];d=e['display_scope']
        rec=dict(raw=raw,final_rows=r['final_removed_rows'],new_rows=r['new_rows'],existing_rows=r['existing_rows'],
            inherited_outside_rows=r['inherited_outside_center_rows'],new_outside_rows=r['new_outside_center_rows'],
            contact_dark_pixels=int(((lum<.2)&c).sum()),contact_pixels=int(c.sum()),
            contact_min_luminance=float(lum[c].min()),contact_luminance_p05=float(np.quantile(lum[c],.05)),
            contact_alpha_mean=float(a[c].mean()),contact_alpha_max=float(a[c].max()),
            query_dark_pixels=int(((lum<.2)&q).sum()),display_dark_pixels=int(((lum<.2)&d).sum()),
            display_pixels=int(d.sum()),residual_outside_rows=r['final_after_summary']['residual_outside_rows'],
            evaluation_passes=len(r['history']),converged=r['converged'])
        records.append(rec)
def ext(key):
    low=min(records,key=lambda x:x[key]);high=max(records,key=lambda x:x[key])
    return dict(min=low[key],min_raw=low['raw'],max=high[key],max_raw=high['raw'])
result=dict(schema='r2-review-triage-v1',raws=[x['raw'] for x in records],records=records,
    ranges={k:ext(k) for k in ['final_rows','new_rows','existing_rows','inherited_outside_rows','new_outside_rows','contact_dark_pixels','contact_alpha_mean','query_dark_pixels','display_dark_pixels','residual_outside_rows','evaluation_passes']},
    lowest_contact_luminance_frames=sorted(records,key=lambda x:(x['contact_luminance_p05'],x['contact_min_luminance']))[:6],
    highest_contact_alpha_frames=sorted(records,key=lambda x:x['contact_alpha_mean'],reverse=True)[:6],
    caveat='Luminance<0.2 is only saved-white diagnostic triage, not semantic recall, clean removal or temporal stability proof. Existing outside IDs are inherited legacy; new outside selection remains separately audited.',
    GPU_calls=0,training_updates=0)
output=root.parent/'r2_quality_stats.json';output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
