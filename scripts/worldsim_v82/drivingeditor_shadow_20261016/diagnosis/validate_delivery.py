import pathlib,re,json
import numpy as np
from PIL import Image
p=pathlib.Path('outputs/shadow_diagnosis')
txt=(p/'diagnosis.html').read_text(encoding='utf-8')
refs=re.findall(r'(?:src|href)="([^"]+)"',txt)
missing=[r for r in refs if not (p/r).is_file()]
assert not missing,missing
a=json.loads((p/'gpu/GPU_diagnostic_receipt.json').read_text())
assert a['GPU_baseline_FP32_check']['raw_RGB_MAE']==0
f=json.loads((p/'focus/focus_GPU_receipt.json').read_text())
assert f['all_group_alpha_matches_baseline']
s=json.loads((p/'measured_summary.json').read_text())
assert all(r['freeze_both']['MAE_to_anchor']==0 for r in s['fixed_time'])
mask=np.asarray(Image.open(p/'focus/reviewed_artifact_footprint.png'))>0
assert mask.sum()==1287
with np.load(p/'focus/group_exact_contributions.npz') as q:
    assert np.array_equal(q['footprint'],mask)
    for i,name in enumerate(q['names']):
        assert abs(float(q['group_features'][i][mask].mean())-f['groups'][name]['mean_alpha_T'])<1e-7
assert '真实车影／假几何' in txt
for file in p.rglob('*.png'):
    try:
        with Image.open(file) as im: im.verify()
    except Exception as e:
        raise RuntimeError(str(file)) from e
for file in p.rglob('*.npz'):
    with np.load(file,allow_pickle=False) as q:
        for key in q.files:
            arr=q[key]
            if np.issubdtype(arr.dtype,np.number): assert np.isfinite(arr).all(),(file,key)
result={'links_checked':len(refs),'missing_links':missing,'zero_baseline_error':True,'seven_freeze_both_controls':True,'footprint_pixels':int(mask.sum()),'GPU_group_features_match_receipt':True,'all_local_PNG_and_NPZ_readable':True}
(p/'delivery_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
