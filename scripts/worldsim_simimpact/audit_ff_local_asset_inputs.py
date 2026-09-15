"""核对几何干预保留的强度、时间、历史和射线排列；不增加检测前向。"""
import json
from pathlib import Path
import numpy as np
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
rows=[]
for m in ['omega512','pi3x']:
    for v in ['six','twelve']:
        root=F/f'local_asset_inputs_r2/{m}_{v}';baseline=(F/f'scene-0004/{m}_fill_missing.npz') if v=='six' else root/'original_twelve.npz'
        p=np.load(baseline)['points'];q=np.load(root/'local_radial.npz')['points'];assert p.shape==q.shape
        assert np.array_equal(p[:,3:],q[:,3:]);history=p[:,4]>0;assert np.array_equal(p[history],q[history])
        distances=np.linalg.norm(p[:,:3]-q[:,:3],axis=1);changed=distances>1e-4
        rows.append({'method':m,'variant':v,'same_shape':True,'intensity_and_time_exactly_equal':True,'history_exactly_equal':True,
            'changed_current_point_positions':int(changed.sum()),'maximum_position_change_m':float(distances.max()),'baseline':str(baseline)})
        assert not changed[history].any()
(F/'local_asset_inputs_r2/input_isolation_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
