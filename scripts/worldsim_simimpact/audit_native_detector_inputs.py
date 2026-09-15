"""检查真实/原生合成检测输入的坐标、时间与反射率，先排除接口解释。"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
real=D/'real_scene0004_r1';native=D/'native_scans_scene0004_step030000_r1';out=[]
for i in range(8):
    g=np.load(real/f'input_{i:02d}.npz')['points'];gg=g[g[:,4]==0]
    tree=cKDTree(gg[:,:3]);row={'index':i,'real_current_points':len(gg),'real_all_points':len(g),
        'real_intensity_quantiles':np.quantile(gg[:,3],[0,.1,.5,.9,1]).tolist(),
        'real_age_groups':np.unique(g[:,4]).tolist()}
    for k in ['raw','median']:
        p=np.load(native/f'{k}_{i:02d}.npz')['points'];pp=p[p[:,4]==0]
        near=tree.query(pp[:,:3],workers=2)[0];reverse=cKDTree(pp[:,:3]).query(gg[:,:3],workers=2)[0]
        row[k]={'current_points':len(pp),'all_points':len(p),'age_groups':np.unique(p[:,4]).tolist(),
            'intensity_quantiles':np.quantile(pp[:,3],[0,.1,.5,.9,1]).tolist(),
            'xyz_quantiles':np.quantile(pp[:,:3],[.1,.5,.9],axis=0).tolist(),
            'pred_to_real_nn_m':np.quantile(near,[.1,.5,.9]).tolist(),
            'real_to_pred_nn_m':np.quantile(reverse,[.1,.5,.9]).tolist()}
    out.append(row)
(D/'input_alignment_audit.json').write_text(json.dumps(out,indent=2))
for r in out:print(r['index'],'intensity',r['real_intensity_quantiles'],r['raw']['intensity_quantiles'],'NN raw/median',r['raw']['pred_to_real_nn_m'],r['median']['pred_to_real_nn_m'],flush=True)
