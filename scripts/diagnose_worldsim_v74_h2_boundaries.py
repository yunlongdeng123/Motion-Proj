"""冻结资产的数值/小位姿误差诊断，不调参、不重训。"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.surface_state import Surface,square,concatenate
from motion_proj.worldsim_v74_h2.first_event_trace import trace,metrics
def load(path):
    with np.load(path) as a:return Surface(a['center'],a['rotation'],a['radius'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
    for folder in sorted(a.run.glob('*-*')):
        if not folder.is_dir():continue
        with np.load(folder/'query.npz') as z:base={k:z[k] for k in z.files}
        for magnitude in [.005,.01]:
            # 显式模拟标定误差：同一实测 range，观测原点/方向被扰动；不是正确标签的新射线增强。
            delta=np.array([magnitude,-magnitude,0]);rot=Rotation.from_rotvec([0,magnitude/10,0]).as_matrix()
            obs={**base,'origins_actor_m':base['origins_actor_m']+delta,'directions_actor':base['directions_actor']@rot.T}
            obs['points_actor_m']=obs['origins_actor_m']+obs['directions_actor']*obs['observed_first_range_m'][:,None]
            for method in ['INITIAL','TRUTH_ASSISTED','C6_TINY','C1','C3']:
                rows.append({'case':folder.name,'method':method,'position_error_axis_m':magnitude,'rotation_error_rad':magnitude/10,'metrics':metrics(load(folder/f'{method}.npz'),obs)})
    s=concatenate(square([0,0,0],1),square([0,0,1e-11],1))
    o=np.array([[0,0,-1],[1,0,-1],[1+1e-10,0,-1],[1-1e-10,0,-1],[-.5,0,-1e-9],[0,0,0],[0,0,0]],float)
    d=np.array([[0,0,1]]*4+[[1,0,1e-9],[1,0,0],[0,0,1]],float);d/=np.linalg.norm(d,axis=1,keepdims=True)
    obs={'origins_actor_m':o,'directions_actor':d};a64=trace(s,obs);a32=trace(s,obs,np.float32)
    exact={'names':['center_shared_fan','closed_outer_edge','just_outside','just_inside','grazing','coplanar','origin_on_first_plane'],
        'float64_events_per_ray':np.diff(a64['offsets']).tolist(),'float32_events_per_ray':np.diff(a32['offsets']).tolist(),
        'float64_first':[float(x) if np.isfinite(x) else None for x in a64['first']],
        'expected_boundary':'outside is empty; coplanar has no isolated event; t=0 excluded; distinct patches 1e-11 apart remain distinct',
        'scope':'float64 reference resolves this construction; no claim of arbitrary-precision exact arithmetic'}
    result={'task':'WS-V74-H2-NUMERICS-01','frozen_geometry_noise_sensitivity':rows,'analytic_edge_fixture':exact,
        'noise_policy':'same asset and same observation corruption for every control; no fitted tolerance changes'}
    (a.run/'boundary_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(exact))
if __name__=='__main__':main()
