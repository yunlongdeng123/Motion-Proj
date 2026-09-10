"""只在各自 FIT 的 BUILD 帧间定标，保留测量/局部平面代理的边界。"""
import json, time, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.data import load_build

start=time.time()
index=json.loads(Path('/root/autodl-tmp/data/worldsim_v74/p0_r1/index.json').read_text())
cases=index['cases'] if isinstance(index,dict) else index
summary={}
for dataset in ['nuscenes','av2']:
    residuals=[]; spacings=[]; objects=[]
    for row in cases:
        if row['dataset']!=dataset or row['role']!='FIT' or row['build_points']<3: continue
        folder=Path(row['build_file']).parent
        b=load_build(folder)
        pts=b['support_points_actor_m']
        # 无需 QUERY，跨 BUILD 帧的局部平面残差仅用于数值容差估计。
        frames=np.repeat(np.arange(len(b['frame_offsets'])-1),np.diff(b['frame_offsets']))
        pos=b['positive_actor'].astype(bool)&~b['ambiguous_owner'].astype(bool)
        pp=b['points_actor_m'][pos]; ff=frames[pos]
        origins=b['origins_actor_m'][pos]; dirs=b['directions_actor'][pos]
        valid_count=0
        for f in np.unique(ff):
            ref=pp[ff!=f]; ids=np.flatnonzero(ff==f)
            if len(ref)<6 or not len(ids): continue
            dist,nn=cKDTree(ref).query(pp[ids],k=min(16,len(ref)))
            local=ref[nn]; center=local.mean(1)
            delta=local-center[:,None]
            val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/local.shape[1])
            normal=vec[:,:,0]; cos=np.abs((normal*dirs[ids]).sum(1))
            orth=np.abs(((pp[ids]-center)*normal).sum(1))
            take=(dist[:,0]<=.3)&(cos>=.2)&(val[:,1]>1e-6)
            residuals.extend((orth[take]/cos[take]).tolist()); valid_count+=int(take.sum())
        dd=cKDTree(pts).query(pts,k=min(4,len(pts)))[0]
        positive=dd[:,1:][dd[:,1:]>1e-5]
        if len(positive): spacings.extend(positive.tolist())
        objects.append({'case_id':row['case_id'],'cross_frame_comparisons':valid_count,'build_points':len(pts)})
    rr=np.asarray(residuals); ss=np.asarray(spacings)
    quant={str(q):float(np.quantile(rr,q)) for q in [.5,.9,.95]} if len(rr) else {}
    epsilon=float(np.clip(quant.get('0.9',.2),.02,.2))
    spacing=float(np.clip(np.median(ss),.03,.3))
    summary[dataset]={'objects':len(objects),'comparisons':len(rr),'range_proxy_quantiles_m':quant,
                      'epsilon_obs_m':epsilon,'point_spacing_m':spacing,
                      'carrier_half_width_m':float(np.clip(4*spacing,.15,.8)),
                      'residual_above_epsilon_fraction':float(np.mean(rr>epsilon)) if len(rr) else None,
                      'cases':objects}
config={'task':'WS-V74-METHOD-TOURNAMENT-01','seed':7401,'confirmation_seed':7402,
        'data_root':'/root/autodl-tmp/data/worldsim_v74/p0_r1','face_budget':4096,'hit_band_m':.2,
        'dataset_parameters':{k:{n:v for n,v in d.items() if n in ['epsilon_obs_m','point_spacing_m','carrier_half_width_m']} for k,d in summary.items()},
        'wex':{'carriers':64,'grid':4,'domain_margin':.05,'max_outer_iterations':100,'beam_configurations':64,'qp_max_iter':10000,'qp_eps':1e-5,'wall_seconds':300,'construction_rounds':2},
        'rif':{'initial_grid':7,'refinement_rounds':3,'max_nodes':2500,'wall_seconds':300,'free_margin':.01},
        'dcs':{'initial_patches':32,'pricing_rounds':20,'proposals_per_round':8,'polygon_sides':8,'wall_seconds':300},
        'fit_only':True,'visual_input':False,
        'failure_ledger_refs':['V73-F02','V73-F03','V73-F04','V73-F05','V73-F09','V74-F01']}
out=ROOT/'docs/autoresearch/worldsim_v74/calibration';out.mkdir(parents=True,exist_ok=True)
(out/'fit_geometry.json').write_text(json.dumps({'task':'WS-V74-FIT-CALIBRATION-01','datasets':summary,'wall_seconds':time.time()-start,
    'limitations':'Cross-BUILD local-plane residual includes shape curvature, timing and box ownership error; not sensor noise ground truth. Values above the capped tolerance remain explicit conflicts; no rays or objects excluded from evaluation.'},indent=2)+'\n')
(ROOT/'configs/worldsim_v74/tournament.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps({k:{n:v for n,v in d.items() if n!='cases'} for k,d in summary.items()},indent=2),flush=True)
