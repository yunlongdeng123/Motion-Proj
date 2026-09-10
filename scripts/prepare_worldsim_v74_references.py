"""公共参考资产：原样复用V73 r6/r7/R8，及4096面PCA有限surfel融合。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.common import write_json,save_mesh,farthest_ids
from motion_proj.worldsim_v74.c_dcs import geometry_context,patches_from_parameters,export_patches
from motion_proj.worldsim_v74.data import load_build
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text());base=Path('/root/autodl-tmp/runs/worldsim_v73')
cohort=json.loads((Path(cfg['data_root'])/'probe_cohort.json').read_text());methods=['surfel_pca_4096','V73_r6','V73_r7','V73_R8'];rows=[]
paths={'nuscenes':{'V73_r6':base/'WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6',
    'V73_r7':base/'WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7',
    'V73_R8':base/'WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8'},
    'av2':{m:base/'WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1'/sub for m,sub in [('V73_r6','lidar_r6'),('V73_r7','joint_r7'),('V73_R8','lidar_r8')]}}
write_json(args.output/'manifest.json',{'task':'WS-V74-COMMON-REFERENCES-01','methods':methods,'config':cfg,'cases':cohort['cases'],
    'boundary':'historical V73 checkpoints/surfaces unchanged; AV2 historical references use nuScenes-trained prior and are not same-dataset V74 controls. Surfel PCA uses current BUILD only.',
    'failure_ledger_refs':cfg['failure_ledger_refs']})
for case in cohort['cases']:
    b=load_build(Path(case['build_file']).parent)
    for method in methods:
        out=args.output/case['dataset']/case['case_id']/method;out.mkdir(parents=True)
        if method=='surfel_pca_4096':
            t=time.monotonic();context=geometry_context(b);ids=farthest_ids(context['points'],512)
            patch=patches_from_parameters(context,ids,np.zeros((len(ids),7)),cfg['dataset_parameters'][case['dataset']]['carrier_half_width_m'])
            v,f=export_patches(patch,np.arange(len(ids)));record={'status':'complete','method':method,'faces':len(f),'wall_seconds':time.monotonic()-t,'prior':'build local PCA, fixed FIT spacing, 512 finite 8-triangle patches'}
        else:
            source=paths[case['dataset']][method]/(case['owner']+'_surface.pt')
            if not source.exists():
                rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'folder':str(out),'record':{'status':'historical_surface_missing','expected':str(source)}});continue
            raw=torch.load(source,map_location='cpu',weights_only=False);v=raw['vertices_actor_m'].numpy();f=raw['faces'].numpy()
            record={'status':'complete','method':method,'faces':len(f),'wall_seconds':None,'source':str(source),'cost_boundary':'surface reuse; historical training/inference cost separately reported'}
        save_mesh(out/'final.npz',v,f);write_json(out/'reconstruction.json',record)
        rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'folder':str(out),'record':record})
write_json(args.output/'assets.json',rows);print('reference asset entries',len(rows),flush=True)
