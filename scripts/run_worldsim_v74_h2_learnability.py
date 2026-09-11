"""一次 CPU 最小连续几何学习能力实验；不声称完整 E1/P2 或新颖性通过。"""
import os,sys,json,time,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_worldsim_v74_h2_mechanisms import make_scene,write,save_obs
from motion_proj.worldsim_v74_h2.surface_dynamics import SurfaceDynamics,features,apply_continuous
from motion_proj.worldsim_v74_h2.first_event_trace import metrics
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(7411);rng=np.random.default_rng(7421);start=time.monotonic();tasks=[]
    for family in ['multilayer','shared_support','missing_support','grazing_thin']:
        for i in range(3):
            truth,_,build,query=make_scene(family,i,7421)
            initial=truth.copy();delta=rng.uniform(-.25,.25,truth.center.shape);rot=rng.uniform(-.12,.12,truth.center.shape);rad=rng.uniform(-.45,.45,truth.radius.shape)
            initial.center+=delta;initial.rotation=Rotation.from_rotvec(rot).as_matrix()@truth.rotation;initial.radius*=np.exp(rad)
            target=np.c_[-delta,-rot,-rad].astype(np.float32)
            folder=a.output/(family+f'-{i:02d}');folder.mkdir();initial.save(folder/'initial.npz');truth.save(folder/'target.npz');save_obs(folder/'build.npz',build);save_obs(folder/'query.npz',query)
            tasks.append((folder,initial,build,query,features(initial,build),torch.from_numpy(target)))
    config={'task':'WS-V74-H2-LEARNABILITY-01','role':'synthetic FIT memorization only','seed':7411,'scene_seed':7421,
        'tasks':len(tasks),'width':32,'epochs':400,'optimizer':'Adam lr=0.003','gpu_used':False,
        'features':'raw full plane depth for order; invertible asinh depth/residual/boundary/uv features, no clipping',
        'scope':'one continuous joint geometry step; oracle matched patches; topology policy and 8-step rollout remain untested',
        'A_and_C2_information':'identical full plane/intersection feature set, coordinates and depths; C2 may infer ordering',
        'failure_ledger_refs':['V74-F04','V74-F09'],'failure_ledger_delta':'V74-H2-F05: raw depth ordering and invertible feature scaling replace clipping; r1 retained'}
    write(a.output/'manifest.json',config);rows=[]
    for name,ordered in [('A',True),('C2',False)]:
        torch.manual_seed(7411);model=SurfaceDynamics(32,ordered);opt=torch.optim.Adam(model.parameters(),lr=.003);curve=[]
        for epoch in range(400):
            opt.zero_grad();loss=torch.stack([((model(*ft)-target)**2).mean() for _,_,_,_,ft,target in tasks]).mean();loss.backward();opt.step()
            if epoch%20==0 or epoch==399:curve.append({'epoch':epoch,'mse':float(loss.detach())})
        model.eval();torch.save({'state_dict':model.state_dict(),'width':32,'ordered':ordered,'config':config},a.output/f'{name}.pt')
        errors=[];quality=[]
        with torch.no_grad():
            for folder,initial,build,query,ft,target in tasks:
                pred=model(*ft).numpy();errors.append(float(np.sqrt(np.mean((pred-target.numpy())**2))))
                asset=apply_continuous(initial,pred);asset.save(folder/f'{name}.npz')
                quality.append({'case':folder.name,'initial_query':metrics(initial,query),'output_query':metrics(asset,query),'output_build':metrics(asset,build)})
        row={'model':name,'parameters':sum(p.numel() for p in model.parameters()),'curve':curve,
            'mean_delta_rmse':float(np.mean(errors)),'max_delta_rmse':float(np.max(errors)),
            'quality':quality,'wall_s_cumulative':time.monotonic()-start}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['quality','curve']}),flush=True)
    write(a.output/'results.json',{'models':rows,'wall_s':time.monotonic()-start,'claim':'continuous output can/cannot fit known targets, judged from actual errors; no method necessity established',
        'full_P1_pass':None,'human_verdict':None})
if __name__=='__main__':main()
