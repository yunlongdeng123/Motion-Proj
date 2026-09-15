"""Instrument official HUGSIM + official LTF modules without replacing simulator physics."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import zipfile
import dataclasses
import importlib

BASE = Path('/root/autodl-tmp/external/worldsim_simimpact')
ROOT = Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
HUG = BASE/'HUGSIM'
sys.path[:0] = [str(HUG),str(HUG/'sim'),str(BASE/'NAVSIM')]
os.chdir(HUG)
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from hugsim_env.envs.hug_sim import HUGSimEnv
from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from hugsim.dataparser import parse_raw
import open3d as o3d

def extract(zip_path,dest):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target=(dest/name).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError(name)
        z.extractall(dest)

def scalarize(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {k:scalarize(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [scalarize(v) for v in x]
    return x

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True)
    parser.add_argument('--steps',type=int,default=401)
    parser.add_argument('--tag',default='native')
    parser.add_argument('--collision-sample-stride',type=int,default=1)
    parser.add_argument('--collision-cloud',type=str,default=None)
    parser.add_argument('--controller-fixed-iterations',action='store_true')
    args=parser.parse_args()
    random.seed(20260915); np.random.seed(20260915); torch.manual_seed(20260915)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    controller_module=importlib.import_module('sim.ilqr.lqr')
    if args.controller_fixed_iterations:
        controller_module.lqr._solver_params=dataclasses.replace(controller_module.lqr._solver_params,max_solve_time=None)
    out=ROOT/'rollouts'/args.scene/args.tag
    out.mkdir(parents=True,exist_ok=False)
    model_root=ROOT/'assets'/'extracted'
    model_dir=model_root/args.scene
    if not model_dir.exists():extract(ROOT/'assets'/'scenes'/'nuscenes'/f'{args.scene}.zip',model_root)
    scen_dir=ROOT/'assets'/'scenarios'
    if not scen_dir.exists():extract(ROOT/'assets'/'scenarios.zip',scen_dir)
    candidates=list(model_root.glob(f'**/{args.scene}/cfg.yaml'))
    if not candidates:raise FileNotFoundError(f'No cfg for {args.scene}')
    model_dir=candidates[0].parent
    cfg=OmegaConf.merge({'scenario':OmegaConf.load(scen_dir/'nuscenes'/f'{args.scene}-easy-00.yaml')},
        {'base':OmegaConf.load(HUG/'configs/sim/nuscenes_base.yaml')},
        {'camera':OmegaConf.load(HUG/'configs/sim/nuscenes_camera.yaml')},
        {'kinematic':OmegaConf.load(HUG/'configs/sim/kinematic.yaml')})
    cfg.update(OmegaConf.load(model_dir/'cfg.yaml'))
    cfg.model_path=str(model_dir)
    cfg.base.model_base=str(model_dir.parent)
    cfg.base.realcar_path=str(ROOT/'assets'/'3DRealCar')
    OmegaConf.save(cfg,out/'resolved_config.yaml')
    env=HUGSimEnv(cfg=cfg,output=str(out))
    source_count=len(env.points)
    if args.collision_cloud:
        env.points=torch.from_numpy(np.load(args.collision_cloud)['points'].astype(np.float32)).cuda()
    if args.collision_sample_stride>1:
        env.points=env.points[::args.collision_sample_stride]
    (out/'collision_protocol.json').write_text(json.dumps({'source_count':source_count,'evaluated_count':len(env.points),
        'sample_stride':args.collision_sample_stride,'external_cloud':args.collision_cloud,
        'scope':'Collision-channel sensitivity control; RGB, renderer, vehicle physics and policy unchanged. Not a natural badcase.' if args.collision_sample_stride>1 else 'Native unless explicit external cloud'},indent=2))
    obs,info=env.reset(seed=20260915)
    config=TransfuserConfig();config.latent=True
    agent=TransfuserAgent(config,lr=1e-4,checkpoint_path=str(ROOT/'assets/ltf_seed_0.ckpt'))
    agent.initialize();agent.eval()
    data=parse_raw((obs,info))
    cpu_traj=agent.compute_trajectory(data['input']).poses
    agent.cuda()
    def predict(ob,inf):
        inputs=parse_raw((ob,inf))['input']
        feats={}
        for builder in agent.get_feature_builders():feats.update(builder.compute_features(inputs))
        feats={k:v.unsqueeze(0).cuda() for k,v in feats.items()}
        with torch.no_grad():pred=agent(feats)['trajectory'][0].cpu().numpy()
        return pred,feats
    gpu_traj,feats=predict(obs,info)
    qa={'strict_checkpoint_loading':True,'cpu_gpu_max_abs_trajectory_difference_m':float(np.max(np.abs(cpu_traj-gpu_traj))),
        'torch':torch.__version__,'policy_device':'cuda','initial_policy_inputs':{k:list(v.shape) for k,v in feats.items()},
        'renderer':'official HUGSIM_splat RGB+ED+S','controller':'official traj2control / iLQR','simulator':'official HUGSimEnv.step',
        'controller_max_solve_time':controller_module.lqr._solver_params.max_solve_time,
        'controller_max_iterations':controller_module.lqr._solver_params.max_ilqr_iterations,
        'controller_comparison':'Diagnostic fixed iteration budget; removes official 50ms wall-clock stopping' if args.controller_fixed_iterations else 'Official 50ms stopping'}
    (out/'runtime_qa.json').write_text(json.dumps(qa,indent=2))
    if qa['cpu_gpu_max_abs_trajectory_difference_m']>1e-3:raise RuntimeError('Unexpected CPU/GPU policy discrepancy')
    with (out/'initial_observation.pkl').open('wb') as f:pickle.dump((obs,info),f)
    frames=[]; trace=[]; timing=[]; started=time.time()
    for step in range(args.steps):
        ts=time.time()
        if step%8==0:
            Image.fromarray(obs['rgb']['CAM_FRONT']).save(out/f'front_{step:04d}.jpg',quality=95)
        traj,_=predict(obs,info)
        waypoints=traj[:,[1,0]].copy();waypoints[:,0]*=-1
        acc,steer=traj2control(waypoints,info)
        preinfo=info
        obs,reward,terminated,truncated,info=env.step({'acc':acc,'steer_rate':steer})
        global_traj=traj_transform_to_global(traj[:,:2],preinfo['ego_box'])
        frames.append({'time_stamp':preinfo['timestamp'],'is_key_frame':True,'ego_box':preinfo['ego_box'],
            'obj_boxes':preinfo['obj_boxes'],'obj_names':['car']*len(preinfo['obj_boxes']),
            'planned_traj':{'traj':global_traj,'timestep':0.5},'collision':info['collision'],'rc':info['rc']})
        trace.append(scalarize({'step':step,'pre_info':{k:v for k,v in preinfo.items() if k!='cam_params'},
            'post_info':{k:v for k,v in info.items() if k!='cam_params'},'trajectory':traj,
            'acc':acc,'steer_rate':steer,'reward':reward,'terminated':terminated,'truncated':truncated}))
        timing.append(time.time()-ts)
        if step%8==0 or terminated:print(args.scene,step,'position',info['ego_pos'],'collision',info['collision'],'rc',info['rc'],flush=True)
        if terminated or truncated:break
    Image.fromarray(obs['rgb']['CAM_FRONT']).save(out/'front_final.jpg',quality=95)
    (out/'trace.json').write_text(json.dumps(trace,indent=2))
    payload=[{'type':'closeloop','frames':frames}]
    with (out/'data.pkl').open('wb') as f:pickle.dump(payload,f)
    summary={'scene':args.scene,'steps':len(trace),'simulated_seconds':float(info['timestamp']),
        'collision':bool(info['collision']),'route_completion':float(info['rc']),
        'terminal':bool(terminated or truncated),'wall_seconds':time.time()-started,
        'mean_step_seconds':float(np.mean(timing)),'gpu_peak_GiB':torch.cuda.max_memory_allocated()/1024**3,
        'status':'COMPLETED' if terminated or truncated else 'HORIZON_LIMIT'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if len(frames)>3:
        ground=np.asarray(o3d.io.read_point_cloud(str(out/'ground.ply')).points)
        scene=np.asarray(o3d.io.read_point_cloud(str(out/'scene.ply')).points)
        result=hugsim_evaluate(payload,ground,scene)
        (out/'official_eval.json').write_text(json.dumps(scalarize(result),indent=2))

if __name__=='__main__':main()
