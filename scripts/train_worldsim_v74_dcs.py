"""DCS 小提议器：各自 FIT 的离线几何搜索教师，同预算无需求训练控制。"""
import argparse,json,sys,time,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.data import load_build
from motion_proj.worldsim_v74.common import write_json,farthest_ids
from motion_proj.worldsim_v74.c_dcs import geometry_context,patches_from_parameters,patch_incidence,patch_cost,master,pricing_features,PricingNetwork

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);rng=np.random.default_rng(7401);start=time.monotonic()
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text());index=json.loads((Path(cfg['data_root'])/'index.json').read_text())['cases']
selected=[]
for dataset in ['nuscenes','av2']:
    logs=sorted({r['log_id'] for r in index if r['dataset']==dataset and r['role']=='FIT'})
    for log in logs:
        pool=sorted([r for r in index if r['dataset']==dataset and r['role']=='FIT' and r['log_id']==log and r['build_points']>=8],key=lambda r:(r['build_points'],r['case_id']))
        if pool:selected.append(pool[len(pool)//2])
write_json(args.output/'manifest.json',{'task':'WS-V74-C-FIT-TRAIN-01','seed':7401,'cases':selected,'config':cfg,
    'teacher':'8 fixed stochastic continuous geometry proposals per anchor; true BUILD reduced cost plus 0.1 FIT extra-ray utility tie preference',
    'states':'8/16/32/64 FPS initial libraries; no DEV/FINAL data','epochs':200,'batch_size':64,'lr':.002,
    'control':'same architecture, same targets/init/update budget, alpha/beta/eta and demand summary zeroed',
    'failure_ledger_refs':cfg['failure_ledger_refs']})
checkpoints={};result=[]
for dataset in ['nuscenes','av2']:
    folder=args.output/dataset;folder.mkdir();tokens_all=[];global_all=[];target_all=[];teachers=[]
    parameters=cfg['dataset_parameters'][dataset];width=parameters['carrier_half_width_m'];eps=parameters['epsilon_obs_m']
    for case in [r for r in selected if r['dataset']==dataset]:
        b=load_build(Path(case['build_file']).parent);context=geometry_context(b);p=context['points']
        with np.load(case['query_truth_file'],allow_pickle=False) as raw:q={k:raw[k] for k in raw.files}
        for statecount in [8,16,32,64]:
            init=farthest_ids(p,min(statecount,len(p)));patches=patches_from_parameters(context,init,np.zeros((len(init),7)),width)
            A,E=patch_incidence(patches,b,eps);demand=master(A,E,patch_cost(patches,width),cfg['face_budget'])
            anchors=np.argsort(-demand['alpha'],kind='stable')[:min(8,len(p))]
            tok,glob=pricing_features(b,context,anchors,demand,width)
            candidates=rng.normal(size=(len(anchors),8,7))*np.array([.25,.25,.1,.2,.2,.3,.3])
            candidates[:,0]=0.;candidates=np.clip(candidates,-np.array([1,1,.4,.6,.6,.7,.7]),np.array([1,1,.4,.6,.6,.7,.7]))
            anchor_repeat=np.repeat(anchors,8);proposals=patches_from_parameters(context,anchor_repeat,candidates.reshape(-1,7),width)
            NA,NE=patch_incidence(proposals,b,eps);cost=patch_cost(proposals,width)
            price=cost-np.asarray(NA.T@demand['alpha']).ravel()+np.asarray(NE.T@demand['beta']).ravel()+8*demand['eta']
            QA,QE=patch_incidence(proposals,q,.2)
            utility=np.asarray(QA.sum(0)-QE.sum(0)).ravel()
            score=(price-.1*utility).reshape(len(anchors),8);chosen=score.argmin(1)
            target=candidates[np.arange(len(anchors)),chosen]
            if tok.shape[1]<16:tok=np.repeat(tok,16//tok.shape[1]+1,axis=1)[:,:16]
            tokens_all.append(tok);global_all.append(glob);target_all.append(target.astype(np.float32))
            teachers.append({'case_id':case['case_id'],'state_initial_patches':len(init),'anchors':anchors.tolist(),
                'chosen':chosen.tolist(),'true_price':price.reshape(len(anchors),8)[np.arange(len(anchors)),chosen].tolist(),
                'fit_extra_ray_utility':utility.reshape(len(anchors),8)[np.arange(len(anchors)),chosen].tolist()})
        print(dataset,case['case_id'],'teacher complete',flush=True)
    tokens=np.concatenate(tokens_all);global_features=np.concatenate(global_all);targets=np.concatenate(target_all)
    np.savez_compressed(folder/'training_examples.npz',tokens=tokens,global_features=global_features,parameters=targets)
    write_json(folder/'teacher_events.json',teachers);checkpoints[dataset]={}
    for mode in ['full','no_demand']:
        torch.manual_seed(7401);net=PricingNetwork().cuda();optim=torch.optim.Adam(net.parameters(),lr=.002)
        tt=tokens.copy();gg=global_features.copy()
        if mode=='no_demand':tt[:,:,-2:]=0;gg[:]=0
        x=torch.tensor(tt,device='cuda');g=torch.tensor(gg,device='cuda');target=torch.tensor(targets,device='cuda')
        history=[];updates=0;beg=time.monotonic()
        for epoch in range(200):
            order=torch.randperm(len(x),device='cuda');losses=[]
            for ids in order.split(64):
                pred=net(x[ids],g[ids]);loss=torch.nn.functional.smooth_l1_loss(pred,target[ids]);optim.zero_grad();loss.backward();optim.step()
                updates+=1;losses.append(float(loss))
            if epoch%20==0 or epoch==199:history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))})
        path=folder/f'{mode}.pt';torch.save({'state_dict':net.cpu().state_dict(),'seed':7401,'dataset':dataset,'mode':mode,'epochs':200,'updates':updates},path)
        checkpoints[dataset][mode]=str(path);result.append({'dataset':dataset,'mode':mode,'examples':len(x),'updates':updates,'parameters':sum(p.numel() for p in net.parameters()),'history':history,'wall_seconds':time.monotonic()-beg})
        print(dataset,mode,'training complete',updates,flush=True)
    write_json(args.output/'training.json',result)
write_json(args.output/'checkpoints.json',checkpoints)
cfg['dcs_checkpoints']=checkpoints
(ROOT/'configs/worldsim_v74/tournament.json').write_text(json.dumps(cfg,indent=2)+'\n')
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
