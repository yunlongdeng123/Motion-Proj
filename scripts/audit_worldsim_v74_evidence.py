"""只读已有事件资产，补齐首次退化与实际生片价值取证；不回流求解器。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.common import ray_state,write_json
from motion_proj.worldsim_v74.c_dcs import geometry_context,patches_from_parameters,patch_incidence
from motion_proj.worldsim_v74.data import load_build
parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--synthetic',action='store_true');args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
manifest=json.loads((args.assets/'manifest.json').read_text());cfg=manifest['config'];records=[]
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def state(path,obs,eps):
    z=arrays(path);return ray_state(z['vertices_actor_m'],z['faces'],obs,eps)
if args.synthetic:
    cases=[{'case_id':p.name,'dataset':'synthetic','folder':p} for p in sorted(args.assets.glob('missing_support_*'))]
else:cases=manifest['cases']
for case in cases:
    if args.synthetic:
        src=case['folder'];build=arrays(src/'build.npz');query=arrays(src/'query.npz');params=cfg
    else:
        src=args.assets/case['dataset']/case['case_id'];build=load_build(Path(case['build_file']).parent);query=arrays(case['query_truth_file']);params={**cfg,**cfg['dataset_parameters'][case['dataset']]}
    context=None
    for method in manifest['methods']:
        folder=src/method
        if not (folder/'initial.npz').exists():continue
        qprevious=state(folder/'initial.npz',query,.2);bprevious=state(folder/'initial.npz',build,params['epsilon_obs_m']);prev_path=folder/'initial.npz'
        positive=query['positive_actor'].astype(bool);qowned=np.flatnonzero(positive&~query['ambiguous_owner'].astype(bool))
        events=[];first_decline=None;out=args.output/case['dataset']/case['case_id']/method;out.mkdir(parents=True,exist_ok=True)
        for i,path in enumerate(sorted(folder.glob('event_*.npz'))):
            qcurrent=state(path,query,.2);bcurrent=state(path,build,params['epsilon_obs_m'])
            lost=np.flatnonzero(positive&qprevious['hit']&~qcurrent['hit']);gained=np.flatnonzero(positive&~qprevious['hit']&qcurrent['hit']);new_early=np.flatnonzero(~qprevious['early']&qcurrent['early'])
            event={'event':i,'asset':str(path),'query_owned_count':int(positive.sum()),'query_hits_before':int(qprevious['hit'][positive].sum()),'query_hits_after':int(qcurrent['hit'][positive].sum()),
                   'lost_query_hit_rays':lost.tolist(),'gained_query_hit_rays':gained.tolist(),'new_query_early_rays':new_early.tolist()}
            if first_decline is None and (len(lost)>len(gained) or len(new_early)):
                first_decline={'before':str(prev_path),'after':str(path),'event':i,'meaning':'first net loss of owned QUERY hits or new early QUERY ray; diagnostic only, never used to select asset'}
                np.savez_compressed(out/'first_degradation_rays.npz',lost_hit_ids=lost,gained_hit_ids=gained,new_early_ids=new_early,
                    observed_first_range_m=query['observed_first_range_m'],before_first_range_m=qprevious['first_range_m'],after_first_range_m=qcurrent['first_range_m'])
            pricing=folder/f'pricing_{i:02d}.npz'
            if pricing.exists():
                if context is None:context=geometry_context(build)
                z=arrays(pricing);proposals=patches_from_parameters(context,z['anchors'],z['parameters'],params['carrier_half_width_m'])
                A,E=patch_incidence(proposals,build,params['epsilon_obs_m']);Q,QE=patch_incidence(proposals,query,.2)
                needed=~bprevious['hit'][context['owned']]&~bprevious['early'][context['owned']]
                qneeded=~qprevious['hit'][qowned]&~qprevious['early'][qowned]
                new_support=np.asarray(A[needed].sum(0)).ravel();free=np.asarray(E.sum(0)).ravel()
                qsupport=np.asarray(Q[qneeded].sum(0)).ravel();qfree=np.asarray(QE.sum(0)).ravel()
                useful=(new_support>0)&(free==0);quseful=(qsupport>0)&(qfree==0)
                event.update({'proposed':len(useful),'useful_build_proposals':int(useful.sum()),'useful_query_proposals':int(quseful.sum()),
                    'proposal_new_build_support':new_support.astype(int).tolist(),'proposal_new_query_support':qsupport.astype(int).tolist(),
                    'proposal_build_early_count':free.astype(int).tolist(),'proposal_query_early_count':qfree.astype(int).tolist()})
            events.append(event);qprevious=qcurrent;bprevious=bcurrent;prev_path=path
        proposed=sum(e.get('proposed',0) for e in events);ub=sum(e.get('useful_build_proposals',0) for e in events);uq=sum(e.get('useful_query_proposals',0) for e in events)
        row={'dataset':case['dataset'],'case_id':case['case_id'],'method':method,'first_degradation':first_decline,'proposed':proposed,'useful_build':ub,'useful_query':uq,
             'useful_build_rate':ub/proposed if proposed else None,'useful_query_rate':uq/proposed if proposed else None,
             'definition':'proposal repairs at least one previously missed/late first return and creates no early hit on any observed ray; QUERY diagnostic never enters inference; actual selected event hits are reported separately'}
        write_json(out/'event_evidence.json',{'summary':row,'events':events});records.append(row)
    write_json(args.output/'summary.json',{'source':str(args.assets),'records':records,'wall_seconds':time.monotonic()-start});print(case['case_id'],'evidence saved',flush=True)
