"""已授权预算的顺序生命周期与配对；全程服从road_asset单GPU租约。"""
from pathlib import Path
import json
import os
import subprocess
import sys

G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
code=G/'code';root=G/'check3';smoke=root/'smoke_r1'
command=[sys.executable,str(code/'train_real_pair.py')]

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def run(name,arm,mode,steps,stop=None,resume=None):
    output=(smoke/name) if mode=='smoke' else root/arm/'formal_r1'
    args=command+['--arm',arm,'--mode',mode,'--steps',str(steps),'--output',str(output)]
    if stop is not None:args+=['--stop-step',str(stop)]
    if resume is not None:args+=['--resume',str(resume)]
    print(json.dumps(dict(starting=name,arm=arm,mode=mode,output=str(output))),flush=True)
    with (root/(name+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(completed=name)),flush=True)
    return output

if not (smoke/'save2/completed.json').is_file():raise RuntimeError('I0 initialsave2 must complete first')
if read(smoke/'save2/completed.json')['actual_updates']!=2:raise RuntimeError('Initial smoke budget mismatch')
source=smoke/'save2/checkpoint000002.pt'
left=run('restore3','I0','smoke',4,3,source)
right=run('replay3','I0','smoke',4,3,source)

# CPU读取完整快照；不创建第二个native上下文。
import torch
sys.path.insert(0,str(code))
from train_real_pair import differences
a=torch.load(left/'checkpoint000003.pt',map_location='cpu');b=torch.load(right/'checkpoint000003.pt',map_location='cpu')
different=differences(a,b);maxima={}
def compare(x,y,path=''):
    if torch.is_tensor(x):
        if not torch.equal(x,y):maxima[path]=float((x.double()-y.double()).abs().max())
    elif isinstance(x,dict):
        for key in x:compare(x[key],y[key],path+'/'+str(key))
    elif isinstance(x,(list,tuple)):
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
compare(a,b)
unexpected=[path for path in different if not (path.startswith('/model/delta_values') or path.startswith('/optimizer/state/'))]
if unexpected or any(value>1e-6 for value in maxima.values()):raise RuntimeError('Native replay mismatch:'+str((different,maxima)))
write(smoke/'replay_comparison.json',dict(full_snapshot_compared=True,bitwise_exact=not different,differences=different,tensor_maxima=maxima,
    RNG_sampler_trace_contract_exact=not unexpected,parameter_tolerance=1e-6,formal_updates=0))
finish=run('finish4','I0','smoke',4,4,source)
w4=run('W4_save1','W4','smoke',2,1)
w4_done=run('W4_restore2','W4','smoke',2,2,w4/'checkpoint000001.pt')
for path in (smoke/'save2',left,right,finish,w4,w4_done):
    reload=read(path/'public_reload.json');guard=read(path/'joint_K.json');known=read(path/'shadow_known_persistence.json')
    if not guard['all_exact'] or any(row['RGB_max']>1e-6 or row['alpha_max']>1e-6 for row in reload['rows']):raise RuntimeError('Smoke persistence/K failed')
    if known['effective_RGB_changed_rows']<=0 or not all(known['prior_ledger_fields_exact'].values()):raise RuntimeError('S-only-known persistence did not exercise effective override')
updates=sum(read(path/'completed.json')['actual_updates'] for path in (smoke/'save2',left,right,finish,w4,w4_done))
if updates!=8:raise RuntimeError('Actual engineering count differs from8')
write(smoke/'completed.json',dict(completed=True,engineering_updates=updates,formal_updates=0,
    I0_and_W4_fullrestore=True,next_update_replay=True,next_parameterization_entry=True,
    public_reload_5frames_and_S_only_known=True,all_joint_K_exact=True))
print(json.dumps(dict(lifecycle_complete=True,engineering_updates=8,starting_formal=True)),flush=True)
run('formal_I0','I0','formal',200)
run('formal_W4','W4','formal',200)
write(root/'pair_completed.json',dict(completed=True,formal_updates=400,engineering_updates=8,context_probe_optimizer_updates=0,
    roles=['I0','W4'],GPU_must_remain_on=True,production_promoted=False))
print(json.dumps(dict(completed=True,formal_updates=400,engineering_updates=8)),flush=True)
