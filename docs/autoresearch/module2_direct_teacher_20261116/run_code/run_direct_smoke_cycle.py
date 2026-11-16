"""Independent-process 2/2/2 lifecycle smoke; every extra update accounted."""
from pathlib import Path
import json
import subprocess
import sys
import torch

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0,str(R/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.road_texture_training import differences

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

protocol=C/'smoke_protocol_v1.json'
if not protocol.exists():
    data=json.loads((C/'protocol_draft.json').read_text());data['formal_authorized']=False
    data['role']='immutable engineering smoke, no formal updates'
    write(protocol,data)
root=C/'phase0/lifecycle_r4'
root.mkdir(parents=True,exist_ok=False)
baseline=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
common=[sys.executable,'-m','drivingeditor_dynamic_edit.road_direct_teacher_training',
    '--manifest',str(baseline/'persistent/scene.json'),'--proof',str(baseline/'persistent/proof.json'),
    '--asset',str(baseline),'--r8-asset','/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/asset_r8_guarded',
    '--supervision',str(C/'supervision_r3'),'--protocol',str(protocol),'--mode','smoke','--schedule','2,2,2']
commands=[('save_color',2,None),('restore_next',3,'save_color/checkpoint000002.pt'),
          ('replay_next',3,'save_color/checkpoint000002.pt'),('next_stage',6,'restore_next/checkpoint000003.pt')]
for name,stop,resume in commands:
    command=common+['--output',str(root/name),'--stop-step',str(stop)]
    if resume:command+=['--resume',str(root/resume)]
    print(json.dumps(dict(starting=name,stop=stop)),flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(completed=name)),flush=True)

left=torch.load(root/'restore_next/checkpoint000003.pt',map_location='cpu')
right=torch.load(root/'replay_next/checkpoint000003.pt',map_location='cpu')
different=differences(left,right)
maxima={}
def tensor_errors(a,b,path=''):
    if torch.is_tensor(a):
        if not torch.equal(a,b):maxima[path]=float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
    elif isinstance(a,dict):
        for key in a:tensor_errors(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        for i,(x,y) in enumerate(zip(a,b)):tensor_errors(x,y,path+'/'+str(i))
tensor_errors(left,right)
non_numeric=[item for item in different if not (item.startswith('/model/') or item.startswith('/optimizer/state/'))]
if non_numeric:raise RuntimeError('replay metadata/RNG/sampler changed:'+str(non_numeric))
if any(value>1e-6 for value in maxima.values()):raise RuntimeError('replay numerical mismatch >1e-6:'+str(maxima))
write(root/'replay_comparison.json',dict(full_snapshot_checked=True,bitwise_exact=not different,differences=different,
    tensor_maxima=maxima,nonparameter_metadata_exact=True,tolerance=1e-6,
    first_restore=json.loads((root/'restore_next/restore.json').read_text()),
    second_restore=json.loads((root/'replay_next/restore.json').read_text())))
write(root/'engineering_accounting.json',dict(formal_updates=0,lifecycle_updates=7,
    detail=dict(save_color=2,restore_next=1,replay_next=1,next_stage=3),
    prior_successful_color_probe_updates=2,total_engineering_updates_this_round=9,
    stages_tested=['color','coverage','joint'],independent_process_restore=True))
write(root/'completed.json',dict(completed=True,formal_updates=0,smoke_updates=7,bitwise_replay=not different))
print(json.dumps(dict(completed=True,root=str(root),replay_differences=different)),flush=True)
