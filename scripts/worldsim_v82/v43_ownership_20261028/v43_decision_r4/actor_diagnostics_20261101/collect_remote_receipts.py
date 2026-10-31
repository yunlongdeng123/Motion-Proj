"""仅取回本任务小收据，保留远端原始数据/派生数据。"""
import json,pathlib,tarfile,subprocess,time
BASE=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
files=[]
for name in ('selector.log','cropped.log','stop_after_020.log','raw_inventory.json','scenes.json'):
    p=BASE/name
    if p.exists():files.append(p)
for p in (BASE/'selected').glob('*/manifest.json'):files.extend([p,p.with_name('source_receipts.json')])
repo='/root/autodl-tmp/motion_proj_v82_gn11_v42'
state=dict(task='V43_ACTOR_DIAGNOSTICS_20261101',scope='raw-media and measured-LiDAR worker only',captured_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    cpu_max=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),owned_pids={str(pid):pathlib.Path('/proc',str(pid)).exists() for pid in (1607,1865,1889)},
    git_status=subprocess.check_output(['git','-C',repo,'status','--short'],text=True),
    git_branch=subprocess.check_output(['git','-C',repo,'branch','--show-current'],text=True).strip(),
    git_HEAD=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip(),
    checkpoint_loads=0,training_updates=0,GPU_queries=0,shutdown_performed=False,
    note='No original dataset/model files written. Selector/crop/stop controller all stopped; derived files and partial probe retained.')
(BASE/'closeout_state.json').write_text(json.dumps(state,indent=2));files.append(BASE/'closeout_state.json')
with tarfile.open(BASE/'execution_receipts.tar.gz','w:gz') as tar:
    for p in files:
        if p.exists():tar.add(p,arcname=str(p.relative_to(BASE)))
print(json.dumps(state))
