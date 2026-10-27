"""Read actual workers, completed boundaries and costs; never a launcher."""
import json,time,os,shutil
from pathlib import Path
from segmentation import ROOT,CONTEXTS,write
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def ident(d):
    if not d:return None
    try:
        s=Path('/proc/%s/stat'%d['pid']).read_text().split();return dict(pid=d['pid'],expected=str(d.get('start_ticks')),actual=s[21],alive=s[2]!='Z' and str(d.get('start_ticks'))==s[21])
    except FileNotFoundError:return dict(pid=d['pid'],alive=False)
state=dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),campaign=read(ROOT/'actor_campaign_dispatch.json'),monitor=read(ROOT/'gpu_peak.json'),disk=dict(zip(['total','used','free'],shutil.disk_usage(str(ROOT)))),authorization=read(ROOT/'authorization.json'),quota=read(ROOT/'quota_state.json'),scenes=[],workers=[],controls=[])
for path in ROOT.glob('*_dispatch_receipt.json'):
    d=read(path);state['workers'].append(dict(receipt=path.name,identity=ident(d),command=d.get('command')))
for ctx in CONTEXTS:
    stages={}
    for stage in ['smoke_fresh','smoke_restore','formal10k']:
        p=ROOT/'actor_campaign_v1'/ctx/stage;s=read(p/'status.json')
        if s:
            stages[stage]=dict(status=s,identity=ident(s),latest_complete_node=read(p/'latest_complete_node.json'),restore_verification=read(p/'restore_verification.json'))
    ds=read(ROOT/'datasets'/ctx/'manifest.json');state['scenes'].append(dict(context=ctx,stages=stages,seed_points=ds['actor_local_voxel_seeds'],initial_outside_box_rows=ds['initial_outside_box_rows']))
for family in ['zero_control_v1','first10k_control_v1']:
    for p in (ROOT/family).glob('*/common_observation_comparison.json'):
        d=read(p);state['controls'].append(dict(family=family,context=d['context'],step=d['step'],frames=len(d['frames']),old_full_psnr_mean=sum(x['old']['full_psnr'] for x in d['frames'])/49,new_full_psnr_mean=sum(x['new']['full_psnr'] for x in d['frames'])/49,old_body_psnr_mean=sum(x['old']['body_psnr'] for x in d['frames'])/49,new_body_psnr_mean=sum(x['new']['body_psnr'] for x in d['frames'])/49,reverse_replay=d['reverse_replay']))
state['unconstrained_control']=read(ROOT/'first_unconstrained10k/status.json');write(ROOT/'state_snapshot.json',state);print(json.dumps(state))
