import os,json,time,sys,subprocess
from pathlib import Path
from segmentation import ROOT,CONTEXTS,write
from v43_resource_control import require_start
previous=ROOT/'actor_motion_dispatch.json'
if previous.exists():
    old=json.loads(previous.read_text());p=Path('/proc/%s/stat'%old['pid'])
    assert not p.exists() or p.read_text().split()[2]=='Z' or p.read_text().split()[21]!=str(old['start_ticks']),'Existing controller still alive'
    child=old.get('child')
    if child:
        p=Path('/proc/%s/stat'%child['pid']);assert not p.exists() or p.read_text().split()[2]=='Z' or p.read_text().split()[21]!=str(child['start_ticks']),'Existing child still alive'
    history=ROOT/'motion_engineering_history';history.mkdir(exist_ok=True);write(history/('dispatch_'+str(old['pid'])+'.json'),old)
status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_formal10k',child=None,completed=[])
def update(**kw):status.update(kw);write(ROOT/'actor_motion_dispatch.json',status)
def run(ctx,stage,command):
    require_start(ctx);update(context=ctx,phase=stage,command=command)
    log=ROOT/('motion_'+ctx+'_'+stage+'.log')
    if log.exists():log=ROOT/('motion_'+ctx+'_'+stage+'_retry_'+str(os.getpid())+'.log')
    with log.open('w') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT);update(child=dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().split()[21]));rc=child.wait()
    update(child=None,returncode=rc);assert rc==0,(ctx,stage,rc)
update()
for ctx in CONTEXTS:
    if ctx in json.loads((ROOT/'authorization.json').read_text()).get('held_contexts',[]):continue
    while True:
        require_start();p=ROOT/'actor_campaign_v1'/ctx/'formal10k/status.json'
        if p.exists() and json.loads(p.read_text()).get('phase')=='complete':break
        time.sleep(5)
    output=ROOT/'actor_motion_v1'/ctx
    summary=output/'summary.json'
    if summary.exists():
        d=json.loads(summary.read_text());assert d['frames_per_case']==31 and d['source_checkpoint_unchanged']
        for case in d['cases']:assert len(list((output/case/'native_rgb').glob('*.png')))==31
    else:run(ctx,'native',[sys.executable,'-u',str(ROOT/'code/render_actor_motion.py'),'--context',ctx])
    for case in ['delete_target','translate_lateral_1m','turn_left15']:
        done=output/case/'propainter/summary.json'
        if done.exists():
            d=json.loads(done.read_text());assert d['status']=='complete' and d['frames']==31 and len(list((output/case/'propainter/model_png').glob('*.png')))==31
        else:run(ctx,'backend_'+case,['/root/autodl-tmp/envs/motionproj/bin/python','-u',str(ROOT/'code/fixed_propainter.py'),'--root',str(output),'--case',case,'--input-kind','native_3d'])
    status['completed'].append(ctx);update()
update(phase='complete')
