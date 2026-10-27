"""Sequential scene smoke->strict restore->fresh10k->evaluation. No old queue."""
import os,sys,json,time,subprocess,shutil
from pathlib import Path
from segmentation import ROOT,CONTEXTS,write
state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='running',contexts=[],formal_budget_per_case=10000,smoke_extra_updates=0,old_queue_resumed=False)
def run(ctx,name,extra):
    assert not (ROOT/'USER_STOP').exists()
    free=shutil.disk_usage(str(ROOT)).free
    assert free>25*1024**3,'Reserve exceeded; preserve checkpoints and request storage handling'
    dest=ROOT/'actor_campaign_v1'/ctx/name
    command=[sys.executable,'-u',str(ROOT/'code/train_actor.py'),'--context',ctx,'--output',str(dest)]+extra
    state.update(context=ctx,stage=name,command=command,free_bytes=free);write(ROOT/'actor_campaign_dispatch.json',state)
    with (ROOT/('actor_'+ctx+'_'+name+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state.update(child=child.pid,child_start_ticks=Path('/proc/%s/stat'%child.pid).read_text().split()[21]);write(ROOT/'actor_campaign_dispatch.json',state);code=child.wait()
    state.update(child=None);write(ROOT/'actor_campaign_dispatch.json',state)
    assert code==0,(ctx,name,code)
    result=json.loads((dest/'status.json').read_text());node=json.loads((dest/'latest_complete_node.json').read_text());assert node['frames']==49
    if 'smoke' in name:state['smoke_extra_updates']+=result['step']-extra_step(extra)
    write(ROOT/'actor_campaign_dispatch.json',state)
    return dest,result,node
def extra_step(extra):
    if '--resume' not in extra:return 0
    return int(Path(extra[extra.index('--resume')+1]).stem.replace('checkpoint',''))
for ctx in CONTEXTS:
    fresh,fs,fn=run(ctx,'smoke_fresh',['--smoke-seconds','300'])
    assert fs['since_first_step']>=300
    restored,rs,rn=run(ctx,'smoke_restore',['--smoke-seconds','300','--resume',fn['checkpoint']])
    assert rs['since_first_step']>=300 and json.loads((restored/'restore_verification.json').read_text())['all_passed']
    final,cs,cn=run(ctx,'formal10k',['--steps','10000'])
    assert cs['step']==10000 and cs['phase']=='complete'
    state['contexts'].append(dict(context=ctx,formal=cs,latest_complete_node=cn,smoke_fresh=fs,smoke_restore=rs));write(ROOT/'actor_campaign_dispatch.json',state)
state.update(phase='complete');write(ROOT/'actor_campaign_dispatch.json',state)
