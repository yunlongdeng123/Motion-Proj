"""真实10分钟保存/评价/恢复smoke，然后fresh固定10K基线；额外更新分账。"""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PYTHON='/root/autodl-tmp/envs/drivingeditor/bin/python'


def main():
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_reference',completed=[])
    def save():
        p=ROOT/'training_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2));q.replace(p)
    save()
    try:
        operation=json.loads((ROOT/'operations_dispatch.json').read_text())
        proc=Path('/proc/%s/stat'%operation['pid'])
        while proc.exists() and proc.read_text().split()[2]!='Z':time.sleep(10)
        assert (ROOT/'phase2_operations/summary.json').exists()
        views=[94,110,126];reference=ROOT/'phase2_reference49/crossview'
        while not all((reference/f'{raw}.npz').exists() for raw in views):time.sleep(10)
        meta=json.loads(Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json').read_text())
        config=dict(seed=0,sampler_seed=20261027,train_views=views,reference_root=str(reference),
            initialization='/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026/full199_initialization.npz',
            meter_to_model=meta['meter_to_model'],initial_static_multiplier=.999,
            purpose='可配置多视图修复基线；三个新开发视图，不改变主副本/固定几何/原loss与LR')
        config_path=ROOT/'repair_config.json';config_path.write_text(json.dumps(config,indent=2))
        jobs=[('smoke_save',['--output',str(ROOT/'smoke_repair'),'--steps','100']),
            ('smoke_restore_train_10min',['--output',str(ROOT/'smoke_repair'),'--steps','101','--min-seconds','600','--resume',str(ROOT/'smoke_repair/checkpoint_100.pth')]),
            ('fresh_repair10k',['--output',str(ROOT/'repair10k'),'--steps','10000'])]
        for name,args in jobs:
            status['phase']=name;save()
            cmd=[PYTHON,str(ROOT/'code/train_repair.py'),'--config',str(config_path)]+args
            with (ROOT/(name+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
                status['child']=dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().split()[21],command=cmd);save()
                result=child.wait();assert result==0,(name,result)
            status.pop('child');status['completed'].append(name);save()
        status.update(phase='repair10k_evaluation_pending',all_training_complete=True);save()
    except Exception:
        status.update(phase='engineering_repair_needed',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
