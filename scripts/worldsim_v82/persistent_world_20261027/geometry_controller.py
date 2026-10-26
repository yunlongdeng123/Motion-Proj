"""由实测深度失配触发的②单变量原型；等待健康10K结束，不中断/重配基线。"""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PYTHON='/root/autodl-tmp/envs/drivingeditor/bin/python'


def main():
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_baseline10k',completed=[])
    def save():
        p=ROOT/'geometry_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2));q.replace(p)
    save()
    try:
        while True:
            current=json.loads((ROOT/'training_status.json').read_text())
            if current.get('all_training_complete'):break
            if current['phase']=='engineering_repair_needed':raise RuntimeError('先修基线控制器，不越过故障启动新实验')
            time.sleep(15)
        config=json.loads((ROOT/'repair_config.json').read_text())
        config.update(surface_moment_weight=.01,surface_tolerance_m=.3,
            purpose='固定①③与原保背景训练合同，只增加有来源支持射线的贡献深度矩约束；不要求道路贡献增大')
        path=ROOT/'geometry_config.json';path.write_text(json.dumps(config,indent=2))
        for name,args in [
            ('geometry_smoke_save',['--output',str(ROOT/'geometry_smoke'),'--steps','100']),
            ('geometry_smoke_restore_10min',['--output',str(ROOT/'geometry_smoke'),'--steps','101','--min-seconds','600','--resume',str(ROOT/'geometry_smoke/checkpoint_100.pth')]),
            ('geometry_fresh10k',['--output',str(ROOT/'geometry10k'),'--steps','10000'])]:
            status['phase']=name;save();cmd=[PYTHON,str(ROOT/'code/train_repair.py'),'--config',str(path)]+args
            with (ROOT/(name+'.log')).open('a') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
                status['child']=dict(pid=p.pid,start_ticks=Path('/proc/%s/stat'%p.pid).read_text().split()[21],command=cmd);save()
                assert p.wait()==0,name
            status.pop('child');status['completed'].append(name);save()
        status.update(phase='paired10k_evaluation_pending',training_complete=True);save()
    except Exception:
        status.update(phase='engineering_repair_needed',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
