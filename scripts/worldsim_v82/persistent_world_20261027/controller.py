"""单GPU序列；等待已登记原生任务，不重复启动；错误保留现场交由代理修复。"""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


def dump(value):
    p=ROOT/'controller_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)


def alive(pid,ticks):
    p=Path('/proc/%s/stat'%pid)
    if not p.exists():return False
    row=p.read_text().split();return row[21]==str(ticks) and row[2]!='Z'


def main():
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],started=time.time())
    old=ROOT/'controller_launch.json'
    if old.exists():
        prior=json.loads(old.read_text())
        assert not alive(prior['pid'],prior['start_ticks']),'旧控制器仍在运行'
    old.write_text(json.dumps(identity,indent=2))
    status=dict(identity,phase='native_render',completed=[],quality_accepted=False,human_verdict=None)
    dump(status)
    try:
        launch=json.loads((ROOT/'launch.json').read_text())['children']['native']
        while alive(launch['pid'],launch['start_ticks']):time.sleep(10)
        summary=ROOT/'phase1_native/summary.json'
        assert summary.exists(), '原生渲染未正常结束，检查native.log并保留现场'
        status['completed'].append('native_render')
        for case in ['delete_all_unrepaired','delete_all_repaired']:
            receipt=ROOT/'phase1_native'/case/'propainter/summary.json'
            if receipt.exists():
                assert json.loads(receipt.read_text())['status']=='complete';continue
            status['phase']='pretrained_video_'+case;status['updated']=time.time();dump(status)
            command=['/root/autodl-tmp/envs/motionproj/bin/python',str(ROOT/'code/inpaint_baseline.py'),
                '--root',str(ROOT/'phase1_native'),'--case',case]
            with (ROOT/('inpaint_'+case+'.log')).open('a') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
                status['child']=dict(pid=process.pid,start_ticks=Path('/proc/%s/stat'%process.pid).read_text().split()[21],command=command);dump(status)
                result=process.wait();assert result==0,(case,result)
            status['completed'].append(status['phase']);status.pop('child',None);dump(status)
        status.update(phase='phase1_video_delivery_and_phase2',pipeline_complete=True,updated=time.time())
        dump(status)
    except Exception:
        status.update(phase='engineering_repair_needed',error=traceback.format_exc(),updated=time.time());dump(status);raise


if __name__=='__main__':main()
