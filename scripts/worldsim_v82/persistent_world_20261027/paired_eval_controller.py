"""两个10K结束后继续真实49帧原生/视频评价；不因训练结束就停任务。"""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


def main():
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_paired_training',completed=[])
    def save():
        p=ROOT/'paired_eval_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2));q.replace(p)
    save()
    try:
        while True:
            current=json.loads((ROOT/'geometry_status.json').read_text()) if (ROOT/'geometry_status.json').exists() else {}
            if current.get('training_complete'):break
            if current.get('phase')=='engineering_repair_needed':raise RuntimeError('几何任务需工程修复，不能假装训练完成')
            time.sleep(15)
        for arm in ['repair10k','geometry10k']:
            for phase,cmd in [
                ('native49',['/root/autodl-tmp/envs/drivingeditor/bin/python',str(ROOT/'code/evaluate_repair_state.py'),'--checkpoint',str(ROOT/arm/'checkpoint_final.pth'),'--output',str(ROOT/'paired_evaluation'/arm)]),
                ('video49',['/root/autodl-tmp/envs/motionproj/bin/python',str(ROOT/'code/inpaint_baseline.py'),'--root',str(ROOT/'paired_evaluation'),'--case',arm])]:
                status['phase']=arm+'_'+phase;save()
                with (ROOT/(status['phase']+'.log')).open('a') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
                    status['child']=dict(pid=p.pid,start_ticks=Path('/proc/%s/stat'%p.pid).read_text().split()[21],command=cmd);save()
                    assert p.wait()==0,status['phase']
                status.pop('child');status['completed'].append(status['phase']);save()
        status.update(phase='paired_outputs_report_pending',evaluation_complete=True);save()
    except Exception:
        status.update(phase='engineering_repair_needed',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
