"""r28有界控制器：CPU固定候选→补Y评价标签→精确输入检查；不启动训练。"""
from pathlib import Path
import os,json,time,fcntl,subprocess,traceback
O=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r28')
S=Path(__file__).parent


def read(p):return json.loads(p.read_text())
def save(d):
    p=O/'controller_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def main():
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'stage':'waiting_existing_CPU_preparation','pid':os.getpid(),'children':[],
           'training_steps':0,'Y_labels_for_evaluation_only':True,'method_SAM_not_automatically_started':True}
    save(state);deadline=time.time()+2400
    try:
        while True:
            try:prep=read(O/'preparation_state.json')
            except (FileNotFoundError,json.JSONDecodeError):prep={}
            if prep.get('stage')=='complete_pending_SAM':break
            if prep.get('pid'):
                os.kill(prep['pid'],0)
                stat=Path(f'/proc/{prep["pid"]}/stat').read_text().split(') ')[1].split()[0]
                if stat=='Z':raise RuntimeError('CPU准备已退出但未完成，检查prepare.log')
            if time.time()>deadline:raise TimeoutError('等待现有CPU准备超过40分钟')
            time.sleep(10)
        queue=read(O/'quality_labels/observed_queue.json')['jobs']
        state.update(total_quality_jobs=len(queue),reused_quality_jobs=len(read(O/'quality_labels/segmentation_state.json')['completed']))
        qs=O/'quality_labels/segmentation_state.json'
        if read(qs)['stage']!='complete_pending_identity_review':
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if gpu:raise RuntimeError('GPU已有计算进程，未启动重复SAM: '+gpu)
            with (O/'quality_labels/segmentation.log').open('a') as log:
                p=subprocess.Popen(['/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python','-u',str(S/'segment_retained.py'),'--root',str(O/'quality_labels')],stdout=log,stderr=subprocess.STDOUT)
                state.update(stage='Y_evaluation_SAM',children=[p.pid]);save(state)
                try:code=p.wait(timeout=3600)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
                if code:raise RuntimeError('Y评价SAM失败，见隔离目录日志')
        if not (O/'instance_quality.json').exists():
            with (O/'instance_quality.log').open('a') as log:
                p=subprocess.Popen(['/root/autodl-tmp/envs/motionproj/bin/python','-u',str(S/'evaluate_instance_audit.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
                state.update(stage='CPU_instance_quality',children=[p.pid]);save(state)
                try:code=p.wait(timeout=1200)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
                if code:raise RuntimeError('实例质量检查失败，见instance_quality.log')
        result=read(O/'instance_quality.json')
        state.update(stage='complete_pending_independent_candidate_review',children=[],counts=result['counts'],training_admission=0);save(state)
    except Exception as e:
        state.update(stage='engineering_error',error=repr(e),traceback=traceback.format_exc());save(state);raise


if __name__=='__main__':main()
