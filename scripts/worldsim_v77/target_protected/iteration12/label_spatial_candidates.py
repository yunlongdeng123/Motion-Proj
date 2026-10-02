"""只为已通过世界空间检查的r43候选补全Y实例质检；不自动训练。"""
from pathlib import Path
import sys,os,time,fcntl,subprocess,traceback
sys.path.insert(0,str(Path(__file__).parent))
import reveal_factory as f
import recover_instance_audit as recover
import evaluate_instance_audit as audit

O=f.T/'r43';ROOT=O/'factory';S=Path(__file__).parent


def setup():
    f.O=O;f.ROOT=ROOT;f.legacy.O=O;f.legacy.ROOT=ROOT
    recover.O=O;recover.p.O=O;audit.O=O
    return f.legacy.geometry()


def main():
    assert f.read(O/'controller_state.json')['stage']=='CPU_complete_pending_geometry_result'
    lock=open(O/'labels_controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    roster=f.read(O/'candidate_roster.json');assert roster['count']==len(roster['cases'])
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r43','stage':'feasible_candidates_full_instance_quality',
        'fixed_roster':str(O/'candidate_roster.json'),'count':roster['count'],'input':'complete Y only for offline labels and QA',
        'not_condition':'method masks and actor-state require separate final-H-erased input and independent QA2',
        'unchanged':'official SAM checkpoint/config/box-only max-area prompt/seed42; sam_full_v2 final H; r32 main+incidental roles',
        'budget':'only fixed spatial candidates, at most one SAM label per source and touched identity; one GPU run timeout1hour',
        'stop':'all labels and quality complete; no threshold changes; no training or model inference',
        'training_steps':0,'training_admission':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'gpu_plan.json').exists():assert f.read(O/'gpu_plan.json')==plan
    else:f.dump(O/'gpu_plan.json',plan)
    state={'stage':'prepare_final_H_and_label_queue','pid':os.getpid(),'started':time.time(),'training_steps':0,'children':[]}
    f.dump(O/'labels_state.json',state)
    try:
        g=setup()
        if not (O/'prepared.json').exists():recover.prepare(g,roster,recovery=False)
        jobs=f.read(O/'quality_labels/observed_queue.json')['jobs'];path=O/'quality_labels/segmentation_state.json'
        labels=f.read(path)
        if not jobs:
            labels.update(stage='complete_pending_identity_review');f.dump(path,labels)
        elif labels.get('stage')!='complete_pending_identity_review':
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:raise RuntimeError('GPU已有计算作业，不启动重复SAM: '+active)
            with (O/'quality_labels/segmentation.log').open('a') as log:
                child=subprocess.Popen(['/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python','-u',str(S/'segment_retained.py'),'--root',str(O/'quality_labels')],stdout=log,stderr=subprocess.STDOUT)
                state.update(stage='Y_quality_SAM',total_jobs=len(jobs),children=[child.pid]);f.dump(O/'labels_state.json',state)
                try:code=child.wait(timeout=3600)
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
                if code:raise RuntimeError('实例质检SAM失败，见segmentation.log')
        state.update(stage='CPU_exact_input_quality',children=[]);f.dump(O/'labels_state.json',state)
        if not (O/'instance_quality.json').exists():audit.main(output_root=O,reveal_policy='primary_plus_preserved')
        result=f.read(O/'instance_quality.json')
        state.update(stage='complete_pending_independent_QA',counts=result['counts'],seconds=time.time()-state['started'],training_admission=0)
        f.dump(O/'labels_state.json',state);print('GEOMETRY_FIRST_LABELS_COMPLETE',result['counts'],flush=True)
    except Exception as error:
        state.update(stage='engineering_error',error=repr(error),traceback=traceback.format_exc());f.dump(O/'labels_state.json',state);raise


if __name__=='__main__':main()
