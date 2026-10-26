"""Wait for one scene30K, real GPU adapter entry check, same-state motion suite.

No case-specific training. Later native scene controllers wait outside the GPU
lock for this scene's first diagnostic round, so training cannot starve edits.
"""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
from resource_control import require_start,stop_requested
from native_scene_controller import same_process
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
VIDEO_PY='/root/autodl-tmp/envs/motionproj/bin/python'
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split()[21]
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--resume-failed',action='store_true');a=p.parse_args()
    dest=ROOT/'motion_suite_execution_v1'/a.context;attempt=1;previous=None
    if a.resume_failed:
        previous=json.loads((dest/'status.json').read_text());assert previous['phase']=='engineering_repair_needed_preserved'
        assert not same_process(previous['pid'],previous['start_ticks'])
        child=previous.get('child');assert not child or not same_process(child['pid'],child['start_ticks'])
        attempt=previous.get('attempt',1)+1;directory=dest/'engineering_attempts'/f'attempt_{attempt}'
        directory.mkdir(parents=True,exist_ok=False);write(directory/'previous_status.json',previous)
    else:dest.mkdir(parents=True,exist_ok=False);directory=dest
    run=ROOT/'native_scene_training_v1'/a.context/'formal30k';suite=ROOT/'motion_edit_suite_v1'/a.context
    assert (suite/'summary.json').exists()
    prepared=json.loads((suite/'summary.json').read_text())['prepared_single_requests'];prepared_ids={x['id'] for x in prepared}
    backend_cases=[x for x in ['delete_target','translate_lateral_1m','speed_075','speed_125','turn_left15','turn_right15'] if x in prepared_ids]
    status=dict(context=a.context,pid=os.getpid(),start_ticks=ticks(os.getpid()),phase='waiting_same_scene_complete30k',child=None,training_updates=0,completed_native_cases=0,completed_backend_cases=0,expected_native_cases=len(prepared),expected_backend_cases=len(backend_cases),attempt=attempt,native_suite_path=str(directory/'native_suite'),previous_failed_attempt=previous)
    def update(**kw):status.update(kw);write(dest/'status.json',status)
    def launch(phase,cmd):
        require_start();update(phase=phase,command=cmd)
        with (directory/(phase+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);update(child=dict(pid=child.pid,start_ticks=ticks(child.pid)))
            while child.poll() is None:
                if stop_requested():child.terminate();child.wait(timeout=45);raise RuntimeError('quota stop; partial evaluation preserved')
                time.sleep(3)
        update(child=None,returncode=child.returncode);assert child.returncode==0,f'{phase} entry failed, preserve for authorized engineering repair'
    update()
    try:
        while True:
            require_start();p=run/'returned.json'
            if p.exists():
                try:returned=json.loads(p.read_text())
                except json.JSONDecodeError:time.sleep(2);continue
                assert returned['reason']=='native_returned' and returned['iteration']==30000
                break
            time.sleep(5)
        base=[PY,str(ROOT/'code/render_motion_suite.py'),'--run',str(run),'--suite',str(suite)]
        launch('native_entry_check',base+['--output',str(directory/'native_entry_check'),'--single-frame'])
        check=json.loads((directory/'native_entry_check/summary.json').read_text());assert check['adapter_actual_native_max_difference']<=2e-6 and check['master_files_and_tensor_versions_unchanged']
        launch('native_suite',base+['--output',str(directory/'native_suite')])
        full=json.loads((directory/'native_suite/summary.json').read_text());assert full['frames']==31 and full['master_roundtrip_difference']==0
        update(completed_native_cases=full['single_request_variants'],continuous_native_executed=full.get('continuous_sequence_executed',False),continuous_native_revisions=full.get('continuous_revision_count',0))
        for case in backend_cases:
            require_start()
            launch('backend_'+case,[VIDEO_PY,str(ROOT/'code/inpaint_baseline.py'),'--root',str(directory/'native_suite'),'--case',case,'--input-kind','native_3d'])
            summary=json.loads((directory/'native_suite'/case/'propainter/summary.json').read_text());assert summary['frames']==31 and summary['input_kind']=='native_3d'
            update(completed_backend_cases=status['completed_backend_cases']+1)
        update(phase='single_request_diagnostics_complete_continuous_pending',quality_accepted=False,human_verdict=None,continuous_sequence_executed=False)
    except Exception:
        update(phase='resource_stop' if stop_requested() else 'engineering_repair_needed_preserved',error=traceback.format_exc());raise

if __name__=='__main__':main()
