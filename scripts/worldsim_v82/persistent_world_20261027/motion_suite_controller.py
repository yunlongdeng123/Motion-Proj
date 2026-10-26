"""Wait for one scene30K, real GPU adapter entry check, same-state motion suite.

No case-specific training. Later native scene controllers wait outside the GPU
lock for this scene's first diagnostic round, so training cannot starve edits.
"""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
from resource_control import require_start,stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
VIDEO_PY='/root/autodl-tmp/envs/motionproj/bin/python'
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split()[21]
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);a=p.parse_args()
    dest=ROOT/'motion_suite_execution_v1'/a.context;dest.mkdir(parents=True,exist_ok=False)
    run=ROOT/'native_scene_training_v1'/a.context/'formal30k';suite=ROOT/'motion_edit_suite_v1'/a.context
    assert (suite/'summary.json').exists()
    status=dict(context=a.context,pid=os.getpid(),start_ticks=ticks(os.getpid()),phase='waiting_same_scene_complete30k',child=None,training_updates=0,completed_native_cases=0,completed_backend_cases=0)
    def update(**kw):status.update(kw);write(dest/'status.json',status)
    def launch(phase,cmd):
        require_start();update(phase=phase,command=cmd)
        with (dest/(phase+'.log')).open('x') as log:
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
        launch('native_entry_check',base+['--output',str(dest/'native_entry_check'),'--single-frame'])
        check=json.loads((dest/'native_entry_check/summary.json').read_text());assert check['adapter_actual_native_max_difference']<=2e-6 and check['master_files_and_tensor_versions_unchanged']
        launch('native_suite',base+['--output',str(dest/'native_suite')])
        full=json.loads((dest/'native_suite/summary.json').read_text());assert full['frames']==31 and full['master_roundtrip_difference']==0
        update(completed_native_cases=full['single_request_variants'])
        for case in ['delete_target','translate_lateral_1m','speed_075','speed_125','turn_left15','turn_right15']:
            require_start()
            launch('backend_'+case,[VIDEO_PY,str(ROOT/'code/inpaint_baseline.py'),'--root',str(dest/'native_suite'),'--case',case,'--input-kind','native_3d'])
            summary=json.loads((dest/'native_suite'/case/'propainter/summary.json').read_text());assert summary['frames']==31 and summary['input_kind']=='native_3d'
            update(completed_backend_cases=status['completed_backend_cases']+1)
        update(phase='single_request_diagnostics_complete_continuous_pending',quality_accepted=False,human_verdict=None,continuous_sequence_executed=False)
    except Exception:
        update(phase='resource_stop' if stop_requested() else 'engineering_repair_needed_preserved',error=traceback.format_exc());raise

if __name__=='__main__':main()
