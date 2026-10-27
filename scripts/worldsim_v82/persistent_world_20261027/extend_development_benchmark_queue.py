"""One-shot guarded extension: eight-family diagnostics before next long base.

Only known zero-update, childless waiting native parents may be replaced.
Never signals a GPU child, active training or any other task.
"""
import json,os,signal,subprocess,time
from pathlib import Path
from native_scene_controller import ROOT,PY,same_process,ticks,write
from resource_control import require_start


def main():
    require_start()
    jobs=[
      ('10096619443888687526_2820_000_2840_000',55558,'903692775','473735159277431842_630_095_650_095'),
      ('2555987917096562599_1620_000_1640_000',62344,'903867759','10096619443888687526_2820_000_2840_000'),
      ('13186511704021307558_2000_000_2020_000',83304,'904466223','2555987917096562599_1620_000_1640_000')]
    benchmarks=[row[3] for row in jobs]
    destination=ROOT/'benchmark_interleaving_extension_v1';destination.mkdir(exist_ok=False)
    original={}
    # Validate *all* parents before the first signal or launch.
    for context,pid,start,previous in jobs:
        folder=ROOT/'native_scene_training_v1'/context;state=json.loads((folder/'status.json').read_text())
        assert state['pid']==pid and str(state['start_ticks'])==start and same_process(pid,start)
        assert state['phase']=='waiting_previous_scene_diagnostics' and state['after_motion_context']==previous and not state.get('child')
        assert not list(folder.glob('*.log')) and not any((folder/x).exists() for x in ['sky','smoke_fresh','smoke_restore','formal30k'])
        original[context]=state
    for context in benchmarks:
        assert not (ROOT/'benchmark_execution_v1'/context).exists()
        selected=json.loads((ROOT/'motion_scene_inputs_v1'/context/'native_preparation/selection_reviewed.json').read_text())
        request=json.loads((ROOT/'benchmark_cases_v1/scenes'/context/'requests/vehicle_removal.json').read_text())
        assert selected['track']==request['target_track'] and selected['status']=='agent_visually_reviewed'
        assert request['split']!='heldout_reserved'
    write(destination/'before.json',dict(native_waiters=original,benchmarks=benchmarks,training_updates_lost=0))
    dispatched=[]
    def spawn(command,label):
        with (destination/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        item=dict(pid=child.pid,start_ticks=ticks(child.pid),command=command)
        dispatched.append(item);write(destination/'dispatch_progress.json',dict(processes=dispatched))
        return item
    for context in benchmarks:
        spawn([PY,str(ROOT/'code/benchmark_suite_controller.py'),'--context',context],'benchmark_'+context)
    for context,pid,start,previous in jobs:
        require_start();state=json.loads((ROOT/'native_scene_training_v1'/context/'status.json').read_text())
        assert state==original[context] and same_process(pid,start) and not state.get('child')
        os.kill(pid,signal.SIGTERM)
        end=time.monotonic()+5
        while same_process(pid,start) and time.monotonic()<end:time.sleep(.1)
        assert not same_process(pid,start)
        spawn([PY,str(ROOT/'code/native_scene_controller.py'),'--context',context,'--after-motion',previous,'--after-benchmark',previous,'--mask-first-count',str(original[context].get('mask_first_count',0)),'--resume-waiting'],'native_'+context)
    write(destination/'receipt.json',dict(status='zero_work_waiters_rearmed_with_full_development_diagnostic_dependency',processes=dispatched,
         old_waiters=original,healthy_scene3_training_signalled=False,training_updates_lost=0,
         new_development_benchmark_contexts=benchmarks,heldout_editing_started=False,quality_accepted=False,human_verdict=None))
    print(json.dumps(dispatched))


if __name__=='__main__':main()
