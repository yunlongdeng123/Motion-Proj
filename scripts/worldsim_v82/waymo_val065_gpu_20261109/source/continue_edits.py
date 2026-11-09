"""归属/道路删除与独立原生留出对照，有界队列且无电源操作。"""
from pathlib import Path
import subprocess,json,time,os
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python';CPU='/root/autodl-tmp/envs/motionproj/bin/python'
def read(p):return json.loads(p.read_text())
def status(stage,**kw):(ROOT/'edit_pipeline_status.json').write_text(json.dumps(dict(stage=stage,unix=time.time(),pid=os.getpid(),automatic_shutdown=False,**kw),indent=2))
def run(name,*cmd,py=PY):
    status(name)
    with (ROOT/(name+'.log')).open('x') as f:subprocess.run([py,'-B',*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
def main():
    status('WAIT_NATIVE_AND_V1_EXPORTS')
    deadline=time.monotonic()+7200
    while not (ROOT/'formal10k_v1/summary.json').exists():
        st=read(PREP/'native_pipeline_status.json')
        if st['stage']=='RECOVERABLE_FAILURE':raise RuntimeError(st)
        if time.monotonic()>deadline:raise RuntimeError('native timeout')
        time.sleep(5)
    run('semantic_all159',ROOT/'source/calibrate_semantic_ownership.py','--model',ROOT/'formal10k','--iteration',10000,'--output',ROOT/'semantic_all159')
    run('semantic_all_eval',ROOT/'source/render_experiment.py','--mode','semantic','--output','semantic_all_eval')
    run('world_road',ROOT/'source/prepare_world_road.py')
    run('road_gate32',ROOT/'source/fit_road_gates.py','--output','road_gate32','--steps',32)
    if (ROOT/'road_gate32/checkpoint000032.pt').exists():
        run('road_gates',ROOT/'source/fit_road_gates.py','--output','road_gates','--steps',1000,'--resume',ROOT/'road_gate32/checkpoint000032.pt')
        assert read(ROOT/'road_gates/restore.json')['exact_full_state']
    run('final_all_eval',ROOT/'source/render_experiment.py','--mode','final','--output','final_all_eval')
    # 留出RGB只在所有拟合完成后解码。单独目录不在训练source_path中。
    run('export_heldout',ROOT/'source/export_heldout.py',py=CPU)
    run('all_cameras',ROOT/'source/build_camera_manifest.py')
    run('final_heldout_eval',ROOT/'source/render_experiment.py','--mode','final','--output','final_heldout_eval','--heldout')
    held=list(range(79,119))
    run('native_heldout_v1',ROOT/'source/render_v1.py','--engine',ROOT/'engine','--model',ROOT/'formal10k','--output',ROOT/'native_heldout_v1','--camera-manifest',ROOT/'all_cameras.json','--raws',*held)
    run('native_heldout_delete',ROOT/'source/evaluate_native.py','--engine',ROOT/'engine','--api',ROOT/'api','--model',ROOT/'formal10k','--iteration',10000,'--output',ROOT/'native_heldout_delete','--camera-manifest',ROOT/'all_cameras.json','--selection',PREP/'selected_case.json','--raws',*held)
    status('GPU_EDITS_COMPLETE_PENDING_ANALYSIS_RETRIEVAL_REPORT_PUSH',formal_updates=10000,vehicle_completion_inserted=False,vehicle_reveal_evidence='none verified',human_verdict=None)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('RECOVERABLE_FAILURE',error=repr(e));raise
