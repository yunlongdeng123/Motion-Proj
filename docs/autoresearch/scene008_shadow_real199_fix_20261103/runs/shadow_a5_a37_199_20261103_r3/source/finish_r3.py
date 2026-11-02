"""Finish zero-update evaluation, exact state reload, video and CPU acceptance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,math,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
RUN=Path(__file__).resolve().parent.parent
R2=RUN.with_name('shadow_a5_a37_199_20261103_r2')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'

def main():
    while not (RUN/'formal/receipt.json').exists():
        time.sleep(5)
        status=Path('/proc/3020/stat')
        if not status.exists() or status.read_text().split()[2]=='Z':raise RuntimeError('Trainer exited before receipt')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    subprocess.run([PY,str(RUN/'source/r3_runtime.py'),'--output',str(RUN/'evaluation'),'--evaluate','--checkpoint',str(RUN/'formal/checkpoint001000.pt')],check=True,env=env)
    frames=RUN/'evaluation/all49';video=RUN/'delete5_before_after49.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','10','-start_number','89','-i',str(frames/'%03d_baseline_delete5.png'),'-framerate','10','-start_number','89','-i',str(frames/'%03d_repaired_delete5.png'),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-frames:v','49','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(video)]))
    assert int(probe['streams'][0]['nb_read_frames'])==49
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    loss=[json.loads(x) for x in (RUN/'formal/training.jsonl').read_text().splitlines()]
    assert [x['step'] for x in loss]==list(range(1,1001)) and all(math.isfinite(x['loss']) for x in loss)
    ck=torch.load(RUN/'formal/checkpoint001000.pt',map_location='cpu',weights_only=False)
    steps=[int(v['step']) for v in ck['optimizer']['state'].values()];assert set(steps)=={1000}
    assert all(torch.isfinite(v).all() for v in ck['repair'].values() if v.is_floating_point())
    exact={str(r):(frames/('%03d_repaired_delete5.png'%r)).read_bytes()==(RUN/'formal/step001000'/('%03d_repaired_delete5.png'%r)).read_bytes() for r in [101,105,113]};assert all(exact.values())
    metrics=json.loads((frames/'metrics.json').read_text());actors={}
    for alias in ['5','37']:
        records=[x['actors'][alias] for x in metrics['rows']]
        result={}
        for key,count in [('road_MAE','road_samples'),('shadow_MAE','shadow_samples')]:
            rows=[x for x in records if x[key] is not None];n=sum(x[count] for x in rows)
            b=sum(x['baseline_'+key]*x[count] for x in rows)/n;a=sum(x[key]*x[count] for x in rows)/n
            result[key]=dict(before=b,after=a,reduction=(b-a)/b,frames=len(rows),samples=n,
                improved_frames=sum(x[key]<x['baseline_'+key] for x in rows),frame_macro_after=float(np.mean([x[key] for x in rows])))
        actors[alias]=result
    receipt=dict(updates=1000,previous_failed_updates=dict(R1=1000,R2_native=1000),total_engineering_updates=3000,
        legacy_replay=json.loads((R2/'legacy_a5_replay/replay_receipt.json').read_text()),smoke_updates=0,restore_updates=0,
        saved_optimizer_steps=steps,finite_loss_rows=1000,restored_anchor_PNGs_exact=exact,frames_decoded=49,
        CUDA_peak_allocated=max(x['cuda_peak_allocated'] for x in loss),CUDA_peak_reserved=max(x['cuda_peak_reserved'] for x in loss),
        training_seconds=json.loads((RUN/'formal/receipt.json').read_text())['seconds'],actors=actors,
        outside_MAE_mean=float(np.mean([x['outside_MAE'] for x in metrics['rows']])),
        outside_MAE_max=max(x['outside_MAE'] for x in metrics['rows']),
        other_vehicle_MAE_mean=float(np.mean([x['other_vehicle_MAE'] for x in metrics['rows'] if x['other_vehicle_MAE'] is not None])),
        invariants=json.loads((RUN/'evaluation/invariants.json').read_text()),input_summary=json.loads((RUN/'inputs/receipt.json').read_text()),
        all_199_are_build_inputs=True,metric_reference_is_proxy=True,inferred_excluded_from_truth=True,
        quality_accepted=False,human_verdict=None,needs_visual_review=True,failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none')
    (RUN/'result_summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
