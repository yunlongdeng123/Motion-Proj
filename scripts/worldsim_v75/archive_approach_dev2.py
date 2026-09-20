"""归档第二个接近任务的终态证据；原始视频/原生点图继续留在runs。"""
import json
from pathlib import Path
import shutil

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
GEN=ROOT/'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1'
REAL=ROOT/'WS-V75-APPROACH-BASELINE-02/20260920-r1'
SCREEN=ROOT/'WS-V75-BRAKING-DEV2-01/20260920-r1'
STATE=ROOT/'WS-V75-APPROACH-STATE-CONTROL-02/20260920-r1'
OUT=Path('/root/autodl-tmp/motion_proj/docs/autoresearch/worldsim_v75/approach_dev2')


def copy_files(source,dest,names):
    dest.mkdir(parents=True,exist_ok=True)
    for name in names:shutil.copy2(source/name,dest/name)


def main():
    assert not OUT.exists()
    assert json.loads((GEN/'queue_result.json').read_text())['status']=='complete'
    assert json.loads((STATE/'result.json').read_text())['status']=='complete'
    copy_files(SCREEN,OUT/'screen',['protocol.json','result.json'])
    copy_files(REAL,OUT/'real',['source.json','protocol.json','result.json','tracker_at_t0.json','real-policy-review.jpg'])
    copy_files(GEN,OUT/'generated',['protocol.json','frozen_utc.txt','error_arm_protocol.json','queue_result.json'])
    copy_files(GEN/'conditioning',OUT/'conditioning',['result.json'])
    for arm in ['gt_clean','dvgt_metric','dvgt_lidar_scaled','reference_lidar']:
        copy_files(GEN/arm,OUT/'generated'/arm,['result.json','decisions.json','dense_reference_result.json'])
    copy_files(GEN/'reconstruction',OUT/'reconstruction',[
        'protocol.json','inference_result.json','geometry_audit.json','readout_ray_control_result.json',
        'reference_supplement_protocol.json','reference_supplement_result.json','support-overlay_ray_control.png'])
    copy_files(GEN/'review',OUT,['comparison.json','closed-loop-results.png','closed-loop-results.svg','actual-policy-inputs.jpg'])
    copy_files(STATE,OUT/'state_control',[p.name for p in STATE.iterdir() if p.suffix in ['.json','.npz']])
    provenance={'status':'complete','source_screen':str(SCREEN),'source_real':str(REAL),'source_generated':str(GEN),'source_state_control':str(STATE),
                'scope':'second braking development task, previously exposed log; not independent confirmation; same seed42/117frames/15actions',
                'resources':{'frozen_logs':6,'frozen_task_windows':18,'selected_tasks':1,'real_detector_calls':36,'dvgt_forward_calls':1,
                             'complete_generation_runs':4,'generated_frames':468,'new_direct_state_frames':468,'command_replay_frames':468},
                'baseline_policy':'world-distance association; rejected IoU control was NOT adopted',
                'reference':'original5-point rejection preserved; same bounded3-past-scan reference protocol, no GT motion compensation',
                'raw_assets_retained':'native arrays, conditions, generated RGB, videos and meshes remain at source run paths',
                'human_verdict':None,'failure_ledger_delta':'none'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for p in OUT.rglob('*.svg'):
        p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    print(OUT)


if __name__=='__main__':main()
