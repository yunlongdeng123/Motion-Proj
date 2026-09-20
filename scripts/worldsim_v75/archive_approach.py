"""保留原始任务目录，归档本轮轻量协议、工程修复和四组闭环证据。"""
import json,shutil
from pathlib import Path


def main():
    P=Path('/root/autodl-tmp/motion_proj'); R=Path('/root/autodl-tmp/runs/worldsim_v75')
    root=R/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2'
    assert json.loads((root/'queue_result.json').read_text())['status']=='complete'
    out=P/'docs/autoresearch/worldsim_v75/approach'; assert not out.exists(); out.mkdir(parents=True)
    sources={
      'raster':R/'WS-V75-RASTER-POLICY-01/20260920-r1',
      'real_r1':R/'WS-V75-APPROACH-BASELINE-01/20260920-r1',
      'real_r2':R/'WS-V75-APPROACH-BASELINE-01/20260920-association-r2',
      'generated_r1':R/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-r1',
      'generated_r2':root}
    def copy(src,dst):
        if src.exists(): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for label,src in sources.items():
        for name in ['protocol.json','result.json','tracker_at_t0.json','frozen_utc.txt','queue_result.json','error_arm_protocol.json',
                     'association_audit.json','association_repair_verification.json','preflight_timestamp_assertion.py']:
            copy(src/name,out/label/name)
        for arm in ['gt_clean','dvgt_metric','dvgt_lidar_scaled','reference_lidar']:
            for name in ['result.json','decisions.json','dense_reference_result.json']:
                copy(src/arm/name,out/label/arm/name)
    for name in ['protocol.json','inference_result.json','readout_ray_control_result.json','geometry_audit.json',
                 'reference_supplement_protocol.json','reference_supplement_result.json','support-overlay_ray_control.png']:
        copy(root/'reconstruction'/name,out/'reconstruction'/name)
    for name in ['comparison.json','closed-loop-results.png','closed-loop-results.svg','actual-policy-inputs.jpg']:
        copy(root/'review'/name,out/name)
    B=Path('/root/autodl-tmp/backups/v75-approach-20260920')
    for name in ['ground_asset_check.json','verify_association_repair.py','rgb_idm_policy.pre-association.py',
                 'run_approach_closed_loop.r1.py','assess_following_closed_loop.py']:
        copy(B/name,out/'engineering'/name)
    provenance={'task_ids':['WS-V75-RASTER-POLICY-01','WS-V75-APPROACH-BASELINE-01','WS-V75-APPROACH-CLOSEDLOOP-01','WS-V75-APPROACH-REFERENCE-01'],
      'sources':{k:str(v) for k,v in sources.items()},'base_commit_before_this_turn':'db13cb14',
      'role':'single exposed development log; r1/r2 are engineering repair, not independent samples or seed replication',
      'generated_frames':585,'complete_generated_runs':5,'final_comparison_runs':4,'dvgt_forwards':1,'actual_detector_calls':36,
      'detector_note':'35 warmup/real baseline plus1 reconstruction input detector; saved5 observations reused for raster-only check',
      'extra_information':['known metric height raster for policy and ego ground snap','20 real history frames for policy initialization',
                           'known route/calibration, GT actor size/yaw and shared other trajectories','background LiDAR for scale arm',
                           'three past LiDAR sweeps for reference arm, source scan partial point-time filtered'],
      'engineering_repairs':['nonplanar road requires ground map, not extrapolated plane','fixed20 prior captures tolerate nanosecond timestamp jitter',
                             'distance-gated matching before Hungarian prevents valid-pair eviction'],
      'full_assets':'source directories retain generated/condition tensors, all videos, native DVGT outputs, per-frame inputs and logs',
      'failure_ledger_refs':['V74-H2-F22'],'failure_ledger_delta':'none','human_verdict':None}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n'); print(out)


if __name__=='__main__': main()
