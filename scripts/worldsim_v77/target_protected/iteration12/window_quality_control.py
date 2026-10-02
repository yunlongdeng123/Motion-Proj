"""r30单变量控制：固定87条轨迹，只检查20帧窗是否改善数据可用性。"""
from pathlib import Path
import sys,time,os
sys.path.insert(0,str(Path(__file__).parent))
import evaluate_instance_audit as e
f=e.f;O=f.T/'r30'


def main():
    O.mkdir(exist_ok=True)
    if (O/'window_comparison.json').exists():print('already complete');return
    run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r30','phase':'fixed_trajectory_window_length_control',
        'source_run':'r28','cases':87,'original_window':[0,30],'controlled_window':[5,25],
        'RGB_exposure_seconds':[2.9,1.9],'no_new_positions_speeds_masks_thresholds':True,
        'Y_annotation_context_frames':30,'Y_annotations_are_evaluation_only':True,
        'condition_and_model_must_use_only_20_frames_if_later_admitted':True,
        'training_steps':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert f.read(O/'run.json')==run
    else:f.dump(O/'run.json',run)
    f.dump(O/'controller_state.json',{'stage':'CPU_window_quality_control','pid':os.getpid(),'training_steps':0})
    before={c['case_id']:c for c in f.read(e.O/'instance_quality.json')['cases']}
    # 真实数据回归：30帧默认路径必须保留已知四种输出，包含唯一正例。
    control=O/'unchanged_30_frame_regression';ids=['Q001','Q002','Q023','Q060']
    if not (control/'instance_quality.json').exists():e.main(output_root=control,case_ids=ids)
    after_control=f.read(control/'instance_quality.json')['cases']
    for c in after_control:
        expected=before[c['case_id']]
        for key in ['technical_candidate','reason','quality','identity_flags','scope_flags']:
            assert c[key]==expected[key],(c['case_id'],key,'30帧结果被修改')
    if not (O/'instance_quality.json').exists():e.main(frame_start=5,frame_count=20,output_root=O)
    result=f.read(O/'instance_quality.json');new={c['case_id']:c for c in result['cases']}
    comparison={'original_counts':f.read(e.O/'instance_quality.json')['counts'],'controlled_counts':result['counts'],
        'new_technical_candidates':[cid for cid,r in new.items() if r['technical_candidate'] and not before[cid]['technical_candidate']],
        'retained_technical_candidates':[cid for cid,r in new.items() if r['technical_candidate'] and before[cid]['technical_candidate']],
        'lost_technical_candidates':[cid for cid,r in new.items() if not r['technical_candidate'] and before[cid]['technical_candidate']],
        'unchanged_30_frame_regression_passed':True,'independent_QA':'pending_for_new_positives','training_admission':0,
        'scope':'data quality control, not model quality; shortened evidence may hurt recoverability','human_verdict':None}
    f.dump(O/'window_comparison.json',comparison)
    f.dump(O/'controller_state.json',{'stage':'complete_pending_result_review','pid':os.getpid(),'training_steps':0})
    print('WINDOW_CONTROL',comparison,flush=True)

if __name__=='__main__':main()
