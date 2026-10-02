"""r32：只改变显露任务的角色准入，复用87条30帧轨迹及全部像素。"""
from pathlib import Path
import sys,json,time,shutil
sys.path.insert(0,str(Path(__file__).parent))
import evaluate_instance_audit as audit
from reveal_roles import primary_and_preserved
T=audit.f.T;O=T/'r32'

def main():
    O.mkdir(exist_ok=True)
    plan={'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r32',
        'single_change':'require at least one main B to reveal; incidental B must retain geometry/identity/evidence checks, not be forced to >=30% occlusion',
        'source':'r28 fixed 87 trajectories, 30 scenes, 30 actual exposures',
        'mask_position_speed_identity_depth_ego_thresholds_changed':False,
        'all_affected_B_other_frame_support_min':.5,'main_B_max_occlusion_min':.30,
        'Y_SAM_role':'offline labels only; method conditions must be rebuilt from final masked RGB',
        'new_GPU_jobs':0,'training_steps':0,'stop_rule':'one paired audit; independently QA any newly eligible case',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    assert not (O/'run.json').exists();audit.f.dump(O/'run.json',plan)
    # 真实多B失败保留回归：不能用有充分证据的一辆车掩盖另一辆全程不可见的车。
    checks=[]
    for row in audit.f.read(T/'r31/gate_detail.json')['cases']:
        proc=row['process'];ratios={t:p['occlusion_fractions'] for t,p in proc['protected'].items()}
        roles,why=primary_and_preserved(ratios,proc)
        if row['case_id'] in {'Q035','Q046'}:assert roles is not None
        elif row['case_id'] in {'Q066','Q067','Q068'}:assert why=='insufficient_other_frame_evidence'
        checks.append({'case_id':row['case_id'],'roles':roles,'reason':why})
    audit.f.dump(O/'role_regression.json',{'cases':checks,'unchanged_pixels':True})
    # 旧默认分支在真实例上保持原结果，不改写任何r28产物。
    old=audit.f.read(T/'r28/instance_quality.json');before={c['case_id']:c for c in old['cases']}
    audit.main(output_root=O/'legacy_regression',case_ids=['Q035','Q046','Q060'])
    for c in audit.f.read(O/'legacy_regression/instance_quality.json')['cases']:
        for key in ['technical_candidate','reason','quality','identity_flags','scope_flags']:
            assert c[key]==before[c['case_id']][key],(c['case_id'],key)
    audit.main(output_root=O,reveal_policy='primary_plus_preserved')
    current=audit.f.read(O/'instance_quality.json')
    newset={r['case_id'] for r in current['cases'] if r['technical_candidate']}
    oldset={r['case_id'] for r in old['cases'] if r['technical_candidate']}
    audit.f.dump(O/'comparison.json',{'old_counts':old['counts'],'new_counts':current['counts'],
        'retained':sorted(newset&oldset),'new_candidates':sorted(newset-oldset),'lost':sorted(oldset-newset),
        'new_admission':False,'training_steps':0})
    audit.f.dump(O/'controller_state.json',{'stage':'complete_pending_independent_data_QA'})
    print('PRIMARY_REVEAL_CONTROL',sorted(newset-oldset),flush=True)

if __name__=='__main__':main()
