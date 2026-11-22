from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
run = root/'takeback/execution_r3/post_geometry_ablation'
metrics = run/'evaluation/geometry_ablation/metrics.json'
proof = json.loads((run/'proof.json').read_text(encoding='utf-8'))
evaluation = json.loads(metrics.read_text(encoding='utf-8'))
evaluation['vehicle_geometry_gate'].update(passed=False, grey_review='REJECTED_AFTER_SIX_VIEW_REVIEW')
evaluation['root_review'] = dict(verdict='REJECTED', metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
    metrics_path=str(Path(proof['checkpoint']['path']).parent.parent/'evaluation/geometry_ablation/metrics.json'),
    local_metrics_path=str(metrics), reviewed_views=[f'novel_{i:02d}' for i in range(6)],
    comparison='actual unified Gaussian expected-depth grey versus pure frozen TRELLIS in identical cameras',
    findings=['front wheel and wheel arch crossing strands in novel01/02',
              'rear window and tailgate surface anomalies in novel03',
              'remaining edge haze; shared scene occlusion and K changes also fail separately'],
    decision='No formal appearance or joint stage; preserve actual candidate and complete state.')
(root/'root_geometry_review.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2),encoding='utf-8')
result = dict(schema='a20.unified_reconstruction.final_iteration_receipt.v1',
    method='single a20 object-coordinate Gaussian field; source labels only for confidence and evidence',
    final_asset_status='geometry_candidate_rejected', initialized_rows=419005,
    initialized_by_source=dict(old=8925, TRELLIS=410080),
    final_active_by_source=dict(old=8114, TRELLIS=410080), final_active_rows=418194,
    total_global_retired_old_rows=811, TRELLIS_rows_resampled=0,
    selected_asset_formal_updates=40, selected_asset_postprocess_optimizer_updates=0,
    formal_updates_all_separate_trials=dict(r1_geometry=40,r2_geometry=40,r3_geometry=40),
    smoke_updates_each=dict(geometry=10, discarded_replay=2, discarded_appearance=1,discarded_joint=1),
    smoke_trials=3, formal_appearance_updates=0, formal_joint_updates=0,
    selected_actor=proof['actor'], full_state_checkpoint=proof['checkpoint'],
    exact_G40_Adam_restore=proof['exact_G40_Adam_restore'],
    saved_checkpoint_exact_reload=proof['saved_checkpoint_exact_reload'],
    metric=dict(points=246, definition='native alpha-normalized expected camera-ray range',
                initial_MAE_m=0.12461521955040412, final_MAE_m=evaluation['opaque_mae_m'],
                final_P90_m=evaluation['opaque_p90_m']),
    vehicle_geometry_passed=False, scene_handoff_passed=False,
    stop_reason='six-view geometry not accepted; no formal material stage',
    build_raws=[86,110], heldout_frames_opened=[], GPU_keep_on=True, power_operations=0)
(root/'final_iteration_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(final_active_rows=result['final_active_rows'],status=result['final_asset_status'])))
