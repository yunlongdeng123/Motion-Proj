"""汇总已完成实验的轻量证据；不运行模型。"""
from pathlib import Path
import sys, shutil, json, tarfile, datetime
sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read, dump

T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928')
out = T/'closeout_evidence'
out.mkdir(exist_ok=True)
files = {
    'r1': ['registration.json','identity_metadata.json','mask_state.json','mask_admission.json','drive_state.json','review_index.json','neighbor_audit.json','guard/summary.json'],
    'r2': ['registration.json','state.json','review/provenance.json'],
    'r3': ['proposal_registration.json','selected_command.json','registration.json','actor_layers/render_summary.json','review/summary.json'],
    'r4': ['registration.json','summary.json','data_audit.json','manifest.json'],
    'r5': ['registration.json','mask_state.json','shape_state.json','paint_state.json','canonical.json','ground_audit.json','orientation_registration.json','sequence_registration.json','actor_layers/render_summary.json','actor_layers/placement_checks.json','orientation180/actor_layers/render_summary.json','review_marked/summary.json'],
    'r6': ['registration_results.json'],
    'r7': ['registration.json','validation.json','actor_layers/render_summary.json','actor_layers/placement_checks.json','review/summary.json'],
}
copied, missing = [], []
for run, names in files.items():
    for name in names:
        src = T/run/name
        if not src.exists():
            missing.append(str(src.relative_to(T)))
            continue
        dst = out/run/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        copied.append(str(src.relative_to(T)))
for name in ['observations.json','validation.json','html_validation.json']:
    shutil.copy2(T/name,out/name)
report = dict(task_id=T.name, run_ids=['r'+str(i) for i in range(1,8)],
    status='bounded_batch_complete_with_failures_and_gates', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='3 preserved scenes +6 fixed new scenes; r30 geometry reconnection; INSERT; factual and grounding controls; MOVE screen; clean-RGB supervision candidates',
    new_scenes=6, generated_scenes=5, generated_unique_frames=150, input_blocked=['processed_756'],
    model_forwards=dict(driveeditor_windows=20,omega=10,hunyuan_shape=1,hunyuan_pbr=1),
    move_executed=False,training_steps=0,training_admitted=False,data_mask_clip_candidates=78,data_unique_camera_clips=28,
    human_verdict=None, preserved=['scene_0255 r18>r15 user verdict','r18','r21','r30','old GLBs','all new counterexamples'],
    failure_ledger_refs=['V77-F02'], failure_ledger_delta='updated V77-F02',
    conclusion='425 vehicle regeneration without other annotated vehicle overlap is actionable; 382 contains real hidden neighbor11; factual appearance and grounding not accepted; no MOVE claim',
    limitations=['GT camera/actor/LiDAR assistance','per-time B_t only','cross-camera DELETE incomplete','INSERT only10 frames','official000 raw log unknown','day/night data split imbalance','browser interaction untested'],
    next='Freeze this baseline; any next diagnosis converts held-out cases to development; first inspect425 conditioning and latent hole coverage; do not repeat seeds or unchanged PBR',
    copied=copied, optional_paths_missing=missing)
dump(T/'closeout.json',report)
shutil.copy2(T/'closeout.json',out/'closeout.json')
with tarfile.open(T/'closeout_evidence.tar','w') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file(): tar.add(p,arcname=str(p.relative_to(out)))
print('CLOSEOUT_EVIDENCE',len(copied),'MISSING_OPTIONAL',missing)
