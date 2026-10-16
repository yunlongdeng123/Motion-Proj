"""Record the selected run after image review; keep remaining goals explicit."""
from pathlib import Path
import json,datetime,hashlib,shutil
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/shadow_sequence_v2';WORK=BASE/'work/shadow_sequence_v2'
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
selected={'training':'unselected_clean_v1','evaluation':'final49_unselected_v1','actual_total_updates':18000,
 'selection_basis':'all49 comparison review plus native foreground/edge/edit checks; numerical metrics alone not used',
 'checkpoint_sha256':'501524d0bd257c00ad59b762e9e468cad2db47cbfadc31a3a2098d062d0a9320',
 'scope':'confidence-supported reconstructed road in this49-frame scene; overall problem only partially solved',
 'rejected_previous_candidate':'contribution_frozen_v1: unselected foreground red fragments in raw117–137',
 'reviewed_utc':now}
actual=hashlib.sha256((OUT/selected['training']/'final.pt').read_bytes()).hexdigest();assert actual==selected['checkpoint_sha256']
write(OUT/'selected.json',selected)
review={'status':'REVIEWED','evaluation':selected['evaluation'],'comparison_frames_reviewed':49,
 'comparison_pages':['page_1.jpg','page_2.jpg','page_3.jpg','page_4.jpg'],
 'native_original_view_frames_reviewed':[89,101,106,125,137],
 'native_anchor_edit_states_reviewed':{'translation':[12,36],'yaw30':[12,36]},
 'no_new_severe_road_or_protected_vehicle_artifact':True,
 'scope_of_above_gate':'supported road and protected original-view foreground; excludes known original asset yaw defects and outside-authority residuals',
 'red_fragments_previous_candidate':'VISUALLY_REJECTED; raw125 inspected before/zero-update separation/final refit; all49 final contact review shows removed red fragments',
 'remaining_failures':[
  {'where':'raw106–108, lower left image boundary','finding':'dark shard remains outside green recovery authority; whole-image clean-road goal NOT met'},
  {'where':'fixed raw101, edited yaw +30 and -30 degrees','finding':'severe pre-existing vehicle asset surface defects; editable asset quality NOT passed'},
  {'where':'outside reconstructed road','finding':'original temporal freedom and baked shadows remain; all-scene time invariance NOT passed'}],
 'all_stream_frames_manually_reviewed':False,'paired_edit_GT':False,'unseen_views_certified':False,
 'review_method':'all49 comparison contact frames plus listed native-size views; no claim of real-time human video playback',
 'reviewed_utc':now}
write(OUT/'evidence/visual_review.json',review)
process=json.loads((OUT/'evidence/completion_process_check.json').read_text());assert not any(j['alive'] for j in process['jobs'])
assert not process['compute_processes'].strip()
write(OUT/'recovery_manifest.json',{'selected':selected,'remote_root':'/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2',
 'python':'/root/autodl-tmp/envs/drivingeditor/bin/python','ssh_alias':'wm-vgpu-1011',
 'parent_checkpoint':'../rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt',
 'required_remote_data':['surface_guarded_data/surface.npz','guard_refinement_v2/overrides.npz','guard_refinement_v2/foreground_proxies.npz','guard_refinement_v2/fallback_boxes.json','unselected_foreground_audit/selection.pt','../shadow_BC_20261016_v1/implementation_deferred_v1','../shadow_BC_20261016_v1/selection_v1','../shadow_BC_20261016_v1/selection_plane_v1','../shadow_BC_20261016_v1/runs/deferred_1000_v1/R4/addon_001000.pt','../infra','../inputs/scene0008_089_137','../rigid_post30k/scene0008_089_137/evaluation'],
 'local_delivery_standalone_full_scene':False,'local_addon_checkpoint_present':True,
 'evaluation_command':'/root/autodl-tmp/envs/drivingeditor/bin/python implementation/evaluate_sequence.py --output <new-unused-output-directory> --restore unselected_clean_v1/final.pt',
 'command_working_directory':'remote_root','source_snapshot':'source_snapshots/unselected_clean_v1',
 'original_checkpoint_overwritten':False,'GPU_poweroff_requested_or_performed':False})
state=json.loads((WORK/'state.json').read_text(encoding='utf-8-sig'))
state.update(phase='final_delivery_validation',completed_training_updates=18000,new_training=selected['training'],new_evaluation=selected['evaluation'],
 next_training=None,pending_readonly=None,owned_workers_all_exited=True,updated_utc=now)
state['planned_repair_updates']={'background':500,'shadow':250,'joint':1250}
write(WORK/'state.json',state)
for p in WORK.glob('*.py'):shutil.copy2(p,OUT/'scripts'/p.name)
print('Selected18k run recorded; scope failures preserved; all workers finished.')
