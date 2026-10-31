"""Reconcile changing completed-media snapshots without inventing P decisions."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
facts=load(P/'source_only_second_screen_facts.json');protocol=load(P/'source_only_screen_protocol.json');elig=load(P/'actor_eligibility.json')
actors={(c,t):a for c,rows in elig['actors'].items() for t,a in rows.items()}
assert len(actors)==1228 and len(elig['actors'])==15
assert all(v['formal_gate'] is None for a in actors.values() for v in a['operations'].values())
assert len(protocol['badcase_mapping'])==13 and all(v['current_actor_causal_failure_label'] is None for v in protocol['badcase_mapping'])
with (P/'preparation45_source_diagnostics.csv').open(encoding='utf-8-sig',newline='') as stream:csvrows=list(csv.DictReader(stream))
assert len(csvrows)==45
selected={(r['context_id'],r['track_id']) for r in csvrows}
assert selected=={key for key,a in actors.items() if a['in_original_preparation45']}
assert all(all(r['formal_'+q+'_gate']=='' for q in ['none','delete','shift','rotate']) for r in csvrows)
for f in protocol['source_hashes']:
    assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
audits=[load(p) for p in (P/'media_scene_audits').glob('*.json')]
audits.sort(key=lambda a:a['scene_key'])
snapshots=elig['metadata']['media_measurement_snapshots'];completed={s['context_id'] for s in snapshots}
for a in audits:
    assert a['all_nonzero_measured_frames_have_nonempty_independent_display'] and a['source_identity_tuple_unique_within_actor']
    assert a['per_frame_voxel_and_cap_recomputed']
    assert a['original_control_frames_checked']==a['original_control_frames_required']
    assert sum(x['actual_measured_LiDAR']['actual_measured_box_rows'] for key,x in actors.items() if key[0]==a['context_id'])==a['box_filtered_measured_rows']
numeric=load(P/'media212_cross_runtime_numeric_audit.json')
assert numeric['status']=='STRICT_2E_5_PARITY_FAIL_RETAINED'
assert numeric['by_sensor']['side_LiDAR']['cache_minus_windows32']['rows_over_strict_2e_5']==0
assert all(x['original_record_payload_CRC_matches'] for x in numeric['original_audit_record_payload_links'])
correction=load(P/'media035_color_frustum_counterexample.json')
assert correction['status']=='RESOLVED_SOURCE_COLOR_FRUSTUM_CHECK_PASS'
known=[a for a in actors.values() if a['actual_measured_LiDAR']['status']=='complete_decoded_range_image_box_rows']
pending=[a for a in actors.values() if a['actual_measured_LiDAR']['status']!='complete_decoded_range_image_box_rows']
assert all(a['actual_measured_LiDAR']['actual_measured_box_rows'] is None for a in pending)
strict_pass=[a for a in audits if a['status']=='PASS_STORED_POINT_IDENTITIES_AND_THREE_ORIGINAL_RECORD_CONTROLS']
strict_fail=[a for a in audits if a['status']=='STORED_IDENTITIES_PASS_STRICT_COORDINATE_PARITY_FAIL']
assert all(a['context_id']=='14705303724557273004_3105_000_3125_000' for a in strict_fail)
complete=len(completed)==len(audits)==15 and not pending
files=[P/'actor_eligibility.json',P/'actor_eligibility.js',P/'preparation45_source_diagnostics.csv',P/'source_only_screen_protocol.json',
       P/'source_only_second_screen_facts.json',P/'media035_independent_audit.json',P/'media035_color_frustum_counterexample.json',P/'media212_cross_runtime_numeric_audit.json']
files+=sorted((P/'media_scene_audits').glob('*.json'))
result=dict(schema='independent_source_only_actor_diagnostics_review_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
    status='COMPLETE_STORED_IDENTITY_CHECKS_WITH_DECLARED_STRICT_CROSS_RUNTIME_PARITY_FAILURE' if complete else 'IN_PROGRESS_COMPLETED_MEDIA_CHECKS_ONLY',
    source_only=True,attachment_actions_not_executed=True,independent_edit_reference_or_student_output_used=False,
    registered_scenes=15,registered_vehicles=1228,original_preparation_queue45=45,operation_gate_records=4912,
    formal_gate_assigned_true=0,formal_gate_assigned_false=0,formal_unknown_actors=1228,formal_qualified_count=None,
    original_source_structural_checks=facts['all'],preparation45_source_checks=facts['preparation45'],
    completed_actual_media_scenes=len(completed),actors_with_complete_actual_measurements=len(known),actors_measurements_pending=len(pending),
    actually_measured_box_rows=sum(a['actual_measured_LiDAR']['actual_measured_box_rows'] for a in known),
    actors_with_measured_zero_box_rows=sum(a['actual_measured_LiDAR']['zero_measured_box_rows'] for a in known),
    preparation45_complete_actual_measurements=sum(actors[k]['actual_measured_LiDAR']['actual_measured_box_rows'] is not None for k in selected),
    audited_scenes=len(audits),audited_actors=sum(a['actors'] for a in audits),
    audited_box_rows=sum(a['box_filtered_measured_rows'] for a in audits),
    audited_independent_frame_display_rows=sum(a['independent_frame_display_rows'] for a in audits),
    labelled_actor_frame_observations_checked=sum(a['labelled_actor_frame_observations'] for a in audits),
    original_raw_control_frames_checked=sum(len(a['original_control_frames_checked']) for a in audits),
    strict_2e_5_coordinate_control_scene_passes=len(strict_pass),strict_2e_5_coordinate_control_scene_failures=len(strict_fail),
    cross_runtime_failure=dict(context_id=numeric['context_id'],status=numeric['status'],strict_tolerance_m=2e-5,
        cache_Windows32_metrics=numeric['metrics']['cache_minus_windows32'],
        cache_FP64_metrics=numeric['metrics']['cache_minus_reference64'],Windows32_FP64_metrics=numeric['metrics']['windows32_minus_reference64'],
        source_order_counts_and_original_control_CRC_links_verified=True,point_ids_nearest_neighbor_reassigned=False,
        interpretation='millimetre-scale arithmetic difference retained, not exact parity, not proof of box identity or capability failure',evidence='media212_cross_runtime_numeric_audit.json'),
    color_frustum_engineering_correction=dict(status=correction['status'],evidence='media035_independent_audit.json',
        old_color_backup_preserved=True,measured_geometry_and_source_ids_exactly_unchanged=True),
    explicit_limits=['all operation qualification remains unknown;45 is a preparation queue, not45 passes',
        'no_FRONT732 includes563 with other camera projected annotation; current videos are FRONT only',
        'annotation LiDAR counts and actual decoded box rows are separate fields and denominators',
        'actual box rows may include ground/occluders and repeated measurement identity across overlapping actors',
        'per-frame 6cm/cap512 and aggregate6cm/cap12000 affect display only; all measured rows remain inNPZ',
        'surface/view/background completeness and f_geom/f_view/f_bg are not inferred from sectors, points, centers or branch membership',
        'declared source49 window is not verified G training camera split',
        'three source geometry controls are not a full-sequence independent decoder replay',
        'bodymask inventory median0 does not establish every one of49 masks empty; positive-frame count unknown',
        'badcase1..13 taxonomy does not assign causal failure labels to these actors',
        'no model modified, noGPU/remote query by this reviewer, no human quality acceptance or browser interaction claim'],
    source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files])
(P/'independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','completed_actual_media_scenes','audited_scenes','audited_actors','audited_box_rows','original_raw_control_frames_checked','strict_2e_5_coordinate_control_scene_failures']}))
