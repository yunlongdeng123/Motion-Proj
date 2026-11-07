from pathlib import Path
import hashlib,json,shutil,time
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_v44_full5_integration_20261107_r1';OLD=B/'scene008_v44_rgb_asset_20261107_r1'
MODEL=B/'scene008_full_dynamic_edit_20261107_r1/inference10k'
C=ROOT/'contract';C.mkdir(parents=True,exist_ok=True)
assert not (C/'execution_contract.json').exists()
cfg=json.loads((MODEL/'effective_config.json').read_text())
source=Path(cfg['source_path'])/'transform_matrix.txt';dest=C/'full5_transform_matrix.txt'
assert not dest.exists();shutil.copy2(source,dest);assert source.read_bytes()==dest.read_bytes()
def ident(p):return dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
contract=dict(schema='scene008.v44_full5_integration.paired.v1',status='SEALED_BEFORE_CANDIDATE_GPU',created_unix=time.time(),
 baseline=dict(root=str(OLD),input_contract=str(OLD/'contract/input_contract.json'),cached_formal=str(OLD/'results/full49_r1/v44'),
  replay=str(ROOT/'baseline4/v44'),root_exact_audit=str(C/'root_baseline4_exact_png_audit.json'),forbidden_substitute='scene008_road_r0_v44_20261107_r1'),
 candidate=dict(full5_model=str(MODEL),iteration=10000,requested_source_commit='50d6daa8',retained_actor_ids=[5,18,26],
  full5_transform_path=str(dest),transform_source=ident(source),transform_copy=ident(dest),identities_from_checkpoint_only=True,
  no_original_full5_PLY_or_mask_for_selection=True,baseline_static_road_gates_env_C2_frozen=True,shared_single_depth_sort=True),
 pilot_raws=[105,113,130,137],states=['keep','delete'],new_training_updates=0,new_background_solves=0,pixel_composite=False,
 domains=dict(a5='original observations actor_labels==1',upper='original a5 y <= min_y + .25*(max_y-min_y)',
  full49_reference=dict(full_pixels=121128,upper_pixels=25906,full_mask_sha256='aa8e7d3b56a8c35ce6177ad1e3378df7afcbc83e6db5d813bdddbfc55a717419',
    upper_index_sha256='d5c24aeee22bb804311e28f89de6f0652438465263c3e45a2f9d245d6b67379f',gt_sha256='e1865e95d0f6e549bb104f7cec64f3a339bb23cb824ed471b269c2c6b8c0079a'),
  pilot_a5_pixels={'105':10773,'113':0,'130':0,'137':0},pilot_upper_pixels={'105':2091,'113':0,'130':0,'137':0},
  a6='original obs.visible[0]',a6_depth='original obs.depth_valid[0], target obs.depth_m[0] metres',
  a37='original obs.visible[1], KEEP only',road_keep='original obs.road',
  road_delete='original obs.road_edit intersect fixed_reference.depth_m[:,3]>0',
  road_targets='observed RGB for KEEP, original road_edit_rgb reobserved proxy for DELETE; not hidden ground truth',
  protect='original obs.protect minus original a6/a37 visible',
  blue_absolute='abs((B-targetB)-.5*((R-targetR)+(G-targetG))) on same original road domain; no predicted-alpha selection',
  original_blue_roi={'113':[276,175,309,220],'137':[276,175,309,220]},original_blue_roi_empty_is_null=True),
 numeric_gates=dict(source='Root pre-registration confirmed before candidate; original V44 evaluator had statistics only',
  a6_RGB_worst_frame_delta_max=.001,a6_depth_m_worst_frame_delta_max=.01,KEEP_a37_RGB_worst_frame_delta_max=.001,
  road_RGB_pooled_delta_max=.001,road_RGB_worst_frame_delta_max=.005,
  protected_RGB_pooled_delta_max=.001,protected_RGB_worst_frame_delta_max=.005,
  fixed_road_absolute_blue_pooled_delta_max=.001,fixed_road_absolute_blue_worst_frame_delta_max=.005,
  raw105_a5_mask_MAE_delta_strictly_below=0.,upper_MAE_report_without_visual_override=True,
  negative_deltas_are_improvement=True,pooled_weighting='sum/count per state, never frame mean',
  inherited_alpha_conservation_maxabs_strict_below=2e-5),
 exact_guards=dict(baseline_png_replay_max=0,shared_physical_rows_and_colours_byte_equal=True,shared_env_byte_equal=True,
  shared_fields_exclude_retagged_diagnostic_features=True,keep_delete_changes_only_a37=True,delete_a37_joint_group_max=0.,
  added_owner_ids=[5,18,26],shared_source_versions_unchanged=True),
 visual_gates=dict(root_four_frame_review_required=True,raw105_roof_windshield_visible_recovery_required=True,
  static_potential_duplicate_must_be_annotated=True,no_quality_claim_from_alpha_or_owner_only=True),
 all49_release=dict(automatic=False,requires_root_explicit_four_frame_images_and_guard_approval=True),
 shutdown=False,reset_card=False)
(C/'execution_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=contract['status'],path=str(C/'execution_contract.json'),sha256=hashlib.sha256((C/'execution_contract.json').read_bytes()).hexdigest(),transform=ident(dest))))
