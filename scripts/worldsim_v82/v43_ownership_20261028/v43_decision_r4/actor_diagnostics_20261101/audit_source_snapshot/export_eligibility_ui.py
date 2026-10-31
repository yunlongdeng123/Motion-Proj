"""Export script-loadable source facts; intentionally never assigns formal gates."""
import csv,hashlib,json,statistics
from pathlib import Path

DEST=Path(__file__).resolve().parent
facts=json.loads((DEST/'source_only_second_screen_facts.json').read_text(encoding='utf-8'))
media_by_context={};media_snapshots=[]
for path in sorted((DEST.parent/'media/generated/scenes').glob('*/manifest.json')):
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if not manifest.get('full_sequence') or manifest.get('LiDAR_status')=='pending':continue
    data_path=path.parent/'media.json';data=json.loads(data_path.read_text(encoding='utf-8'))
    if data.get('media_state')=='RGB_and_3D_boxes_complete_LiDAR_pending':continue
    context=data['context_id'];assert data['frame_count']==data['full_raw_frame_denominator']==manifest['raw_frame_count']
    assert len(data['actors'])==manifest['actors']
    assert all(a['lidar'].get('measured_points_before_voxel') is not None for a in data['actors'].values())
    assert sum(a['lidar']['measured_points_before_voxel'] for a in data['actors'].values())==manifest['measured_box_rows']
    media_by_context[context]=data
    media_snapshots.append(dict(context_id=context,media_path=str(data_path),media_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        manifest_path=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=data['frame_count'],actors=len(data['actors'])))
bycontext={};csvrows=[]
def measured_scope(trajectory,first,last):
    subset=[p for p in trajectory if first<=p['raw']<=last];counts=[p['measured_lidar_points'] for p in subset]
    return dict(first_raw_frame=first,last_raw_frame=last,raw_frame_denominator=last-first+1,
        labelled_frame_denominator=len(counts),missing_track_frames=last-first+1-len(counts),
        actual_measured_box_rows=sum(counts),source_frames=sum(n>0 for n in counts),
        median_measured_points=statistics.median(counts) if counts else None,
        zero_measured_labelled_frames=sum(n==0 for n in counts),zero_measured_box_rows=sum(counts)==0 if counts else None)
for r in facts['rows']:
    reasons=list(r['source_diagnostic_flags'])+list(r['engineering_invalid_reasons'])
    if not r['native_scene_checkpoint_inventory_present']:reasons.append('scene_G_not_in_existing_inventory')
    if not r['native_dynamic_asset_mapping_present']:reasons.append('target_dynamic_asset_not_mapped_static_candidates_still_possible')
    if r['mask_inventory'] is None:reasons.append('bodymask_inventory_missing')
    elif r['mask_inventory'].get('median_pixels')==0:reasons.append('bodymask_inventory_median_pixels_zero')
    if r['mask_inventory'] is not None:reasons.append('bodymask_positive_frame_count_unmeasured')
    prep=r['preparation_49frame_facts']
    if prep is not None and not all(prep['fixed_first_middle_last_anchor_label_presence'].values()):reasons.append('preparation49_fixed_anchor_source_pose_missing')
    media=media_by_context.get(r['context_id'])
    if media is None:
        measured=dict(status='pending_actual_range_image_decode',actual_measured_box_rows=None,source_frames=None,
            median_measured_points=None,zero_measured_box_rows=None,raw_frame_denominator=r['raw_frames_denominator'],
            labelled_frame_denominator=None,zero_measured_labelled_frames=None,per_frame_counts=None,
            recommended49=None,declared_native_window=None)
    else:
        actor=media['actors'][r['track_id']];counts=[p['measured_lidar_points'] for p in actor['trajectory']]
        assert sum(counts)==actor['lidar']['measured_points_before_voxel']
        assert sum(n>0 for n in counts)==actor['lidar']['source_frames']
        measured=dict(status='complete_decoded_range_image_box_rows',actual_measured_box_rows=sum(counts),
            source_frames=sum(n>0 for n in counts),median_measured_points=actor['lidar']['median_measured_points'],
            zero_measured_box_rows=sum(counts)==0,raw_frame_denominator=media['full_raw_frame_denominator'],
            labelled_frame_denominator=len(counts),zero_measured_labelled_frames=sum(n==0 for n in counts),
            per_frame_counts=[dict(raw=p['raw'],timestamp_micros=p['timestamp_micros'],actual_measured_box_rows=p['measured_lidar_points']) for p in actor['trajectory']],
            semantics='decoded sensor rows inside same-frame annotation box; may include ground/occluders; not instance/surface completeness',
            cloud_row_identity='context + raw_frame + deterministic decoded source_cloud_row',
            recommended49=measured_scope(actor['trajectory'],*r['preparation_window']) if r['preparation_window'] else None,
            declared_native_window=measured_scope(actor['trajectory'],*r['declared_native_G_window_inventory']) if r['declared_native_G_window_inventory'] else None)
        if measured['zero_measured_box_rows']:reasons.append('zero_actual_decoded_box_rows')
        if measured['zero_measured_labelled_frames']>0:reasons.append('some_labelled_frames_have_zero_actual_decoded_box_rows')
    common_unknown=['full_scene_G_none_geometry_and_depth_support_unmeasured','native_camera_time_world_model_binding_unvalidated_for_this_actor','candidate_U_and_shared_H0_not_validated_for_this_actor']
    operations={}
    for q in ['none','delete','shift','rotate']:
        unknown=list(common_unknown)
        if q in ['delete','shift']:unknown.append('historical_background_reveal_support_unmeasured')
        if q in ['shift','rotate']:unknown.append('required_surface_and_view_dependent_color_support_unmeasured')
        diagnostic=list(reasons)
        if q=='rotate' and r['rotate_plus30deg_missing_box_bearing_bins_proxy']:diagnostic.append('illustrative_rotate30_box_bearing_bins_missing_NOT_surface_test')
        operations[q]=dict(formal_gate=None,formal_gate_status='unknown',
                           observed=dict(source_box_records_finite_and_positive=not bool(r['engineering_invalid_reasons']),
                                         full_sequence_FRONT_projected_frames=r['full_sequence_facts']['FRONT_projected_frames'],
                                         full_sequence_any_camera_projected_frames=r['full_sequence_facts']['any_camera_projected_frames'],
                                         preparation49_FRONT_projected_frames=None if r['preparation_49frame_facts'] is None else r['preparation_49frame_facts']['FRONT_projected_frames'],
                                         actual_measured_box_rows=measured['actual_measured_box_rows'],actual_measured_source_frames=measured['source_frames']),
                           f_geom=None,f_view=None,f_bg=None,f_motion=None,
                           diagnostic_reasons=diagnostic,unknown_reasons=unknown,
                           execution_blockers=list(r['engineering_invalid_reasons']),
                           physical_capability_failure=None,evaluation_only_capability_bin=None)
    mask=r['mask_inventory']
    subject=dict(schema='actor_preedit_source_diagnostics_v1',context_id=r['context_id'],track_id=r['track_id'],
                 scene=r['scene'],scene_key=r['scene_key'],actor_alias=r['actor_alias'],in_original_preparation45=r['in_preparation45'],
                 formal_gate=None,formal_gate_status='unknown',source_role='annotation_assisted_preedit_source_diagnostics',
                 full_sequence=r['full_sequence_facts'],recommended49=r['preparation_49frame_facts'],
                 declared_native_window_source=r['declared_native_window_source_facts'],
                 existing_G_inventory=dict(scene_checkpoint_recorded=r['native_scene_checkpoint_inventory_present'],
                    declared_raw_window=r['declared_native_G_window_inventory'],training_camera_split_verified=False,
                    target_dynamic_asset_mapped=r['native_dynamic_asset_mapping_present'],actor_index=r['dynamic_actor_index_inventory'],
                    note='missing dynamic mapping does not exclude misplaced static vehicle Gaussian candidates'),
                 bodymask_inventory=dict(file_count=None if mask is None else mask['frames'],
                    median_pixels=None if mask is None else mask['median_pixels'],positive_frame_count=None,
                    source=None if mask is None else mask['source'],
                    median_zero_implies_at_least_zero_frames=None if mask is None or mask['median_pixels']!=0 else (mask['frames']+1)//2,
                    per_frame_positive_support_verified=False),
                 actual_measured_LiDAR=measured,
                 views_proxy=dict(FRONT_full_sequence_sectors=r['FRONT_source_12sector_bins'],
                    source_bank_role=r['observation_bank_role'],source_bank_sectors=r['observation_bank_bins'],
                    illustrative_rotate30_required_sectors=r['rotate_plus30deg_required_box_bearing_bins_proxy'],
                    illustrative_rotate30_missing_sectors=r['rotate_plus30deg_missing_box_bearing_bins_proxy'],
                    illustrates_previous_positive30degree_yaw_only=True,
                    surface_complete=None,actual_G_training_support=None,
                    note=r['proxy_view_scope']),
                 source_kinematics=dict(max_xy_finite_difference_speed_mps=r['max_xy_finite_difference_speed_mps'],
                    max_abs_heading_rate_radps=r['max_abs_heading_rate_radps'],
                    max_frame_gap=r['source_max_frame_gap'],
                    threshold_decision=None),
                 operations=operations,diagnostic_reasons=reasons,
                 source_GT_or_edit_result_read=False)
    bycontext.setdefault(r['context_id'],{})[r['track_id']]=subject
    if r['in_preparation45']:
        f=r['full_sequence_facts'];w=r['preparation_49frame_facts']
        csvrows.append(dict(scene=r['scene'],scene_key=r['scene_key'],actor_alias=r['actor_alias'],context_id=r['context_id'],track_id=r['track_id'],
                queue_role='original_preparation45_NOT_45_passes',raw_frame_denominator=f['frame_denominator'],raw_labelled_frames=f['labelled_frames'],
                full_FRONT_frames=f['FRONT_projected_frames'],full_any_camera_frames=f['any_camera_projected_frames'],
                recommended_first=w['first_raw_frame'],recommended_last=w['last_raw_frame'],recommended_frame_denominator=w['frame_denominator'],recommended_FRONT_frames=w['FRONT_projected_frames'],
                recommended_missing_track_frames=w['missing_track_frames'],recommended_fixed_anchor_label_presence=json.dumps(w['fixed_first_middle_last_anchor_label_presence'],separators=(',',':')),
                recommended_any_camera_frames=w['any_camera_projected_frames'],recommended_LiDAR_annotation_positive_frames=w['annotation_LiDAR_positive_labelled_frames'],
                FRONT_sector_count=len(r['FRONT_source_12sector_bins']),
                illustrative_rotate30_missing_sector_count=len(r['rotate_plus30deg_missing_box_bearing_bins_proxy']),
                existing_scene_G_inventory=r['native_scene_checkpoint_inventory_present'],target_dynamic_mapping=r['native_dynamic_asset_mapping_present'],
                bodymask_file_count=None if mask is None else mask['frames'],bodymask_median_pixels=None if mask is None else mask['median_pixels'],positive_mask_frames=None,
                actual_LiDAR_status=measured['status'],actual_measured_box_rows=measured['actual_measured_box_rows'],
                actual_measured_source_frames=measured['source_frames'],actual_median_measured_points=measured['median_measured_points'],
                actual_zero_measured_box_rows=measured['zero_measured_box_rows'],actual_labelled_frame_denominator=measured['labelled_frame_denominator'],
                actual_zero_measured_labelled_frames=measured['zero_measured_labelled_frames'],
                actual_recommended49_measured_rows=None if measured['recommended49'] is None else measured['recommended49']['actual_measured_box_rows'],
                actual_recommended49_positive_frames=None if measured['recommended49'] is None else measured['recommended49']['source_frames'],
                actual_recommended49_labelled_frames=None if measured['recommended49'] is None else measured['recommended49']['labelled_frame_denominator'],
                actual_recommended49_median_points=None if measured['recommended49'] is None else measured['recommended49']['median_measured_points'],
                formal_none_gate=None,formal_delete_gate=None,formal_shift_gate=None,formal_rotate_gate=None,
                formal_status='unknown_uncalibrated',diagnostic_reasons=';'.join(reasons)))

assert len(bycontext)==15 and sum(map(len,bycontext.values()))==1228 and len(csvrows)==45
assert all(o['formal_gate'] is None for ctx in bycontext.values() for a in ctx.values() for o in a['operations'].values())
meta=dict(schema='actor_preedit_source_diagnostics_bundle_v1',registered_actors=1228,registered_scenes=15,
          original_preparation45=45,formal_known_gate_count=0,formal_unknown_actor_count=1228,
          measured_scope='source annotation export structural and observation facts only',
          operation_gate_count=4912,formal_known_operation_gate_count=0,formal_qualified_actor_count=None,
          no_FRONT_means_no_other_camera_evidence=False,
          box_or_sector_means_surface_complete=False,edit_reference_used=False,edited_result_used=False,
          complete_actual_LiDAR_media_scenes=len(media_by_context),complete_actual_LiDAR_media_actors=sum(len(m['actors']) for m in media_by_context.values()),
          media_measurement_snapshots=media_snapshots,
          all_summary=facts['all'],preparation45_summary=facts['preparation45'])
known=[a for ctx in bycontext.values() for a in ctx.values() if a['actual_measured_LiDAR']['actual_measured_box_rows'] is not None]
facts['actual_range_image_layer']=dict(schema='additive_decoded_box_measurement_layer_v1',
    complete_scenes=len(media_by_context),complete_actors=len(known),pending_actors=1228-len(known),
    actual_measured_box_rows=sum(a['actual_measured_LiDAR']['actual_measured_box_rows'] for a in known),
    actors_measured_zero_box_rows=sum(a['actual_measured_LiDAR']['zero_measured_box_rows'] for a in known),
    annotation_fields_unchanged=True,formal_P=None,source_snapshots=media_snapshots,
    meaning='actual decoded range-image rows inside same-frame source annotation box; not a semantic/body-surface completeness measure')
for row in facts['rows']:
    measured=bycontext[row['context_id']][row['track_id']]['actual_measured_LiDAR']
    row['actual_range_image_box_measurement']={key:measured[key] for key in ['status','actual_measured_box_rows','source_frames','median_measured_points',
        'zero_measured_box_rows','raw_frame_denominator','labelled_frame_denominator','zero_measured_labelled_frames','recommended49','declared_native_window']}
(DEST/'source_only_second_screen_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(DEST/'actor_eligibility.json').write_text(json.dumps(dict(metadata=meta,actors=bycontext),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(DEST/'actor_eligibility.js').write_text('window.ACTOR_ELIGIBILITY_META='+json.dumps(meta,ensure_ascii=False,separators=(',',':'))+';\nwindow.ACTOR_ELIGIBILITY='+json.dumps(bycontext,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
with (DEST/'preparation45_source_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
(DEST/'preparation45_second_screen.csv').write_bytes((DEST/'preparation45_source_diagnostics.csv').read_bytes())
print(json.dumps(dict(actors=1228,contexts=15,operations=4912,formal_passes=None,formal_unknown_actors=1228,csv_rows=45,
                     noFRONT_but_other_camera=sum(a['full_sequence']['FRONT_projected_frames']==0 and a['full_sequence']['any_camera_projected_frames']>0 for c in bycontext.values() for a in c.values()))))
