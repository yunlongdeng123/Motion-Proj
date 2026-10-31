"""Source-only second-screen facts. No reference/edit-result inputs or P labels."""
import gzip,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/V43_R4_GPU_20261031'
DEST=Path(__file__).resolve().parent


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):return bool(np.isfinite(np.asarray(value,dtype=float)).all())
def scope_facts(records,first,last):
    subset=[r for r in records if first<=r['frame_index']<=last];front=[r for r in subset if r['front_bbox_xyxy'] is not None]
    percam={str(c):sum(c in r['camera_names'] for r in subset) for c in range(1,6)}
    return dict(first_raw_frame=first,last_raw_frame=last,frame_denominator=last-first+1,
                fixed_first_middle_last_anchor_label_presence={str(i):any(r['frame_index']==i for r in subset) for i in [first,(first+last)//2,last]},
                labelled_frames=len(subset),missing_track_frames=last-first+1-len(subset),
                FRONT_projected_frames=len(front),any_camera_projected_frames=sum(bool(r['camera_names']) for r in subset),
                projected_frames_by_camera=percam,
                view_sectors_by_camera={str(c):sorted({int(r['view_12sector_bins_by_camera'][str(c)]) for r in subset if str(c) in r['view_12sector_bins_by_camera']}) for c in range(1,6)},
                annotation_LiDAR_positive_labelled_frames=sum(r['annotation_lidar_points_in_box']>0 for r in subset),
                annotation_LiDAR_positive_FRONT_frames=sum(r['annotation_lidar_points_in_box']>0 for r in front),
                annotation_LiDAR_count_median_on_labelled_frames=float(np.median([r['annotation_lidar_points_in_box'] for r in subset])) if subset else None,
                finite_box_center_heading_size_records=sum(finite(r['center_world_m']+[r['heading_world_rad']]+r['size_lwh_m']) for r in subset),
                strictly_positive_lwh_records=sum(all(v>0 for v in r['size_lwh_m']) for r in subset),
                timestamp_count=len(subset),timestamp_interval_denominator=max(0,len(subset)-1),
                timestamps_strictly_increasing=all(b['timestamp_micros']>a['timestamp_micros'] for a,b in zip(subset,subset[1:])),
                minimum_timestamp_micros=min([r['timestamp_micros'] for r in subset],default=None),
                maximum_timestamp_micros=max([r['timestamp_micros'] for r in subset],default=None),
                source_track_gap_count=sum(b['frame_index']>a['frame_index']+1 for a,b in zip(subset,subset[1:])))
registry_path=OUT/'actor_registry_named.json';queue_path=OUT/'actor_preparation_queue_named.json'
registry=load(registry_path)['rows'];queue=load(queue_path)['rows']
reg={(x['context_id'],x['track_id']):x for x in registry};qmap={(x['context_id'],x['track_id']):x for x in queue}
assert len(reg)==len(registry)==1228 and len(qmap)==len(queue)==45
folders=[p.parent for p in (OUT/'actor_screen_horizonforge').glob('scene_*/summary.json')]
folders += [p.parent for p in (OUT/'actor_screen_existing/existing_normalized_v1').glob('*/summary.json')]
source_receipts=[];rows=[];scene_stats=[]
for folder in sorted(folders):
    summary=load(folder/'summary.json');context=summary['context_id'];n=summary['frames'];byid=defaultdict(list)
    for name in ['summary.json','all_vehicle_observations.jsonl.gz']:
        p=folder/name;source_receipts.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    with gzip.open(folder/'all_vehicle_observations.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            obs=json.loads(line);byid[obs['track_id']].append(obs)
    assert len(byid)==summary['all_vehicle_tracks']
    for tid,obs in byid.items():
        key=(context,tid);assert key in reg
        item=reg[key];obs.sort(key=lambda x:x['frame_index']);indices=[o['frame_index'] for o in obs]
        stamps=[o['timestamp_micros'] for o in obs]
        engineering=[]
        if len(set(indices))!=len(indices):engineering.append('duplicate_track_frame_record')
        if not all(0<=i<n for i in indices):engineering.append('frame_outside_source_denominator')
        if not all(b>a for a,b in zip(stamps,stamps[1:])):engineering.append('nonmonotonic_or_duplicate_timestamp')
        for o in obs:
            if not finite(o['center_world_m']+[o['heading_world_rad']]+o['size_lwh_m']):engineering.append('nonfinite_source_pose_or_dimensions')
            if not all(v>0 for v in o['size_lwh_m']):engineering.append('nonpositive_box_dimension')
        assert len(obs)==item['track_labelled_frames']
        front=[o for o in obs if o['front_bbox_xyxy'] is not None]
        assert len(front)==item['front_projected_frames']
        source_bins=sorted({int(o['view_12sector_bins_by_camera']['1']) for o in front})
        assert source_bins==item['front_view_sectors_proxy']
        times=np.array(stamps,dtype=float)/1e6;xyz=np.array([o['center_world_m'] for o in obs]);yaw=np.unwrap([o['heading_world_rad'] for o in obs])
        speed=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)/np.diff(times) if len(obs)>1 else np.array([])
        yawrate=np.abs(np.diff(yaw))/np.diff(times) if len(obs)>1 else np.array([])
        warnings=[]
        if not front:warnings.append('zero_FRONT_projected_observations')
        if front and not any(o['annotation_lidar_points_in_box']>0 for o in front):warnings.append('zero_annotation_box_LiDAR_count_on_FRONT_observed_frames')
        gaps=[b-a for a,b in zip(indices,indices[1:])]
        if any(g>1 for g in gaps):warnings.append('source_track_has_frame_gaps')
        gwindow=item.get('native_GS_49_frame_window');window=None
        if key in qmap:
            q=qmap[key];window=[q['recommended_window_first'],q['recommended_window_last']]
        native_window=None if not gwindow else [gwindow['start_raw'],gwindow['end_raw']]
        if window and native_window and window!=native_window:warnings.append('preparation_window_differs_from_existing_G_training_window')
        bank=[o for o in front if native_window is None or native_window[0]<=o['frame_index']<=native_window[1]]
        bank_bins=sorted({int(o['view_12sector_bins_by_camera']['1']) for o in bank})
        target=[o for o in front if window is None or window[0]<=o['frame_index']<=window[1]]
        required_rotate_bins=sorted({(int(o['view_12sector_bins_by_camera']['1'])-1)%12 for o in target})
        unseen=sorted(set(required_rotate_bins)-set(bank_bins))
        native=item.get('native_asset_mapping')
        row=dict(scene=item['scene'],scene_key=item['scene_key'],actor_alias=item['actor_alias'],context_id=context,track_id=tid,
                 in_preparation45=key in qmap,raw_frames_denominator=n,labelled_frames=len(obs),missing_track_frames=n-len(obs),
                 front_projected_frames=len(front),front_LiDAR_annotation_positive_frames=sum(o['annotation_lidar_points_in_box']>0 for o in front),
                 engineering_invalid_reasons=sorted(set(engineering)),source_diagnostic_flags=warnings,
                 source_max_frame_gap=max(gaps) if gaps else None,
                 max_xy_finite_difference_speed_mps=float(speed.max()) if len(speed) else None,
                 max_abs_heading_rate_radps=float(yawrate.max()) if len(yawrate) else None,
                 FRONT_source_12sector_bins=source_bins,FRONT_source_distinct_sector_count=len(source_bins),
                 declared_native_G_window_inventory=native_window,preparation_window=window,
                 full_sequence_facts=scope_facts(obs,0,n-1),
                 preparation_49frame_facts=scope_facts(obs,*window) if window else None,
                 declared_native_window_source_facts=scope_facts(obs,*native_window) if native_window else None,
                 observation_bank_role='source_FRONT_in_declared_native_window_NOT_verified_G_training_camera_split' if native_window else 'full_raw_FRONT_annotation_proxy_NOT_current_G_support',
                 observation_bank_bins=bank_bins,
                 rotate_plus30deg_required_box_bearing_bins_proxy=required_rotate_bins,
                 rotate_plus30deg_missing_box_bearing_bins_proxy=unseen,
                 rotate_plus30deg_supported_unique_bin_fraction_proxy=(len(required_rotate_bins)-len(unseen))/len(required_rotate_bins) if required_rotate_bins else None,
                 proxy_view_scope='30degree quantized camera-to-box-center bearing only; no surface, color, occlusion or per-Gaussian support proof',
                 native_dynamic_asset_mapping_present=bool(native and native.get('status')=='asset_and_raw_track_bound' and native.get('dynamic_actor_index') is not None),
                 native_scene_checkpoint_inventory_present=bool(native and native.get('static_checkpoint')),
                 dynamic_actor_index_inventory=None if not native else native.get('dynamic_actor_index'),
                 inventory_mask_evidence_present=item.get('mask_evidence') is not None,
                 mask_inventory=item.get('mask_evidence'),positive_mask_frame_count=None,
                 f_geom=None,f_view=None,f_bg=None,f_motion=None,formal_P=None,evaluation_only_capability_bin=None,
                 unknown_reasons=['full_G_none_body_and_sparse_depth_support_not_measured','required_surface_visibility_not_measured','world_background_reveal_ray_support_not_measured','world_model_camera_native_time_transform_not_validated_for_this_actor'],
                 edit_GT_or_edited_result_read=False)
        rows.append(row)
    scene_stats.append(dict(context_id=context,tracks=len(byid),raw_frames=n))
assert {(x['context_id'],x['track_id']) for x in rows}==set(reg)
selected=[r for r in rows if r['in_preparation45']]


def stats(values):
    return dict(tracks=len(values),engineering_invalid=sum(bool(x['engineering_invalid_reasons']) for x in values),
                flag_counts=dict(Counter(f for r in values for f in r['source_diagnostic_flags'])),
                observed_FRONT_sector_count_histogram=dict(sorted(Counter(x['FRONT_source_distinct_sector_count'] for x in values).items())),
                rotate30_missing_bin_proxy_tracks=sum(bool(x['rotate_plus30deg_missing_box_bearing_bins_proxy']) for x in values),
                declared_native_G_window_inventory_registered=sum(x['declared_native_G_window_inventory'] is not None for x in values),
                native_dynamic_asset_mapping_inventory=sum(x['native_dynamic_asset_mapping_present'] for x in values),
                mask_evidence_inventory=sum(x['inventory_mask_evidence_present'] for x in values),
                mask_inventory_median_pixels_positive=sum(bool(x['mask_inventory'] and x['mask_inventory'].get('median_pixels',0)>0) for x in values),
                mask_inventory_median_pixels_zero=sum(bool(x['mask_inventory'] and x['mask_inventory'].get('median_pixels')==0) for x in values),
                preparation49_missing_fixed_anchor_tracks=sum(bool(x['preparation_49frame_facts'] and not all(x['preparation_49frame_facts']['fixed_first_middle_last_anchor_label_presence'].values())) for x in values),
                formal_qualification_known=0,formal_qualification_unknown=len(values),
                full_G_geometry_unknown=len(values),surface_view_unknown=len(values),background_support_unknown=len(values))
result=dict(schema='r4_source_only_actor_second_screen_facts_v1',scope='1228 registered annotated vehicles plus unchanged preparation45; legal raw annotation exports only',
            source_role='annotation_assisted_preedit_diagnostics_not_prediction_or_formal_qualification',
            actor_denominator=1228,preparation_queue_denominator=45,query_denominator_if_registered_3_operations=3684,
            all=stats(rows),preparation45=stats(selected),scenes=scene_stats,rows=rows,
            exclusions_from_current_execution={'engineering_invalid':sum(bool(x['engineering_invalid_reasons']) for x in rows)},
            formal_P=None,thresholds_calibrated=False,used_edit_reference=False,used_native_edit_result=False,
            notes=['Box bearing rotation proxy uses the previously registered positive 30degree world-Z yaw solely to expose measurement gaps; no P_view decision is made.',
                   'Sparse point count is annotation 3D box count, not measured body-mask depth coverage.',
                   'Actual unique registry denominator is1228; the unchanged preparation queue contains45.',
                   'Missing annotation is not physical absence; no-front is only lack of FRONT projected annotation evidence.',
                   'Kinematic finite differences are descriptive; annotation jitter and occlusion gaps prevent uncalibrated speed cutoffs from becoming engineering exclusions.'],
            source_files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in [registry_path,queue_path]]+source_receipts)
(DEST/'source_only_second_screen_facts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(DEST/'preparation45_source_facts.json').write_text(json.dumps(dict(schema=result['schema'],summary=result['preparation45'],rows=selected),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['actor_denominator','preparation_queue_denominator','all','preparation45']},ensure_ascii=True))
