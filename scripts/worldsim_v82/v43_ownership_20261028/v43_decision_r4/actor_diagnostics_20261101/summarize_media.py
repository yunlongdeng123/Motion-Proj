"""全15场就绪后生成机器收口，不覆盖root报告。"""
import json,pathlib
BASE=pathlib.Path(__file__).resolve().parent
def main():
    cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'));rows=[];actors=[];views=[]
    for scene in cfg['scenes']:
        root=BASE/'generated/scenes'/scene['context_id']
        if not (root/'frame_point_views.json').exists():raise RuntimeError('not ready '+scene['scene_key'])
        row=json.loads((root/'manifest.json').read_text(encoding='utf8'));media=json.loads((root/'media.json').read_text(encoding='utf8'))
        rows.append(row);views.append(json.loads((root/'frame_point_views.json').read_text()))
        actors.extend(dict(context_id=scene['context_id'],track=track,**a['lidar']) for track,a in media['actors'].items())
    qa=json.loads((BASE/'video_inventory_audit.json').read_text());assert qa['completed_scenes']==15
    stats=json.loads((BASE/'spatial/actual_spatial_stats.json').read_text(encoding='utf8'));assert stats['complete_scenes']==15
    files=[p for p in (BASE/'generated').rglob('*') if p.is_file() and 'projection_diagnostics_r1' not in p.parts and 'color_diagnostics_r1' not in p.parts]
    q=[a for a in actors if a['selected_preparation_queue']]
    report=dict(schema='actor_media_closeout_v1',scope='raw-media and measured-LiDAR worker; excludes separate root-owned GS checkpoint comparisons',scenes=len(rows),scene_denominator=len(cfg['scenes']),full_RGB_sequences=len(rows),
        full_3D_box_overlay_sequences=len(rows),actual_H264_videos=len(rows)*2,
        original_raw_frames=sum(r['raw_frame_count'] for r in rows),original_raw_frame_denominator=sum(s['expected_frames'] for s in cfg['scenes']),
        vehicle_tracks=len(actors),vehicle_track_denominator=sum(len(s['actors']) for s in cfg['scenes']),prepared_queue_targets=len(q),
        queue_targets_with_actual_points=sum(a['measured_points_before_voxel']>0 for a in q),
        tracks_with_actual_points=sum(a['measured_points_before_voxel']>0 for a in actors),tracks_without_actual_points=sum(a['measured_points_before_voxel']==0 for a in actors),
        measured_sensor_points=sum(r['measured_sensor_points'] for r in rows),measured_OBB_rows=sum(r['measured_box_rows'] for r in rows),
        measured_OBB_row_note='Actor-box membership rows; the same source point may appear in overlapping boxes.',
        aggregated_display_points=sum(a['display_points'] for a in actors),independent_frame_display_points=sum(v['display_points'] for v in views),
        labelled_actor_frame_observations=sum(v['label_observations'] for v in views),output_files=len(files),output_bytes=sum(p.stat().st_size for p in files),
        output_excludes='preserved invalid projection/color diagnostic revisions',render_resolution=[960,640],video_fps=10,
        source_world_to_actor='per-frame original 3D box center/heading and frame vehicle/world pose; x forward y left z up',
        geometry_source='raw 3D annotations; FRONT visibility whitelist; calibrated provided-image-pose approximation',
        LiDAR_source='actual five-LiDAR double-return range images; TOP pixel ego compensation; NLZ filtering',
        query_or_training_GPU_calls=0,training_updates=0,model_checkpoint_loads=0,shutdown_performed=False,remote_CPU_quota='50000 100000',
        remote_controllers_started=['selector','single-thread actor point crop','stop-after-020 guard'],remote_controllers_state='all stopped; no pending restart',
        wall_time_note='Per-scene wall times overlap; do not sum them as elapsed wall clock.',
        per_scene_build_wall_seconds={r['scene_key']:r['wall_seconds'] for r in rows},remote_212_actual_decode_wall_seconds=344.7115869577974,
        extra_initial_035_5frame_smoke=dict(wall_seconds=3.752,measured_sensor_points=883935,box_rows=94626),
        partial_remote_537_probe=dict(completed_frames_exact=None,recorded_completed_frame_lower_bound=1,conservative_completed_frame_upper_bound=41,
            last_logged_raw=0,next_progress_raw=40,reason='terminated after switching to local full-raw path; only raw0 progress emitted; no raw40 progress; partial original record retained'),
        safeguards=['Source datasets/models not modified.','Existing scene_key and aN aliases preserved.',
                    'Per-frame independent voxel data; aggregate subset never reused as frame data.'],
        limitations=['OBB points may contain road/occluders; not instance segmentation or a completed vehicle.',
            'Provided image pose approximates rolling shutter and object exposure timing.','TOP pixel ego compensation excludes within-scan actor motion.',
            'RGB has finite-FOV guard but no visibility zbuffer.',
            '212 strict cross-runtime 20um parity fails at sub-1mm sampled maximum; source identity/counts/OBB checks pass; see independent numeric audit.'],
        complete_body=None,closed_surface=None,formal_P=None,edit_success=None,human_verdict=None,scenes_summary=rows)
    (BASE/'media_final_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k in ('scenes','original_raw_frames','vehicle_tracks','queue_targets_with_actual_points','measured_sensor_points','measured_OBB_rows','output_bytes')}))
if __name__=='__main__':main()
