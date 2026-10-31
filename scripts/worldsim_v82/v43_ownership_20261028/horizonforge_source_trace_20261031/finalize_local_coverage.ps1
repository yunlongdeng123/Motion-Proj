$stageRoot = 'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\work\horizon_trace'
$outputRoot = 'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\outputs\HorizonForge_Waymo_Source_Trace_20261031'
$inventoryRoot = Join-Path $outputRoot 'horizon_waymo_inventory_20261031_r1'
$mappingPath = Join-Path $outputRoot 'visual_verification\final_all11\scene_context_mapping.json'
$mapping = Get-Content -Raw -LiteralPath $mappingPath | ConvertFrom-Json
$network = Get-Content -Raw -LiteralPath (Join-Path $stageRoot 'range_sorted_val\network_receipt.json') | ConvertFrom-Json
$raw = Get-Content -Raw -LiteralPath (Join-Path $inventoryRoot 'all_raw_inventory.json') | ConvertFrom-Json
$prior = Get-Content -Raw -LiteralPath (Join-Path $inventoryRoot 'sorted_val_source_coverage.json') | ConvertFrom-Json
$dense = Get-Content -Raw -LiteralPath (Join-Path $stageRoot 'sorted_val035_dense\manifest.json') | ConvertFrom-Json
$firstframes = Get-Content -Raw -LiteralPath (Join-Path $stageRoot 'existing_raw_firstframes_summary.json') | ConvertFrom-Json
if ($mapping.mappings.Count -ne 11 -or $network.candidates.Count -ne 10) { throw 'Final mapping/probe denominator mismatch' }
if ($network.network_payload_bytes_read -gt $network.payload_budget_bytes) { throw 'Network budget exceeded' }
$evidenceRoot = Join-Path $outputRoot 'source_probes'
New-Item -ItemType Directory -Force -Path $evidenceRoot,(Join-Path $outputRoot 'provenance_scripts') | Out-Null
Copy-Item -LiteralPath (Join-Path $stageRoot 'range_sorted_val') -Destination $evidenceRoot -Recurse -Force
Copy-Item -LiteralPath (Join-Path $stageRoot 'sorted_val035_dense\manifest.json') -Destination (Join-Path $evidenceRoot 'scene035_full_front_manifest.json')
Copy-Item -LiteralPath (Join-Path $stageRoot 'sorted_val035_dense\frame_timestamps.csv') -Destination (Join-Path $evidenceRoot 'scene035_frame_timestamps.csv')
Copy-Item -LiteralPath (Join-Path $stageRoot 'existing_raw_firstframes_summary.json') -Destination (Join-Path $evidenceRoot 'existing_raw50_firstframes_summary.json')
Get-ChildItem -LiteralPath $stageRoot -File -Filter '*.py' | Copy-Item -Destination (Join-Path $outputRoot 'provenance_scripts')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $outputRoot 'provenance_scripts\finalize_local_coverage.ps1')
$rows = @()
foreach ($match in $mapping.mappings) {
    $scene = $match.hf_scene
    $context = $match.context_id
    $existing = @($raw.files | Where-Object { $_.context_id -eq $context -and $_.suffix -in @('.tfrecord','.tfrecords') })
    $probe = @($network.candidates | Where-Object {$_.scene_number -eq $scene})
    $component = @($prior.rows | Where-Object {$_.scene_number -eq $scene})[0]
    $sensor = @($firstframes.frames | Where-Object {$_.context_id -eq $context})
    $relative = 'individual_files/validation/segment-' + $context + '_with_camera_labels.tfrecord'
    $uri = 'https://hf-mirror.com/datasets/AnnaZhang/waymo_open_dataset_v_1_4_3/resolve/8a6fccd2b022077308c167a4d3c6f646d6e5e65b/' + $relative
    $size = if ($existing.Count) {$existing[0].bytes} else {$probe[0].full_raw_expected_bytes}
    $rows += [pscustomobject][ordered]@{
        hf_scene=$scene;context_id=$context;correspondence='image_matched_first_frame';
        correspondence_evidence='visual_verification/final_all11/scene_context_mapping.json';
        match_ssim_gray=$match.ssim_gray;match_orb_ransac_inliers=$match.ransac_inliers;
        pinned_raw_object_uri=$uri;source_relative_path=$relative;full_raw_object_bytes=$size;
        full_raw_file_located=($existing.Count -gt 0);full_raw_local_paths=@($existing | ForEach-Object {$_.path});
        full_raw_current_sha256_recomputed=$false;
        source_probe_role=$(if ($scene -eq '035') {'existing_local_full_TFRecord'} else {'bounded_first_record_only'});
        source_probe_receipt=$(if ($scene -eq '035') {'source_probes/scene035_full_front_manifest.json'} else {'source_probes/range_sorted_val/sorted_val_' + $scene + '/source_receipt.json'});
        first_record_payload_saved=$(if ($scene -eq '035') {'available_inside_existing_full_raw'} else {'false_only_original_JPEG_and_receipt_kept'});
        first_record_crc_execution_record='header_and_payload_passed';
        independent_posthoc_payload_CRC_recomputed=$false;
        original_FRONT_raw0_JPEG_available=$true;
        additional_FRONT_RGB_analysis_frame_count=$(if ($scene -eq '035') {198} else {0});
        analysis_RGB_role='resized_original_observed_RGB_not_native_GS_or_generated_RGB';
        prepared_camera_files_located=$component.prepared_camera_file_count;
        prepared_pose_files_located=$component.prepared_pose_file_count;
        prepared_lidar_files_located=$component.prepared_lidar_file_count;
        prepared_annotation_files_located=$component.prepared_annotation_file_count;
        GS_checkpoints_located_in_scanned_context_paths=@($component.GS_checkpoint_files_located).Count;
        first_raw_frame_sensor_fields=$(if ($sensor.Count) {$sensor[0].raw_firstframe_sensor_fields} else {$null});
        calibrated_processed_dataset_located=$false;
        missing_for_native_editing=$(if ($scene -eq '035') {'calibrated native preprocessing, chosen target/masks/trajectory contract, frozen Gaussian reconstruction/checkpoint'} else {'complete TFRecord and later-frame RGB/sensors, calibrated preprocessing, target/masks/trajectory contract, frozen Gaussian reconstruction/checkpoint'});
        absence_scope='not_located_in_declared_scanned_roots_and_context_paths_not_global_nonexistence_proof'
    }
}
$shared = @('14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095')
$intersection = @($shared | Where-Object {$_ -in $rows.context_id})
$result = [ordered]@{
    schema='horizonforge_confirmed_context_source_inventory_v1';rows=$rows;
    exact_context_count=11;complete_raw_contexts_located=@($rows | Where-Object {$_.full_raw_file_located}).Count;
    complete_raw_scene_numbers_located=@($rows | Where-Object {$_.full_raw_file_located} | ForEach-Object {$_.hf_scene});
    new_bounded_first_record_contexts=10;new_network_payload_bytes=$network.network_payload_bytes_read;
    initial_network_budget_bytes=$network.initial_payload_budget_bytes;final_network_budget_bytes=$network.payload_budget_bytes;
    network_budget_adjustment=$network.budget_adjustment;monetary_network_cost='not_metered';
    total_full_raw_bytes_for_all11=($rows | Measure-Object -Property full_raw_object_bytes -Sum).Sum;
    shared_current_three_contexts=$shared;shared_three_intersection_with_confirmed_HF_contexts=$intersection;
    raw_inventory='horizon_waymo_inventory_20261031_r1/all_raw_inventory.json';
    all_existing_TFRecord_unique_contexts=50;local_raw_firstframe_CRC_probes=50;
    preview_cache_training_contexts=330;preview_cache_is_not_complete_raw_coverage=$true;
    no_new_training_or_reconstruction=$true;new_GPU_calls=0;new_training_updates=0;
    helpers_created='standalone foreground Python inventory/RGB/range scripts under codex_tmp; no daemon/service/proxy';
    controllers_started=0;new_proxy_or_SSH_tunnel_started=$false;shutdown_performed=$false;
    instance_disposition='left_running_in_user_CPU_mode';
    limitations=@('Mapping verified from original FRONT pixel agreement; ten range payloads discarded after CRC/protobuf validation, so independent reviewer did not recompute their payload CRC.',
      'No aligned native GS checkpoint or complete calibrated processed sensor package was located for these11 in searched roots; numeric aliases without identity metadata remain unassigned.',
      'Complete raw source035 embeds all sensor fields; only FRONT RGB/timestamps were decoded in this task.');
    recommended_minimal_CPU_preprocess=[ordered]@{
      source_scene='035';source_context='13299463771883949918_4240_000_4260_000';
      raw_local_path=$rows[3].full_raw_local_paths;
      primitive_module='/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py';
      reusable_functions=@('protobuf','records','front_rgb','calibration_text','lidar');
      output_format_reference='/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/persistent_world_20261027/prepare_motion_sensor_inputs.py:worker';
      plan='First register035 and a fixed frame window; run one CPU worker with CUDA hidden into a new output directory to write original RGB, calibrated five-LiDAR/two-return points, camera calibration, ego poses, timestamps and original annotations. Keep target masks/track association pending until explicitly selected.';
      existing_driver_caveat='prepare_motion_sensor_inputs.py main is hard-coded to current three contexts and3 workers; it is a format/helper reference, not a ready035 CLI. prepare_eval_set main also uses4 workers and train-directory discovery; do not execute these mains unchanged.';
      executed_now=$false
    }
}
$result | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $outputRoot 'confirmed_source_inventory.json')
$rows | ForEach-Object {
    [pscustomobject]@{hf_scene=$_.hf_scene;context_id=$_.context_id;full_raw_object_bytes=$_.full_raw_object_bytes;
        full_raw_located=$_.full_raw_file_located;source_probe_role=$_.source_probe_role;
        original_FRONT_frame0=$_.original_FRONT_raw0_JPEG_available;
        FRONT_analysis_frames=$_.additional_FRONT_RGB_analysis_frame_count;
        prepared_camera_files=$_.prepared_camera_files_located;prepared_pose_files=$_.prepared_pose_files_located;
        prepared_lidar_files=$_.prepared_lidar_files_located;prepared_annotation_files=$_.prepared_annotation_files_located;
        GS_checkpoints_located=$_.GS_checkpoints_located_in_scanned_context_paths;
        pinned_raw_object_uri=$_.pinned_raw_object_uri;missing_for_native_editing=$_.missing_for_native_editing;absence_scope=$_.absence_scope}
} | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $outputRoot 'confirmed_source_inventory.csv')
Write-Output "Confirmed contexts: $($rows.Count); full raw found: $($result.complete_raw_contexts_located); shared3 intersection: $($intersection.Count)"
Write-Output "New payload bytes: $($network.network_payload_bytes_read); full11 total bytes: $($result.total_full_raw_bytes_for_all11)"
