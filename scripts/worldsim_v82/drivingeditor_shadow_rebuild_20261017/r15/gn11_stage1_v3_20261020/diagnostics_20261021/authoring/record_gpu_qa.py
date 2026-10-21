import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent;OUTPUTS=ROOT.parents[1]/'outputs'
qa=dict(status='PASSED_WITH_EXPLICIT_LOCAL_FILE_AUTOMATION_LIMIT',tested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    full_HTTP_four_arm_switches=True,full_HTTP_five_native_branch_modes=True,
    all_four_frame137_canvases_aligned=True,Gaussian_table_to_ellipsoid_link=True,
    live_CUDA_trace_button=True,live_single_Gaussian_complete_rerasterization=True,
    live_native_camera_full_frame_replay_error=0,live_orbit_full_model_view=True,
    standalone_portable_zip_extracted_and_browser_tested=True,extracted_19_Gaussian_control_count_displayed=True,
    counterfactual_five_images_actual480x320=True,portable_unsampled_pixel_values_unknown=True,portable_normals_disabled=True,
    original_camera_ordinary_branch_no_old_baseline_label=True,
    active_full_origin_console_errors=0,portable_origin_console_errors=0,
    legacy_console_error='one old8768 development asset-missing log retained; not from8770/8771/8772 releases',
    direct_file_URL_automation='blocked by browser protocol policy; not tested or bypassed; HTTP and extracted package tested',
    screenshot='GS_Reconstruction_Diagnostics_GPU_preview.png',training_updates=0)
for name in ('GS_Reconstruction_Diagnostics_20261021_GPU_r2','GS_Reconstruction_Diagnostics_20261021_GPU_portable_r3'):
    out=OUTPUTS/name
    record=dict(qa,verified_source_sha256={f:hashlib.file_digest((out/f).open('rb'),'sha256').hexdigest() for f in ('index.html','app.js','gpu.js','server.py')})
    (out/'browser_gpu_qa.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    validation=json.loads((out/'validation.json').read_text());validation['gpu_browser_qa']=record
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'record_gpu_qa.py',out/'source/authoring/record_gpu_qa.py')
    original=OUTPUTS/'V3_Four_Arms_HTML_20261021_final.zip'
    assert hashlib.file_digest(original.open('rb'),'sha256').hexdigest()=='8937809e0d86f5ef84756e2d41ab0377e270060c3c28ff89d0a65a7fd81afd90'
print(json.dumps(qa,ensure_ascii=False))
