import datetime, json, subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
p = root / 'task_manifest.json'
m = json.loads(p.read_text(encoding='utf-8-sig'))
m['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
m['current_experiment_status'] = 'v2_retry_actual_first_backward_passed_running'
m['current_experiment'] = 'opacity_outside30k_to30300'
m['known_findings']['30k_reset'] = 'Original controlled continuation completed: composite recovers by30300 but static branch remains poor on observed background.'
m['outside_retry'] = {
    'attempt1': 'Observer import failed before training; archived unchanged under continuation_diagnostic/attempts/opacity_outside_01_runtime_image_utils_import/',
    'v2_fix': 'Only3 observer import/binding changes; independent review PASS',
    'historical_launcher_pid':34845, 'historical_worker_pid':34848,
    'read_live_pid_before_any_operation':True,
    'first_actual_backward_original10_scalars_exact':True,
    'actual_first_step':30001,
    'actual_first_camera':'image_0/0017034',
    'original_loss':0.2980644106864929,
    'original_psnr':8.590572357177734,
    'additional_opacity_loss_unweighted':2.8153865337371826,
    'coefficient':0.05,
    'first5_GT_mask_domains_exact':True,
    'root_receipt_sha256':'5dddf4cddb06fd560c1128f9654828f28000dc0285c0e50c8733a7a69fe526f4',
    'harness_sha256':'3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154',
    'plan_sha256':'be9e385b37805457d2d9986985c9b7bd65af3fcde66445d3f15d156db5c90a93',
    'final_effect_pending':True,
}
p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
subprocess.run(['scp',str(p),'wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/task_manifest.json'],check=True)
print(json.dumps({'status':'recorded','first_actual_backward_passed':True},ensure_ascii=True))
