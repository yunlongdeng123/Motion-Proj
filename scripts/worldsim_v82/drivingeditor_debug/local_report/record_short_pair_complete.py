import datetime, hashlib, json, subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
summary=json.loads((root/'continuation_summary.json').read_text(encoding='utf8'))
receipt_path=root/'continuation_diagnostic/opacity_outside30k_to30300/receipts/full49_summary.json'
receipt=json.loads(receipt_path.read_text())
assert receipt['harness_exit_code']==receipt['wrapper_exit_code']==0
for key in ['all300_camera_time_shift_LR_exactly_paired','all6_node_eval_RNG_preserved','all_protected_SHA_unchanged']:
    assert receipt[key] is True
metrics={c['case']:next(n['means'] for n in c['nodes'] if n['iteration']==30300 and n['frame_count']==49) for c in summary['cases']}
p=root/'task_manifest.json';m=json.loads(p.read_text(encoding='utf-8-sig'))
m.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_experiment_status='short_pair_complete_long_three_arm_CPU_preparation',gpu_owner_scope='No new GPU launch until root-approved actual long B1 plan',next_experiment='original30k_to32000 B1; original repeat B2; same opacity outside hypothesis V, all original30k state. CPU materialization approved; GPU not yet approved.')
m['outside_retry']['final_effect_pending']=False
m['controlled_short_pair']={'status':'both_complete_exit0','full49_mean_per_frame_metrics':metrics,'variant_receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'all300_inputs_paired':True,'all6_eval_rng_preserved':True,'original_files_unchanged':True,'interpretation':'Static observed region improves markedly with a small composite cost; dynamic background leakage persists. Not repaired or full paper reproduction.','visual_review':'Actual raw10 composite/static/dynamic reviewed. Road restored in static but still blurred/ghosted; dynamic retains background highlights/textures.'}
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
subprocess.run(['scp',str(p),'wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/task_manifest.json'],check=True)
print(json.dumps({'status':'recorded_short_pair_complete','paper_reproduction_complete':False}))
