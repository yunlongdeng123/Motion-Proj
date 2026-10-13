"""Make narrowly scene-aware copies of the existing terminal audit tools."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'heartbeat_20261013_0847'


def put(name, body):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)


audit = (OLD / 'scene8_terminal/audit_scene8_terminal_readonly.py').read_text(encoding='utf-8')
audit = audit.replace('scene0008_023_071_hybrid_30000_seed0_GN11', 'scene0029_069_117_hybrid_30000_seed0_GN11')
audit = audit.replace("-.28, abs_tol=1e-12", "spec['anchor_timestamp'], abs_tol=1e-12")
audit = audit.replace("'raw_source_frame_range': [23, 71]", "'raw_source_frame_range': spec['raw_frame_range']")
audit = audit.replace('pid = 96504', 'pid = 124555')
audit = audit.replace("any('/campaign/followup_scene_prep_20261013/' in p for p in parts)", "any('scene0029_spec.json' in p for p in parts)")
audit = audit.replace("'scheduled_check_label': '2026-10-13 08:47 SGT'", "'scheduled_check_label': '2026-10-13 final red terminal after shared lock'")
audit = audit.replace("check(summary.get('iteration') == step,", "check(summary.get('anchor_pose_frame') == spec['anchor_frame'], 'static_time_anchor_frame', summary.get('anchor_pose_frame'))\n    check(summary.get('iteration') == step,")
put('audit_red_terminal_readonly.py', audit)

extract = (OLD / 'scene8_terminal/extract_scene8_training_rows_readonly.py').read_text(encoding='utf-8')
extract = extract.replace('scene0008_023_071_hybrid_30000_seed0_GN11', 'scene0029_069_117_hybrid_30000_seed0_GN11')
extract = extract.replace('scene0008_GN11_training.log', 'scene0029_GN11_training.log')
extract = extract.replace("'scene_id': 'scene0008_023_071'", "'scene_id': 'scene0029_069_117'")
extract = extract.replace('nps_loss_failures = []', 'nps_loss_failures = []\nall_loss_finite_failures = []\nloss_numeric_fields_seen = set()')
extract = extract.replace("losses = row['losses']", "losses = row['losses']\n        for field, value in losses.items():\n            if type(value) in (int, float):\n                loss_numeric_fields_seen.add(field)\n                if not math.isfinite(value):\n                    all_loss_finite_failures.append({'iteration': iteration, 'field': field, 'value': str(value)})")
extract = extract.replace("'sequence_failures': sequence_failures", "'all_30000_logged_numeric_losses_finite': row_count == 30000 and not all_loss_finite_failures,\n    'all_loss_finite_failures': all_loss_finite_failures,\n    'logged_numeric_loss_fields': sorted(loss_numeric_fields_seen),\n    'sequence_failures': sequence_failures")
put('extract_red_training_rows_readonly.py', extract)

summary = (OLD / 'scene8_terminal/summarize_scene8_terminal_cpu.py').read_text(encoding='utf-8')
summary = summary.replace('import statistics', 'import statistics\nimport sys')
summary = summary.replace('STAGE = Path(__file__).resolve().parent', 'STAGE = Path(sys.argv[1]).resolve()')
summary = summary.replace('scene0008_', 'scene0029_').replace('SCENE0008_', 'SCENE0029_')
summary = summary.replace("summary['anchor_pose_frame'] == '0008033'", "summary['anchor_pose_frame'] == spec['anchor_frame']")
summary = summary.replace("rows[10][key]", "rows[spec['anchor_index']][key]")
summary = summary.replace("'fixed_pose': '0008033'", "'fixed_pose': spec['anchor_frame']")
summary = summary.replace("assert len(samples) == memory['sampling']['samples'] == 5068", "assert len(samples) == memory['sampling']['samples'] and len(samples) > 0")
summary = summary.replace("'scheduled_check': '2026-10-13 08:47 SGT'", "'scheduled_check': '2026-10-13 final red terminal after shared lock'")
summary = summary.replace("'remote_or_repository_modified': False", "'remote_training_assets_or_repository_modified': False, 'new_report_metadata_written': True")
summary = summary.replace("(STAGE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\\n', encoding='utf-8')", "with (STAGE / name).open('x', encoding='utf-8') as stream:\n        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\\n')")
summary = summary.replace("assert memory['error'] is None and memory['error_count'] == 0", "assert memory['error'] is None and memory['error_count'] == 0\nassert training['all_30000_logged_numeric_losses_finite']\nassert training['sampler_contiguous_1_through_30000']\nassert training['all_30000_NPS_loss_finite_enabled_weight_0p01_scaling_pass']\nassert not training['log_error_signature_hits']\nassert set(training['sampler_unique_frame_ids']) == set(spec['frame_ids_with_camera'])")
summary = summary.replace("'all30000_NPS_losses_finite_enabled_scaled_0p01': training", "'all30000_logged_numeric_losses_finite': training['all_30000_logged_numeric_losses_finite'],\n        'logged_numeric_loss_fields': training['logged_numeric_loss_fields'],\n        'all30000_NPS_losses_finite_enabled_scaled_0p01': training")
summary = summary.replace("'same_step_duplicate_counter_missing_calls':", "'same_step_duplicate_counter_missing_calls':")
put('summarize_red_terminal_cpu.py', summary)

svg = (OLD / 'gmc_html/architecture-components.svg').read_text(encoding='utf-8')
put('architecture-components.svg', svg.replace('GMC', '红车'))

for path in HERE.glob('*.py'):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
print('Scene-aware audit, one-pass extraction, saved-spec summarizer and red architecture prepared; syntax PASS.')
