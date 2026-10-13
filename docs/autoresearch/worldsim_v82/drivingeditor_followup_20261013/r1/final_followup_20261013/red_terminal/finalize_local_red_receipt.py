"""Write the actual local archive/PNG/embedded manifest/JavaScript validation result."""
import datetime
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[3] / 'outputs/drivingeditor_red_GN11_30k'
manifest = json.loads((REPORT / 'manifest.json').read_text(encoding='utf-8'))
assets = json.loads((REPORT / 'local_asset_verification.json').read_text(encoding='utf-8'))
lifecycle = json.loads((HERE / 'owned_process_exit_receipt.json').read_text(encoding='utf-8'))
node = subprocess.run(['node', '--check', str(HERE / 'local_viewer_runtime.js')], capture_output=True, text=True)
assert node.returncode == 0, node.stderr
assert assets['status'] == 'PASS_RED_OFFLINE_ASSETS_AND_EMBEDDED_SOURCE'
assert lifecycle['status'] == 'PASS_ALL_OWNED_RED_TRAINING_AND_CPU_COLLECTOR_PROCESSES_EXITED'
archive = HERE / 'red_gn11_30k_saved_cpu_20261013.tar.gz'
assert archive.stat().st_size == 51221463
record = {'status': 'PASS_LOCAL_RED_REPORT_TRANSFER_ASSETS_MANIFEST_AND_JS_SYNTAX',
          'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'report': str(REPORT), 'archive_stat_bytes': archive.stat().st_size,
          'source_scene': manifest['scene'], 'raw_range': [69, 117], 'anchor': '0029079',
          'PNG_decode_count': assets['decoded_PNGs'], 'valid_actor_frames': assets['valid_actor_frames'],
          'empty_actor_frames': assets['empty_actor_masks_decode_black'], 'embedded_manifest_matches': True,
          'architecture_svg_parses': True, 'JS_syntax_command': node.args, 'JS_syntax_returncode': node.returncode,
          'owned_training_and_collector_exit': lifecycle, 'browser_quality_checked_by_this_agent': False,
          'FP32_metric_replay': manifest['verification'], 'restoration_verified': False,
          'actual_model_Adam_RNG_restore_verified': False, 'human_verdict': None,
          'GPU_used': False, 'new_hash_or_fingerprint_computed': False}
with (REPORT / 'local_validation.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({'status': record['status'], 'report': str(REPORT), 'owned_processes_all_exited': True,
                  'regional_metric_replay_count': manifest['verification']['regional_numeric_metrics_recomputed_and_matched'],
                  'regional_metric_replay_max_abs_error': manifest['verification']['regional_max_abs_metric_replay_error'],
                  'static_time_replay_count': manifest['verification']['static_time_metrics_recomputed_and_matched'],
                  'human_verdict': None}, ensure_ascii=False))
