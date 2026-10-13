"""Read one terminal log and one sampler JSONL pass; retain 145 SGCM + 6 NPS rows."""
import datetime
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
ROOT = BASE / 'runs/scene0029_069_117_hybrid_30000_seed0_GN11'
NODES = (1, 1000, 10000, 15000, 20000, 30000)
decoder = json.JSONDecoder()
log_path = BASE / 'scene0029_GN11_training.log'
sgcm_init = []
sgcm_stats = []
log_error_hits = []
with log_path.open(encoding='utf-8', errors='replace') as stream:
    for line in stream:
        for marker, target in (('SGCM_INIT ', sgcm_init), ('SGCM_STATS ', sgcm_stats)):
            start = line.find(marker)
            if start >= 0:
                row, end = decoder.raw_decode(line[start + len(marker):].lstrip())
                target.append(row)
        if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:', line):
            log_error_hits.append(line.strip()[-2000:])

sampler_path = ROOT / 'paired_steps.jsonl'
nps_nodes = []
row_count = 0
sampled_frames = set()
sequence_failures = []
nps_loss_failures = []
all_loss_finite_failures = []
loss_numeric_fields_seen = set()
with sampler_path.open(encoding='utf-8') as stream:
    for index, line in enumerate(stream, 1):
        row = json.loads(line)
        row_count += 1
        iteration = row['iteration']
        if iteration != index:
            sequence_failures.append({'line': index, 'iteration': iteration})
        sampled_frames.add(row['frame_id'])
        losses = row['losses']
        for field, value in losses.items():
            if type(value) in (int, float):
                loss_numeric_fields_seen.add(field)
                if not math.isfinite(value):
                    all_loss_finite_failures.append({'iteration': iteration, 'field': field, 'value': str(value)})
        raw = losses['loss_hybrid_nps_raw']
        weighted = losses['loss_hybrid_nps_weighted']
        weight = losses['loss_hybrid_nps_weight']
        if not all(math.isfinite(v) for v in (raw, weighted, weight)) or weight != .01 or not losses['hybrid_nps_enabled'] or not math.isclose(weighted, .01 * raw, rel_tol=1e-5, abs_tol=1e-6):
            nps_loss_failures.append({'iteration': iteration, 'raw': raw, 'weighted': weighted, 'weight': weight})
        if iteration in NODES:
            nps_nodes.append({'iteration': iteration, 'frame_id': row['frame_id'],
                              'time_shift': row['time_shift'],
                              'loss_hybrid_nps_raw': raw,
                              'loss_hybrid_nps_weighted': weighted,
                              'loss_hybrid_nps_weight': weight,
                              'loss_hybrid_total': losses['loss_hybrid_total'],
                              'loss_hybrid_static_rgb_weighted': losses['loss_hybrid_static_rgb_weighted'],
                              'learning_rates': row['learning_rates']})
assert 'torch' not in sys.modules
receipt = {
    'schema_version': 1, 'scene_id': 'scene0029_069_117', 'arm': 'GN11',
    'observed_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'ssh_alias': 'wm-vgpu-1011', 'python_executable': sys.executable,
    'sources': {'training_log': str(log_path), 'paired_sampler_rows': str(sampler_path)},
    'source_read_passes': {'training_log': 1, 'paired_sampler_rows': 1},
    'SGCM_INIT': sgcm_init, 'SGCM_STATS': sgcm_stats,
    'SGCM_INIT_count': len(sgcm_init), 'SGCM_STATS_count': len(sgcm_stats),
    'NPS_loss_nodes': nps_nodes, 'NPS_loss_node_count': len(nps_nodes),
    'sampler_rows': row_count, 'sampler_unique_frame_ids': sorted(sampled_frames),
    'sampler_contiguous_1_through_30000': row_count == 30000 and not sequence_failures,
    'all_30000_NPS_loss_finite_enabled_weight_0p01_scaling_pass': row_count == 30000 and not nps_loss_failures,
    'all_30000_logged_numeric_losses_finite': row_count == 30000 and not all_loss_finite_failures,
    'all_loss_finite_failures': all_loss_finite_failures,
    'logged_numeric_loss_fields': sorted(loss_numeric_fields_seen),
    'sequence_failures': sequence_failures, 'NPS_loss_failures': nps_loss_failures,
    'log_error_signature_hits': log_error_hits,
    'read_only': True, 'GPU_used': False, 'Torch_imported': False,
    'CUDA_initialized': False, 'remote_files_written': False,
    'checkpoint_loaded': False, 'new_hash_or_fingerprint_computed': False,
    'scope': 'Single scene sampler and logged SGCM accounting; no between-arm sampler equality or unique-lineage historical proof',
    'human_verdict': None,
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
