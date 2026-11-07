"""Read-only CPU metadata audit of four explicitly known native Full5 runs.

Run on the AutoDL host with its existing drivingeditor Python. No engine import,
CUDA initialization, reconstruction, model edit, or whole-weight hash scan.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import ast
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

CAMP = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ENGINE = CAMP / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
RUNS = [CAMP / 'r15_gn11_stage1_v41_20261023/arms/B/run30k',
        CAMP / 'r15_gn11_stage1_v41_20261023/arms/R/run30k',
        CAMP / 'r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11',
        CAMP / 'r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k']
TOKENS = re.compile(r'owner|lineage|parent|uid|source.*ind|initial.*ind|clone.*ind|split.*ind|prune.*ind', re.I)


def stat(path):
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def value_schema(v):
    if torch.is_tensor(v):
        return dict(type='tensor', shape=list(v.shape), dtype=str(v.dtype), device=str(v.device))
    if isinstance(v, dict):
        return dict(type='dict', keys=[str(key) for key in v])
    if isinstance(v, (list, tuple)):
        return dict(type=type(v).__name__, length=len(v))
    return dict(type=type(v).__name__, value=v) if isinstance(v, (str, int, float, bool, type(None))) else dict(type=type(v).__name__)


def provenance_keys(value, path=''):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = path+'/'+str(key)
            if TOKENS.search(str(key)):
                result.append(dict(path=current, value_schema=value_schema(child)))
            result += provenance_keys(child, current)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            result += provenance_keys(child, path+'/'+str(index))
    return result


def checkpoint(path):
    before = stat(path)
    value = torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
    capture, iteration = value
    optimizer = capture[14] if isinstance(capture, (tuple, list)) and len(capture)>14 else None
    with zipfile.ZipFile(str(path)) as archive:
        name = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        metadata = archive.read(name)
    result = dict(file=before, top_type=type(value).__name__, top_length=len(value), iteration=iteration,
                  capture_length=len(capture), capture_fields=[value_schema(v) for v in capture],
                  optimizer_top_keys=list(optimizer) if isinstance(optimizer, dict) else None,
                  optimizer_group_names=[row.get('name') for row in optimizer.get('param_groups', [])] if isinstance(optimizer, dict) else [],
                  optimizer_state_fields={str(key):list(v) for key,v in optimizer.get('state', {}).items()} if isinstance(optimizer, dict) else {},
                  actual_provenance_key_hits=provenance_keys(capture),
                  archive_pickle_metadata_bytes=len(metadata), archive_pickle_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                  full_weight_sha256_recomputed=False, weights_mapped_to_CPU_not_materialized_for_geometry=True)
    result['file_stat_unchanged_after_metadata_read'] = before == stat(path)
    del capture, optimizer, value
    return result


def run_inventory(run):
    result = dict(path=str(run), exists=run.is_dir())
    if not run.is_dir():
        return result
    result['direct_weight_filenames'] = sorted(p.name for p in run.glob('*.pth'))
    nodes = sorted({int(match.group(1)) for p in run.glob('*.pth')
                    for match in [re.search(r'(\d+)\.pth$', p.name)] if match})
    result['component_availability'] = []
    for node in nodes:
        names = [prefix+str(node)+'.pth' for prefix in ('chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state')]
        present = {name:(run/name).is_file() for name in names}
        result['component_availability'].append(dict(iteration=node, files=present,
                                                    render_components_complete=all(present[name] for name in names[:3]),
                                                    full_restore_components_complete=all(present.values())))
    dynamic_paths = sorted(run.glob('dynamic_chkpnt*.pth'))
    result['dynamic_capture_audits'] = [checkpoint(path) for path in dynamic_paths]
    result['training_state_audits'] = []
    for path in sorted(run.glob('training_state*.pth')):
        before = stat(path)
        state = torch.load(str(path), map_location='cpu', weights_only=False)
        selected = {key:value for key,value in state.items() if key not in ('config','python_rng','numpy_rng','torch_rng','cuda_rng')}
        result['training_state_audits'].append(dict(file=before, top_keys=list(state),
            model_runtime=state.get('model_runtime'), actual_runtime_provenance_key_hits=provenance_keys(selected),
            sgcm_top_keys=list(state.get('sgcm', {})),
            SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), file_stat_unchanged=before==stat(path)))
    # Names only, within this exact known run; no campaign-wide discovery.
    result['ownership_sidecar_name_hits'] = sorted(str(p.relative_to(run)) for p in run.rglob('*')
        if p.is_file() and any(token in p.name.lower() for token in ('owner','lineage','parent','source_index','provenance')))
    result['jsonl_examples'] = []
    for name in ('gradient_diagnostics.jsonl','paired_steps.jsonl'):
        path = run/name
        if path.is_file():
            with path.open() as handle:
                first = json.loads(next(handle))
            result['jsonl_examples'].append(dict(path=str(path), bytes=path.stat().st_size, first_record_keys=list(first),
                                                first_record_provenance_key_hits=provenance_keys(first),
                                                complete_history_content_audited=False))
    return result


def main():
    torch.set_num_threads(1)
    source = ENGINE/'scene/gaussian_model.py'
    text = source.read_text()
    methods = []
    for cls in ast.parse(text).body:
        if isinstance(cls, ast.ClassDef) and cls.name in ('GaussianModel','Dynamic_GaussianModel'):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ('capture','densify_and_clone','densify_and_split','prune_points','densification_postfix'):
                    methods.append(dict(class_name=cls.name, method=fn.name, line=fn.lineno, end_line=fn.end_lineno,
                                        calls_capture_sgcm_optimizer=any(isinstance(n,ast.Name) and n.id=='capture_sgcm_optimizer' for n in ast.walk(fn))))
    full = CAMP/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
    preparation = json.loads((full/'dynamic_multi_actor_preparation.json').read_text())
    runs = [run_inventory(path) for path in RUNS]
    result = dict(status='PASS_READ_ONLY_BOUNDED_NATIVE_FULL5_OWNER_AUDIT', recorded_utc=datetime.now(timezone.utc).isoformat(),
                  GPU_initialized=torch.cuda.is_initialized(), optimizer_updates=0, old_model_or_source_modified=False,
                  search_scope=[str(path) for path in RUNS], additional_arm_RP_empty=not any((CAMP/'r15_gn11_stage1_v41_20261023/arms/RP').iterdir()),
                  initial_concat_metadata=dict(path=str(full/'dynamic_multi_actor_preparation.json'),
                    actors=[dict(actor_index=row['actor_index'],raw_track_id=row['raw_track_id'],real_points=row['total_real_points']) for row in preparation['actors']],
                    limitation='This establishes initial PLY ownership only, not the surviving/densified final rows.'),
                  capture_writer_source=dict(path=str(source),SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),methods=methods),
                  runs=runs,
                  strict_B_R_reuse_for_actor_queries=False,
                  conclusion='Native B/R captures contain no actor owner or complete dynamic clone/split/prune lineage. Initial concatenation ranges cannot be propagated to final rows without the missing index history. Do not infer owners from bbox, nearest points, or learned appearance.',
                  absence_limit='Only the four explicit native run paths and named source writer were audited. JSONL first records are schema examples, not an exhaustive scan of every log value.',
                  no_GB_weight_download=True,no_full_weight_hash_rescan=True,no_renderer_or_CUDA_import=True)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
