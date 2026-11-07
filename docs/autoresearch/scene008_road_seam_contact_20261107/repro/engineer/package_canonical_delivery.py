"""CPU-only archive and hash inventory for the final canonical replay evidence."""
from pathlib import Path
import argparse
import json
import tarfile
from world_surface_sampling import sha


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--engineer',type=Path,required=True);args=p.parse_args();base=args.engineer.resolve()
    target=base/'canonical_n2_mip_candidate_v2';capture=base/'canonical_capture_r1'
    final=json.loads((target/'scene_manifest.json').read_text())
    replay=json.loads((target/'parity/receipt.json').read_text());cap=json.loads((capture/'render_tuple.receipt.json').read_text())
    assert replay['forwards']==10 and cap['forwards']==0
    assert replay['all_float_fields_exact'] and replay['all_PNG_exact']
    assert not final['adoption_claim'] and final['status']=='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE'
    assert sha(target/final['new_road']['file'])==final['new_road']['sha256']
    ledger=dict(status=final['status'],adoption_claim=False,asset_sha256=final['new_road']['sha256'],
        scene_manifest_sha256=sha(target/'scene_manifest.json'),
        exact_capture=dict(receipt='canonical_capture_r1/render_tuple.receipt.json',forwards=0,
            gpu_enabled_runtime_window_seconds=cap['seconds'],peak_allocated_bytes=cap['GPU_peak_allocated_bytes'],
            peak_reserved_bytes=cap['GPU_peak_reserved_bytes'],concurrency=cap['preflight']),
        independent_replay=dict(receipt='canonical_n2_mip_candidate_v2/parity/receipt.json',forwards=10,
            gpu_enabled_runtime_window_seconds=replay['seconds'],all_float_fields_exact=True,all_PNG_byte_exact=True,
            peak_allocated_bytes=replay['peak_allocated_bytes'],peak_reserved_bytes=replay['peak_reserved_bytes'],
            concurrency=replay['preflight'],raws=[r['raw'] for r in replay['records'][::2]]),
        additional_engineering_forwards=10,total_additional_gpu_enabled_runtime_window_seconds=cap['seconds']+replay['seconds'],
        original_model_parameter_updates=0,newroad_optimizer_updates=0,
        material='unchanged parent cellfit1000; blue boundary control not selected',
        CPU_formula_checks=dict(attempts=2,GPU_forwards=0,initial_test_error='float32 identity in a float64 analytic test; fixed dtype without tolerance relaxation'),
        budget_scope='additional canonical export/replay engineering only; root-owned n1/n2/AA pilot/full49 budgets are separate',
        numerical_semantics='bitwise float/PNG comparison; no relaxed equivalent label',
        unresolved=final['known_unresolved'],AutoDL_shutdown=False)
    write(base/'canonical_delivery_ledger.json',ledger)
    files=[path for directory in (target,capture) for path in directory.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    names=('world_surface_sampling.py','refined_road_runtime.py','refined_road_runtime.pre_relativeMip.py',
        'relative_mip.py','relative_mip_cpu_check.py','relative_mip_cpu_check.initial_eye_fp32.py','relative_mip_cpu_check.json',
        'prepare_engineering_base.py','cpu_base_receipt.json','canonical_road_runtime.py','capture_refined_tuple.py',
        'export_refined_canonical.py','verify_canonical_replay.py','RELATIVE_MIP_CONTROL.md','CANONICAL_REPLAY.md',
        'package_canonical_delivery.py','canonical_delivery_ledger.json')
    files.extend(base/name for name in names if (base/name).is_file())
    inventory={str(path.relative_to(base)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(set(files))}
    index=base/'canonical_artifact_manifest.json';write(index,dict(files=inventory,scene_status=final['status'],adoption_claim=False))
    archive=base/'canonical_delivery_v2.tar.gz'
    if archive.exists():raise FileExistsError('Preserve delivery archive')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(set(files+[index])):tar.add(path,arcname=str(path.relative_to(base)),recursive=False)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        manifest_sha256=sha(index),files=len(inventory),ledger=ledger)))


if __name__=='__main__':main()
