"""只读整理本轮 owned 来源；不导入 Torch、不改科学 source 或共享 docs。"""
from pathlib import Path
import hashlib
import json

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
F = Q / 'gpu_20261119/field_training'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bound = json.loads((Q / 'gpu_20261119/formal_r1/input_bindings.json').read_text())
    core = ['field_training_state.py', 'material_readers.py', 'build_training_inputs.py',
            'native_training_adapter.py', 'training_factory.py']
    auxiliary = ['test_field_training_state.py', 'audit_factory_cpu.py', 'persistent_pixel_reload.py',
                 'run_persistent_179.py', 'freeze_owned_sources.py']
    sources = {}
    for name in core + auxiliary:
        path = F / 'code' / name
        sources[name] = dict(path=str(path), sha256=sha(path),
            role='executed_formal_core' if name in core else 'CPU_test_or_postformal_auxiliary',
            bound_to_formal=name in core, unchanged_since_formal_start=(bound.get(str(path)) == sha(path)) if name in core else None)
    if not all(sources[name]['unchanged_since_formal_start'] for name in core):
        raise RuntimeError('正式 core source 已改变')
    result = dict(schema='teacher_detail_owned_training_sources.v1',
        parent_task='MODULE2-TEACHER-DETAIL-CPU-20261119', role='authorized_GPU_overlay',
        sources=sources, documentation=dict(path=str(F / 'README.md'), sha256=sha(F / 'README.md')),
        CPU_tests=dict(path=str(F / 'code/test_field_training_state.py'), passed=6,
            remote_environment='/root/autodl-tmp/envs/motionproj', scene_optimizer_updates=0),
        common_mask_audit=str(F / 'common_support_audit_actual.json'),
        root_smoke_receipt=str(Q / 'gpu_20261119/lifecycle_r1/receipt.json'),
        root_formal_receipt=str(Q / 'gpu_20261119/formal_r1/completed.json'),
        postformal_persistence_receipt=str(F / 'persistence_179_r1/receipt.json'),
        owned_postformal_native_forward_calls=2, owned_postformal_optimizer_updates=0,
        scientific_quality_acceptance='independent evaluation and image review; not inferred from engineering checks',
        default_scene_modified=False, automatic_pending_controller=False, GPU_power_action=False)
    output = F / 'owned_source_inventory.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), sources=len(sources), formal_core_unchanged=True)))


if __name__ == '__main__':
    main()
