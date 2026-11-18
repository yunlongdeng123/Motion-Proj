"""正式控制器退出后，唯一 raw179 field/fresh-native-basis 持久化核验。"""
from pathlib import Path
import argparse
import json
import os
import sys

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
RUN = Q / 'gpu_20261119'
sys.path[:0] = [str(Q / 'code'), str(Path(__file__).parent), str(Q / 'pixel_render/code')]
from training_factory import default_config, prepare_inputs, load_native_adapter
from native_basis_adapter import scope_signature
from persistent_pixel_reload import verify_reload_179
from field_training_state import file_sha


def main(output, execute):
    if not execute:
        print(json.dumps(dict(execution_default=False, native_forward_budget=2, optimizer_updates=0,
                              input='formal_r1/M11/C800/field.pt', raw=179)))
        return
    completed = RUN / 'formal_r1/completed.json'
    done = json.loads(completed.read_text())
    if done.get('completed') is not True or done.get('formal_physical_updates') != 3400:
        raise RuntimeError('正式实验尚未完整收口')
    formal_pid = json.loads((RUN / 'formal_r1/started.json').read_text())['PID']
    if Path('/proc/%s' % formal_pid).exists():
        raise RuntimeError('正式进程尚未退出，拒绝第二个 native ctx')
    config = default_config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare_inputs(config, device='cuda')
    authorization = dict(schema='pixel_native_basis_smoke.v1', phase='native_smoke',
        native_allowed=True, gpu_authorized=True, formal_allowed=False, frames=[179],
        scope_world_rows_sha256=scope_signature(prepared.active_world_rows, prepared.qualified),
        delete_actor_ids=[10], max_native_forward_calls=2,
        parent_user_authorization=config['authorization'],
        parent_formal_receipt=str(completed), parent_formal_sha256=file_sha(completed),
        root_instruction='postformal raw179 persistence only; no optimization or new camera scope')
    authpath = output / 'authorization.json'
    authpath.write_text(json.dumps(authorization, indent=2) + '\n')
    field = RUN / 'formal_r1/M11/C800/field.pt'
    expected = RUN / 'formal_r1/M11/C800/rendered/179.npz'
    adapter = None
    try:
        adapter = load_native_adapter(config)
        if adapter.ctx.native_calls != 0:
            raise RuntimeError('新 ctx 已产生未登记的 native calls')
        result = verify_reload_179(field, prepared, adapter, output / 'validation',
                                  authorization=authpath, expected_native_npz=expected)
        if result['native_forward_calls'] != 2:
            raise RuntimeError('实际 native 调用账不等于 2')
        (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if adapter is not None:
            adapter.ctx.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUN / 'field_training/persistence_179_r1')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    main(args.output, args.execute)
