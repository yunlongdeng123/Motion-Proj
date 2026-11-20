"""Necessary zero-update common BUILD initialization; caller-owned GPU lease."""
from pathlib import Path
import json
import random
import sys
import time
import traceback
import numpy as np
from build_inputs import B, ROOT, FRAMES, actual_frames, objective_classes, install_paths, read, write, sha


def initialize():
    install_paths()
    cpu = read(ROOT / 'CPU_inputs.json')
    if not cpu.get('completed') or any(sha(p) != h for p, h in cpu['source']['files'].items()):
        raise RuntimeError('CPU-qualified all-context input identity differs')
    common = ROOT / 'common'
    common.mkdir(exist_ok=False)
    auth = dict(gpu_authorized=True, native_allowed=True, optimizer_updates=0,
                authorization='Root NEW_TASK explicitly permits new caller-owned BUILD initialization and subsequent fixed experiments',
                data_role='BUILD_fit_all_qualified', protocol=str(ROOT / 'protocol.json'),
                protocol_sha256=sha(ROOT / 'protocol.json'), keep_GPU_on=True)
    write(common / 'authorization.json', auth)
    lease = ROOT / 'gpu_lease.json'
    write(lease, dict(gpu_authorized=True, active=True, owner='all_teacher_build',
                      authorization=str(common / 'authorization.json'), phase='common_initialization'))
    t = time.monotonic()
    ctx = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(101)
        random.seed(101)
        np.random.seed(101)
        from native_capacity_inputs import make_context, field_and_reader
        ctx = make_context(lease, 'all_teacher_build')
        model, reader, field_binding = field_and_reader(ctx.base_rgb, 'cuda')
        if len(ctx.active_world_rows) != 151988 or int(reader.eligible.sum()) != 89887:
            raise RuntimeError('registered topology/scope changed')
        with torch.no_grad():
            ci = reader.colors(model).detach().clone()
        with np.load(B / 'formal_r1/common_initial_ci.npz', allow_pickle=False) as z:
            if not np.array_equal(z['active_world_rows'], ctx.active_world_rows) or not np.array_equal(z['eligible_active'], reader.eligible):
                raise RuntimeError('common row identity changed')
            initial_max = float(np.max(np.abs(ci.cpu().numpy() - z['ci0'])))
        if initial_max != 0:
            raise RuntimeError('BUILD GPU ci0 must exactly equal the original actual GPU common ci0')
        frames, data_binding = actual_frames()
        model.save(common / 'initial_field.pt', dict(role='fresh_common_A200_L_H0_no_BUILD_updates', field_binding=field_binding))
        np.savez_compressed(common / 'initial_ci.npz', ci0=ci.cpu().numpy(), active_world_rows=ctx.active_world_rows,
                            eligible_active=reader.eligible)
        predictions = {}
        (common / 'initial').mkdir()
        with torch.no_grad():
            for raw in FRAMES:
                value = ctx.render_road(raw, ci)
                if not np.array_equal(value['alpha'].detach().cpu().numpy().reshape(1280, 1920), frames[raw]['alpha']):
                    raise RuntimeError('common native alpha differs from immutable masks')
                predictions[raw] = value['rgb']
                np.savez_compressed(common / 'initial' / ('%03d_native.npz' % raw),
                                    road_rgb=value['rgb'].cpu().numpy(), road_alpha=frames[raw]['alpha'], K=ctx.K_masks[raw])
        BuildLoss, _ = objective_classes()
        objective = BuildLoss(frames, device='cuda')
        write(common / 'common_loss_snapshot.json', objective.calibrate(predictions))
        old_binding = read(B / 'formal_r1/source_freeze.json')['binding']
        if any(sha(p) != h for p, h in old_binding.items()):
            raise RuntimeError('original method/native source changed')
        binding = dict(old_binding)
        binding.update(data_binding['files'])
        for path in (Path(__file__), ROOT / 'code/build_inputs.py', ROOT / 'protocol.json', ROOT / 'CPU_inputs.json'):
            binding[str(path)] = sha(path)
        for path in (common / 'initial_ci.npz', common / 'initial_field.pt', common / 'common_loss_snapshot.json'):
            binding[str(path)] = sha(path)
        write(common / 'source_freeze.json', dict(binding=binding, field_binding=field_binding, data_binding=data_binding))
        ctx.assert_frozen()
        result = dict(completed=True, data_role='BUILD_fit_all_qualified', ci0_original_GPU_max_abs=initial_max,
                      ci0_exact_original=True, active_rows=151988, eligible_rows=89887,
                      initial_ci=str(common / 'initial_ci.npz'), normalizers=str(common / 'common_loss_snapshot.json'),
                      source_freeze=str(common / 'source_freeze.json'), actual_nativeF=ctx.native_calls,
                      actual_nativeB=0, optimizer_updates=0, seconds=time.monotonic() - t,
                      original_QUERY_results_unchanged=True, old_public_F0_reusable=True, GPU_keep_on=True)
        write(common / 'receipt.json', result)
        print(json.dumps(result), flush=True)
    except Exception:
        write(common / 'failed.json', dict(error=traceback.format_exc(), seconds=time.monotonic() - t))
        raise
    finally:
        if ctx is not None:
            ctx.close()
        state = read(lease)
        state.update(active=False, ended_at=time.time())
        write(lease, state)


if __name__ == '__main__':
    if '--execute' not in sys.argv:
        print(json.dumps(dict(default_GPU=False, output=str(ROOT / 'common'))))
    else:
        initialize()
