"""Finite native-GPU review of preserved pilot checkpoints.

Rendering makes zero updates. Restore parity uses three disposable updates per
selected arm, outside every training budget, and never writes into a pilot.
The update body is extracted from the pinned actual train.py AST, not rederived.
"""
from pathlib import Path
import argparse, ast, copy, gc, hashlib, json, random, sys, time, traceback
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ACTORS = [5, 26, 18, 6, 37]


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False))


def read(p):
    return json.loads(p.read_text())


def implementation(arm):
    return OUT / ('isolated_body_implementation' if arm == 'five_raw_body_isolated' else 'implementation')


def capture(static, instances, optimizer, environment, sampler):
    import torch
    import numpy as np
    return dict(background=static.capture(), instances=instances.capture(optimizer),
        environment=environment.capture(), sampler=sampler.getstate(),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore_rng(saved, sampler):
    import torch
    import numpy as np
    sampler.setstate(saved['sampler'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])


def load(saved, target):
    from runtime import load_models
    target.mkdir()
    return load_models(saved['spec'], target, freeze_background=False, restore=saved)


def gradient_audit(local, target):
    import torch
    from utils.hybrid30k_losses import diagnose_gradients
    p, s, m, result = [local[k] for k in ('package', 'static', 'instances', 'result')]
    named = dict(actor_xyz=m._xyz, actor_scale=m._scaling, actor_opacity=m._opacity,
        actor_color=m._features_dc, static_xyz=s._xyz, static_scale=s._scaling,
        static_opacity=s._opacity, static_color=s._features_dc,
        static_velocity=s._velocity, static_time=s._t)
    preserved = [v for v in p.values() if torch.is_tensor(v) and v.requires_grad]
    metrics = diagnose_gradients(result, named, preserve_grad_tensors=preserved)
    active = {k: v for k, v in named.items() if v.requires_grad}
    before = [(v, v.grad, None if v.grad is None else v.grad.clone()) for v in preserved if v.is_leaf or v.retains_grad]
    try:
        terms = dict(visible_ID=local['mask_loss'], box_boundary=local['boundary'])
        for name, value in terms.items():
            grads = torch.autograd.grad(value, list(active.values()), allow_unused=True, retain_graph=True)
            for key, grad in zip(active, grads):
                metrics[name + '/' + key] = dict(connected=grad is not None,
                    finite=True if grad is None else bool(torch.isfinite(grad).all()),
                    l2=0. if grad is None else float(torch.linalg.vector_norm(grad)))
    finally:
        with torch.no_grad():
            for v, original, val in before:
                if original is not None:
                    original.copy_(val)
                v.grad = original
    raw = p['dynamic_alpha']
    visible = p['dynamic_visible_alpha']
    state_alpha = local['state']['dynamic_alpha']
    isolated = state_alpha is raw
    checks = dict(state_uses_isolated_tensor=isolated,
        state_uses_visible_tensor=state_alpha is visible,
        isolated_visible_max_abs_difference=float((raw - visible).abs().max()),
        isolated_body_MSE=float((raw - local['union']).square().mean()),
        visible_union_MSE=float((visible - local['union']).square().mean()),
        active_weight=result.weights['dynamic_opacity'],
        all_gradient_metrics_finite=all(v for k, v in metrics.items() if k.endswith('_finite')),
        body_static_gradient_l2={k: metrics.get('grad_hybrid/dynamic_opacity/' + k + '/weighted_l2') for k in named if k.startswith('static_')})
    if isolated:
        assert all(v in (None, 0.) for v in checks['body_static_gradient_l2'].values()), 'isolated body alpha must not depend on static geometry'
    write(target / 'loss_gradient.json', dict(status='MEASURED_ACTUAL_CALL_PATH', frame=local['camera'].image_name,
        checks=checks, gradient_metrics=metrics, human_verdict=None))


def make_update(impl, models, effective, sampler, step, hook):
    sys.path.insert(0, str(impl))
    import torch
    import numpy as np
    import kornia
    from runtime import masks_for_frame
    from renderer import render, VISIBILITY_SEMANTICS
    from losses import supervision_contract, local_box_boundary
    from train import instance_mask_objective
    from utils.hybrid30k_losses import compose_hybrid_loss
    from utils.paper_losses import nps_loss, boundary_distance, sky_loss, sparse_inverse_depth_loss
    from utils.loss_utils import ssim
    from utils.general_utils import get_expon_lr_func
    config, scene, static, instances, optimizer, environment, exclusion = models
    args = SimpleNamespace(**effective['arguments'])
    spec = effective['spec']
    policy = dict(config)
    policy['hybrid_lambda_dynamic_opacity'] = args.raw_body_opacity_weight
    policy['hybrid_lambda_dynamic_rgb'] = 5.
    with np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False) as b:
        selected_indices = np.flatnonzero(b['mask_known'].any(axis=0)).tolist()
    selected_ids = [instances.instance_ids[i] for i in selected_indices]
    cameras = scene.getTrainCameras()
    repair = spec.get('background_repair')
    xyz_schedule = get_expon_lr_func(lr_init=config.position_lr_init, lr_final=config.position_lr_final,
        lr_delay_mult=config.position_lr_delay_mult, max_steps=config.iterations)
    monitor = SimpleNamespace(set_phase=lambda *a, **kw: None)
    source = (impl / 'train.py').read_text()
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'camera' for t in n.targets))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If) and 'oneupSHdegree' in ast.unparse(n))
    nodes = copy.deepcopy(loop.body[start:end + 1])
    for i, n in enumerate(nodes):
        if isinstance(n, ast.Expr) and ast.unparse(n) == 'loss.backward()':
            nodes.insert(i, ast.parse('audit_objectives(dict(locals(), static=static, instances=instances))').body[0])
            break
    else:
        raise AssertionError('actual backward node missing')
    nodes += ast.parse("return {'loss':float(loss.detach()),'frame':camera.image_name}").body
    function = ast.FunctionDef(name='actual_update', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=nodes, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    ns = dict(locals(), audit_objectives=hook)
    exec(compile(module, str(impl / 'train.py') + ':actual_update_slice', 'exec'), ns)
    return ns['actual_update'], dict(train_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_start_line=loop.body[start].lineno, source_end_line=loop.body[end].end_lineno,
        update_math_rewritten=False, inserted_read_only_gradient_hook=True)


def run_restore(arm, target):
    import torch
    from restore_contract import verify_runtime_restore, assert_exact_state
    from train import save_checkpoint
    ck = OUT / 'pilots' / arm / 'checkpoint_000300.pt'
    effective = read(OUT / 'pilots' / arm / 'effective_config.json')
    original = (ck.stat().st_size, ck.stat().st_mtime_ns)
    golden = torch.load(ck, map_location='cpu', weights_only=False)
    saved = torch.load(ck, map_location='cuda', weights_only=False)
    models = load(saved, target / 'continuous')
    sampler = random.Random(0)
    restore_rng(golden, sampler)
    cfg, scene, static, instances, optimizer, environment, exclusion = models
    initial = verify_runtime_restore(golden, static, instances, optimizer, environment, sampler)
    update, source = make_update(implementation(arm), models, effective, sampler, 301,
        lambda local: gradient_audit(local, target))
    first = update()
    intermediate = save_checkpoint(target, 301, static, instances, optimizer, environment, sampler,
        effective['spec'], exclusion, training_policy=effective['training_policy'])
    update, _ = make_update(implementation(arm), models, effective, sampler, 302, lambda local: None)
    continuous = update()
    expected_path = target / 'continuous_302_state.pt'
    torch.save(capture(static, instances, optimizer, environment, sampler), expected_path)
    del update, saved, golden, models, cfg, scene, static, instances, optimizer, environment, exclusion
    gc.collect(); torch.cuda.empty_cache()
    golden = torch.load(intermediate, map_location='cpu', weights_only=False)
    saved = torch.load(intermediate, map_location='cuda', weights_only=False)
    models = load(saved, target / 'restored')
    cfg, scene, static, instances, optimizer, environment, exclusion = models
    restore_rng(golden, sampler)
    exact = verify_runtime_restore(golden, static, instances, optimizer, environment, sampler)
    update, _ = make_update(implementation(arm), models, effective, sampler, 302, lambda local: None)
    repeated = update()
    expected = torch.load(expected_path, map_location='cpu', weights_only=False)
    parity_error = None
    try:
        counts = assert_exact_state(expected, capture(static, instances, optimizer, environment, sampler))
    except Exception as e:
        counts = None
        parity_error = repr(e)
    assert original == (ck.stat().st_size, ck.stat().st_mtime_ns)
    result = dict(status='PASSED_RESTORE_AND_EXACT_NEXT_UPDATE' if parity_error is None else 'EXACT_RESTORE_PASSED_NEXT_UPDATE_VARIATION_REQUIRES_REVIEW',
        arm=arm, initial_runtime_restore=initial, intermediate_runtime_restore=exact,
        actual_train_update_source=source, continuous_301=first, continuous_302=continuous, restored_302=repeated,
        next_update_exact=parity_error is None, parity_error=parity_error, comparison_counts=counts,
        disposable_optimizer_updates=3, formal_or_pilot_training_updates=0,
        original_checkpoint_unchanged=True, original_checkpoint=str(ck),
        human_verdict=None, quality_pass=False)
    write(target / 'receipt.json', result)
    return result


def save_png(path, tensor):
    import numpy as np
    from PIL import Image
    array = tensor.detach().clamp(0, 1).cpu()
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.permute(1, 2, 0)
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    Image.fromarray(np.rint(array.numpy() * 255).astype(np.uint8)).save(path)


def render_all(arm, target):
    import torch
    from renderer import render
    ck = OUT / 'pilots' / arm / 'checkpoint_000300.pt'
    original = (ck.stat().st_size, ck.stat().st_mtime_ns)
    saved = torch.load(ck, map_location='cuda', weights_only=False)
    models = load(saved, target / 'model_load')
    cfg, scene, static, instances, optimizer, env, exclusion = models
    tracks = list(instances.instance_ids)
    actors = ACTORS[:len(tracks)]
    bg = torch.zeros(3, device='cuda')
    records = []
    with torch.no_grad():
        for index, camera in enumerate(scene.getTrainCameras()):
            fid = Path(camera.image_name).stem
            folder = target / fid
            folder.mkdir()
            full = render(camera, static, instances, cfg, bg, env_map=env, instance_ids=tracks)
            for name, key in [('full', 'render'), ('static', 'render_s'), ('dynamic', 'raw_render_d'),
                              ('static_alpha', 'static_alpha'), ('dynamic_alpha', 'dynamic_alpha')]:
                save_png(folder / (name + '.png'), full[key])
            save_png(folder / 'observed.png', camera.original_image)
            for i, (actor, track) in enumerate(zip(actors, tracks)):
                deleted = render(camera, static, instances, cfg, bg, env_map=env,
                    instance_ids=[track], edits={track: {'delete': True}})
                assert deleted['instance_visible_contribution'].count_nonzero() == 0
                assert torch.equal(deleted['render_s'], full['render_s'])
                isolated = render(camera, static, instances, cfg, bg, env_map=env, instance_ids=[track],
                    edits={other: {'delete': True} for other in tracks if other != track})
                for name, tensor in [('delete_full', deleted['render']), ('delete_dynamic', deleted['raw_render_d']),
                                     ('isolated', isolated['raw_render_d']), ('isolated_alpha', isolated['dynamic_alpha']),
                                     ('visible_alpha', full['instance_visible_contribution'][i:i+1])]:
                    save_png(folder / ('a%02d_%s.png' % (actor, name)), tensor)
                records.append(dict(frame_id=fid, actor=actor, deleted_visible_alpha_exact_zero=True,
                    static_before_after_bitwise_equal=True, isolated_has_other_actors_removed=True))
            write(target / 'progress.json', dict(arm=arm, frames=index + 1, total=49))
            print(json.dumps(dict(arm=arm, frames=index + 1)), flush=True)
    assert original == (ck.stat().st_size, ck.stat().st_mtime_ns)
    result = dict(status='COMPLETE_49FRAMES_QUALITY_REVIEW_PENDING', arm=arm, frames=49,
        actors=actors, raw128_included=True, records=records, optimizer_updates=0,
        original_checkpoint_unchanged=True, all_frames_are_training_views=True, human_verdict=None,
        static_delete_semantics='same static geometry and appearance; native static RGB is exactly invariant to actor deletion')
    write(target / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['three_replay', 'five_replay', 'five_raw_body', 'five_ID_weight', 'five_raw_body_isolated'])
    parser.add_argument('--mode', required=True, choices=['restore', 'renders'])
    args = parser.parse_args()
    sys.path.insert(0, str(implementation(args.arm)))
    from priority import require_gpu_slot
    slot, admission = require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    target = OUT / 'pilot_review' / args.arm / args.mode
    try:
        if target.exists():
            raise FileExistsError('preserve previous review attempt; diagnose before retry')
        target.mkdir(parents=True)
        import torch
        torch.set_num_threads(2)
        result = run_restore(args.arm, target) if args.mode == 'restore' else render_all(args.arm, target)
        print(json.dumps(dict(status=result['status'], arm=args.arm, mode=args.mode)), flush=True)
    except BaseException:
        if target.exists():
            write(target / 'failure.json', dict(error=traceback.format_exc(), assets_preserved=True))
        raise
    finally:
        slot.close()


if __name__ == '__main__':
    main()
