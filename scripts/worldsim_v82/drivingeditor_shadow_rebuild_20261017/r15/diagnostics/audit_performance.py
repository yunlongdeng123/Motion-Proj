"""R15 4k 只读性能诊断；所有试验只作用于内存副本，不保存模型。"""
import argparse
import collections
import contextlib
import cProfile
import functools
import gc
import io
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
BASE = ROOT / 'backend_restore_r15'
SOURCE = ROOT / 'implementation_r15_train_v3/shadow_rebuild_v3'
sys.path.insert(0, str(SOURCE))
from shadow_rebuild.config import Config, LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.objective import objective
from shadow_rebuild.train import seed_from_state
import shadow_rebuild.tracing as tracing
import shadow_rebuild.raster as raster
import shadow_rebuild.model as model_module


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sync_time(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    saved = torch.load(BASE / 'paused_checkpoint_004000_20261017/checkpoint.pt', map_location='cpu', weights_only=False)
    raw = saved['config'].copy()
    raw['weights'] = LossWeights(**raw['weights'])
    cfg = Config(**raw).validate()
    data = Dataset(BASE / 'data', 'cuda', True)
    model = SceneModel(seed_from_state(saved['state']), cfg).cuda()
    model.load_state_dict(saved['state'])
    model.set_stage(saved['stage'])
    assert saved['input_contract'] == data.manifest
    gen = torch.Generator().manual_seed(0)
    result = dict(run_id='R15_PAUSED_PERFORMANCE_ATTRIBUTION_20261017', checkpoint_step=saved['global_step'],
                  production_SGD_updates=0, diagnostic_parameter_updates=0, production_source_modified=False,
                  torch_version=torch.__version__, GPU=torch.cuda.get_device_name(0), cases=[],
                  scope='Same checkpoint/frame; synchronized forward/backward; no optimizer parameter update. Ablations are diagnostics, not accepted replacements.')
    frame, load_seconds = sync_time(lambda: data.frame(18))
    result['data_load_seconds'] = load_seconds
    original_checkpoint = tracing.checkpoint
    original_integral = tracing._precision_integral
    original_aabb = tracing._ray_aabb
    counts = collections.Counter()

    @functools.wraps(original_integral)
    def counted_integral(o, d, m, p, a, *extra, **kw):
        counts['integral_calls'] += 1
        counts['integral_ray_gaussian_pairs'] += len(o) * len(m)
        return original_integral(o, d, m, p, a, *extra, **kw)

    @functools.wraps(original_aabb)
    def counted_aabb(*a, **kw):
        counts['aabb_calls'] += 1
        counts['aabb_ray_tests'] += len(a[0])
        return original_aabb(*a, **kw)

    @functools.wraps(original_checkpoint)
    def counted_checkpoint(function, *a, **kw):
        counts['checkpoint_calls'] += 1
        return original_checkpoint(function, *a, **kw)

    last = {}
    train_image = model.train_image

    def capture_image(p, g):
        image = train_image(p, g)
        last['image'] = image
        return image

    model.train_image = capture_image
    # 只预热主相机 CUDA 路径；不进行优化器更新。
    cfg.shadows = False
    loss, _, package = objective(model, frame, len(data.actor_ids), 'cuda', 'geometry', gen, None, 4000)
    loss.backward()
    del loss, package
    model.zero_grad(set_to_none=True)
    last.clear()
    cfg.shadows = True
    reference_image = None
    reference_grad = None
    variants = [('baseline', {}), ('shadow_off', {'shadows': False}),
                ('sun_samples_1', {'sun_samples': 1}),
                ('gaussian_chunk_2048', {'gaussian_chunk': 2048}),
                ('checkpoint_disabled', {'disable_checkpoint': True})]
    for label, modifications in variants:
        old = {k: getattr(cfg, k) for k in modifications if k != 'disable_checkpoint'}
        for k, v in modifications.items():
            if k != 'disable_checkpoint':
                setattr(cfg, k, v)
        repeats = 2 if label != 'checkpoint_disabled' else 1
        for repeat in range(repeats):
            record = dict(case=label, repeat=repeat + 1, changes=modifications)
            tracing._precision_integral = counted_integral
            tracing._ray_aabb = counted_aabb
            tracing.checkpoint = (lambda function, *a, **kw: function(*a)) if modifications.get('disable_checkpoint') else counted_checkpoint
            counts.clear()
            model.zero_grad(set_to_none=True)
            last.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                (loss, terms, package), forward = sync_time(lambda: objective(model, frame, len(data.actor_ids), 'cuda', 'geometry', gen, None, 4000))
                record['forward_counts'] = dict(counts)
                _, backward = sync_time(loss.backward)
                record.update(forward_seconds=forward, backward_seconds=backward,
                              forward_backward_seconds=forward + backward,
                              loss=float(loss.detach()), peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                              total_counts_including_recomputation=dict(counts))
                image = last['image'].detach().cpu()
                rows = torch.linspace(0, len(model.field.xyz) - 1, 2048, device='cuda').long()
                grad = model.field.xyz.grad[rows].detach().cpu()
                if reference_image is None:
                    reference_image, reference_grad = image, grad
                record['image_max_abs_delta_from_baseline_linear'] = float((image - reference_image).abs().max())
                record['image_MAE_delta_from_baseline_linear'] = float((image - reference_image).abs().mean())
                record['sampled_xyz_gradient_max_abs_delta'] = float((grad - reference_grad).abs().max())
                record['sampled_xyz_gradient_finite'] = bool(torch.isfinite(grad).all())
                if label == 'baseline' and repeat == 1:
                    _, record['gradient_clip_seconds'] = sync_time(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True))
                    _, record['screen_gradient_accumulation_seconds'] = sync_time(lambda: model.field.accumulate_screen_gradients(package))
                    # lr=0 只计量 Adam 状态更新/内核；主参数逐位不发生 SGD 更新。
                    optimizer = model.optimizer('geometry')
                    optimizer.load_state_dict(saved['optimizer'])
                    for group in optimizer.param_groups:
                        group['lr'] = 0.0
                    _, record['zero_lr_Adam_seconds'] = sync_time(optimizer.step)
                    del optimizer
                record['status'] = 'COMPLETE_ZERO_PARAMETER_UPDATE'
            except RuntimeError as error:
                record.update(status='FAILED', error=str(error)[:2000])
            finally:
                loss = terms = package = None
                model.zero_grad(set_to_none=True)
                last.clear()
                gc.collect()
                torch.cuda.empty_cache()
            result['cases'].append(record)
            write(out / 'performance.json', result)
            print(json.dumps(record), flush=True)
        for k, v in old.items():
            setattr(cfg, k, v)
    tracing.checkpoint = original_checkpoint
    tracing._precision_integral = original_integral
    tracing._ray_aabb = original_aabb

    # 模块边界使用同步计时，避免把异步工作错误归到下一个模块。
    components = collections.defaultdict(list)
    raster_calls = []
    patches = []

    def patch(obj, name, label):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*a, **kw):
            value, seconds = sync_time(lambda: original(*a, **kw))
            components[label].append(seconds)
            return value
        patches.append((obj, name, original))
        setattr(obj, name, wrapped)

    original_raster = raster.cuda_raster
    def measured_raster(camera, xyz, cov, opacity, channels, capacity=10, screens=None):
        value, seconds = sync_time(lambda: original_raster(camera, xyz, cov, opacity, channels, capacity, screens))
        raster_calls.append(dict(points=len(xyz), channels=channels.shape[-1], seconds=seconds,
                                 kernel_passes=(channels.shape[-1]+capacity-1)//capacity if len(xyz) else 0))
        return value
    raster.cuda_raster = measured_raster
    patch(model.field, 'evaluate', 'field_evaluate')
    patch(model, 'sky', 'environment_sky')
    patch(model, 'train_image', 'physical_shading')
    patch(model.light, 'visibility', 'visibility')
    profile = cProfile.Profile()
    profile.enable()
    with torch.no_grad():
        (loss, terms, package), seconds = sync_time(lambda: objective(model, frame, len(data.actor_ids), 'cuda', 'geometry', gen, None, 4000))
    profile.disable()
    text = io.StringIO()
    pstats.Stats(profile, stream=text).sort_stats('cumulative').print_stats(70)
    (out / 'cpu_profile.txt').write_text(text.getvalue())
    result['instrumented_no_grad_forward'] = dict(total_seconds=seconds,
        components={k:sum(v) for k,v in components.items()}, raster_calls=raster_calls,
        raster_seconds_by_kind={kind:sum(r['seconds'] for r in raster_calls if (r['channels']>10 if kind=='scene_multichannel' else r['channels']==1 if kind=='per_actor_alpha' else r['channels']==9)) for kind in ['scene_multichannel','per_actor_alpha','isolated_branches']},
        warning='Nested component times must not be added together; instrumented no-grad is not full training throughput')
    result['shadow_cache_present'] = True  # EvaluatedGeometry explicitly has __dict__.
    for obj, name, original in reversed(patches):
        setattr(obj, name, original)
    raster.cuda_raster = original_raster
    del loss, terms, package
    last.clear()
    gc.collect()
    torch.cuda.empty_cache()
    write(out / 'performance.json', result)
    print('COMPONENT_PROFILE_COMPLETE', flush=True)

    # 采样一个完整无梯度前向的 CPU/CUDA 算子，单列 profiler 自身扰动。
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False) as prof:
            with torch.no_grad():
                loss, terms, package = objective(model, frame, len(data.actor_ids), 'cuda', 'geometry', gen, None, 4000)
                torch.cuda.synchronize()
        events = []
        averages = prof.key_averages()
        for event in averages:
            events.append(dict(key=event.key, count=event.count,
                self_cpu_time_us=event.self_cpu_time_total, cpu_time_us=event.cpu_time_total,
                self_cuda_time_us=getattr(event, 'self_cuda_time_total', 0), cuda_time_us=getattr(event, 'cuda_time_total', 0)))
        result['operator_profile'] = sorted(events, key=lambda r:r['self_cpu_time_us'], reverse=True)[:80]
        (out / 'operator_profile.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=60))
        del loss, terms, package, prof
    except Exception as error:
        result['operator_profile_error'] = repr(error)
    result['status'] = 'COMPLETE_TRAINING_REMAINS_PAUSED'
    write(out / 'performance.json', result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
