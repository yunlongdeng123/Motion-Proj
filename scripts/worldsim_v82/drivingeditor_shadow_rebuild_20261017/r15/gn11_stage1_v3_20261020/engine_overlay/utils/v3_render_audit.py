"""只读渲染/容量审计；CPU 不创建 CUDA event，计时不参与优化。"""

from collections import Counter
from contextlib import contextmanager
import time
from types import SimpleNamespace

import torch


RENDER_AUDIT_VERSION = 'V3_RENDER_DENSIFY_ROAD_R1_20261020'
RENDER_AUDIT_POLICY = dict(
    version=RENDER_AUDIT_VERSION, profile_interval=100, road_visibility_interval=5000,
    profile_SGCM_due_iterations=True,
    screen_atol=2e-5, screen_rtol=2e-4,
    topology_budget_semantics='native_pre_densify_trigger_not_new_hard_truncation',
    fixed_road_statistics='excluded_from_generic_densification',
    timing_scope='iteration_start_through_optimizer_SGCM_excludes_visualization_eval_checkpoint',
)
ACTIVE_AUDIT = None


class RenderAudit:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.running = False
        self.cumulative_calls = Counter()

    def begin(self, iteration, gate=False, topology_due=False):
        global ACTIVE_AUDIT
        self.iteration, self.gate = int(iteration), bool(gate)
        self.topology_due = bool(topology_due)
        self.profile = gate or topology_due or iteration == 1 or iteration % RENDER_AUDIT_POLICY['profile_interval'] == 0
        self.calls, self.spans = Counter(), []
        self.running = True
        self.memory_start = None
        if self.cuda and self.profile:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                self.memory_start = torch.cuda.memory_allocated(self.device)
        self.started = time.perf_counter()
        ACTIVE_AUDIT = self

    @contextmanager
    def span(self, label):
        if not self.profile:
            yield
            return
        start_event = end_event = None
        if self.cuda:
            with torch.cuda.device(self.device):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if end_event is not None:
                with torch.cuda.device(self.device):
                    end_event.record()
            self.spans.append((label, (time.perf_counter()-started)*1000, start_event, end_event))

    def finish(self):
        global ACTIVE_AUDIT
        if not self.running:
            raise RuntimeError('render audit has no active iteration')
        if self.cuda and self.profile:
            torch.cuda.synchronize(self.device)
        times = {}
        for label, wall, start, end in self.spans:
            value = times.setdefault(label, dict(calls=0, cpu_dispatch_wall_ms=0., cuda_stream_elapsed_ms=None))
            value['calls'] += 1
            value['cpu_dispatch_wall_ms'] += wall
            if start is not None:
                value['cuda_stream_elapsed_ms'] = (value['cuda_stream_elapsed_ms'] or 0.)+start.elapsed_time(end)
        record = dict(version=RENDER_AUDIT_VERSION, iteration=self.iteration, gate=self.gate,
                      SGCM_due=self.topology_due,
                      backend='actual_cuda_events' if self.cuda else 'cpu_only_no_cuda_measurement',
                      profiled=self.profile, raster_calls=dict(self.calls), spans=times,
                      inclusive_spans_must_not_be_summed=True,
                      wall_ms=(time.perf_counter()-self.started)*1000,
                      timing_scope=RENDER_AUDIT_POLICY['timing_scope'],
                      memory_start_allocated_bytes=self.memory_start,
                      memory_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.cuda and self.profile else None,
                      memory_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.cuda and self.profile else None,
                      memory_is_whole_profiled_iteration_not_attributed_to_one_module=True)
        self.cumulative_calls.update(self.calls)
        record['cumulative_raster_calls_all_training_iterations'] = dict(self.cumulative_calls)
        self.running = False
        if ACTIVE_AUDIT is self:
            ACTIVE_AUDIT = None
        return record


@contextmanager
def audit_span(label):
    audit = ACTIVE_AUDIT
    if audit is None or not audit.running:
        yield
    else:
        with audit.span(label):
            yield


def raster_call(label, rasterizer, **kwargs):
    if ACTIVE_AUDIT is not None and ACTIVE_AUDIT.running:
        ACTIVE_AUDIT.calls[label] += 1
    with audit_span('raster/'+label):
        return rasterizer(**kwargs)


@torch.no_grad()
def capacity_snapshot(static, dynamic, args, expected_road_ids):
    ids = static._road_ids()
    roads = ids[ids >= 0]
    if not torch.equal(roads.sort().values, expected_road_ids.to(roads).sort().values):
        raise RuntimeError('fixed road identity lost, duplicated or replaced by topology/SGCM')
    static_count, road_count = len(static.get_xyz), len(roads)
    dynamic_count = len(dynamic.get_xyz)
    cap, dynamic_cap = int(args.densify_until_num_points), int(args.dynamic_densify_until_num_points)
    return dict(static_total=static_count, ordinary_background=static_count-road_count,
                fixed_road=road_count, dynamic=dynamic_count, static_trigger_cap=cap,
                dynamic_trigger_cap=dynamic_cap, fixed_road_fraction_of_static=road_count/max(static_count, 1),
                fixed_road_fraction_of_static_cap=road_count/cap if cap > 0 else None,
                static_trigger_overshoot=max(static_count-cap, 0) if cap > 0 else None,
                dynamic_trigger_overshoot=max(dynamic_count-dynamic_cap, 0) if dynamic_cap > 0 else None,
                budget_semantics=RENDER_AUDIT_POLICY['topology_budget_semantics'], fixed_road_ids_preserved=True)


def expected_screen_buffers(losses, proxies, road_mask, autograd_fn):
    """零更新gate的独立参考：实际图逐项求导，恢复retain_grad副作用。"""
    with audit_span('gate/screen_reference_autograd'):
        previous = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in proxies]
        base = autograd_fn(losses[0], proxies, proxies)
        body = autograd_fn(losses[1], [proxies[0]], proxies)[0]
        road = autograd_fn(losses[2], [proxies[1]], proxies)[0]
        expected = [old+(g.detach() if g is not None else 0) for old, g in zip(previous, base)]
        if body is not None:
            body = body.detach().clone()
            if road_mask is not None:
                body[:road_mask.numel()][road_mask] = 0
            expected[0] += body
        if road is not None:
            expected[1] += road.detach()
        return expected


@torch.no_grad()
def check_screen_buffers(proxies, expected):
    result = {}
    for label, proxy, reference in zip(['joint', 'static', 'dynamic'], proxies, expected):
        if proxy.grad is None or not torch.isfinite(proxy.grad).all() or not torch.allclose(
                proxy.grad, reference, atol=RENDER_AUDIT_POLICY['screen_atol'], rtol=RENDER_AUDIT_POLICY['screen_rtol']):
            raise RuntimeError('routed retained screen buffer numeric mismatch: '+label)
        result[label] = dict(max_absolute_error=float((proxy.grad-reference).abs().max()),
                             points=len(proxy), nonleaf=not proxy.is_leaf)
    return result


@torch.no_grad()
def check_densification_numeric(model, branch_proxy, branch_visible, branch_radii,
                                 joint_proxy, joint_visible, joint_radii, accumulate_fn):
    """真实梯度在临时统计中运行融合；不污染训练统计、拓扑或Adam。"""
    count = len(model.get_xyz)
    scratch = SimpleNamespace(get_xyz=model.get_xyz, _t=model._t,
        road_unit_mask=getattr(model, 'road_unit_mask', torch.zeros(count, dtype=torch.bool, device=branch_proxy.device)),
        xyz_gradient_accum=branch_proxy.new_zeros(count, 1), denom=branch_proxy.new_zeros(count, 1),
        max_radii2D=branch_radii.new_zeros(count), t_gradient_accum=branch_proxy.new_zeros(count, 1))
    accumulate_fn(scratch, branch_proxy, branch_visible, branch_radii, joint_proxy, joint_visible, joint_radii)
    b, j = branch_visible.bool(), joint_visible.bool()
    active = (b | j) & ~scratch.road_unit_mask
    bg = branch_proxy.grad[:, :2].norm(dim=1)
    jg = joint_proxy.grad[:, :2].norm(dim=1)
    expected = torch.where(b & j, torch.maximum(bg, jg), torch.where(b, bg, jg))
    expected = torch.where(active, expected, torch.zeros_like(expected))
    radius = torch.where(active, torch.maximum(branch_radii, joint_radii), torch.zeros_like(branch_radii))
    for actual, reference in [(scratch.xyz_gradient_accum[:, 0], expected),
                              (scratch.denom[:, 0], active.to(branch_proxy)), (scratch.max_radii2D, radius)]:
        if not torch.allclose(actual, reference, atol=RENDER_AUDIT_POLICY['screen_atol'], rtol=RENDER_AUDIT_POLICY['screen_rtol']):
            raise RuntimeError('branch/joint max or single-view denominator numeric mismatch')
    return dict(points=count, active_nonroad_points=int(active.sum()), fixed_road_excluded=int(scratch.road_unit_mask.sum()),
                max_gradient_error=float((scratch.xyz_gradient_accum[:, 0]-expected).abs().max()),
                denominator_exact=torch.equal(scratch.denom[:, 0], active.to(branch_proxy)))
