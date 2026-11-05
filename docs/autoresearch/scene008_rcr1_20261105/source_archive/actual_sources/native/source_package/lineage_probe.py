"""Read-only ordinary-growth events for one persistent initial lineage.

Install after the training integration and before training():
    handle = install(train, GaussianModel, output_dir, lineage=64990)
    handle.growth_probe = growth_probe       # optional RGB-vector provenance
    handle.set_context(camera=camera, time_shift=shift)  # in record_step

Only an active, non-dynamic densification_and_optimization call is observed.
SGCM calls outside that scope pass through every original method untouched.
Actual postfix parent indices and actual prune masks establish what happened;
recomputed gate fields merely explain the native clone/split decisions.
No RNG, autograd, optimizer, tensor mutation, extra rendering, or resampling.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
from pathlib import Path

import torch


_INSTALLED = {}


def _json(value):
    if torch.is_tensor(value):
        return _json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class _LineageProbe:
    def __init__(self, train, model_class, output_dir, lineage, camera_getter):
        self.train = train
        self.model_class = model_class
        self.lineage = int(lineage)
        self.camera_getter = camera_getter
        self.growth_probe = None
        self.errors = []
        self._patches = []
        self._context = {}
        self._active = None
        self._cycle = None
        self._operation = None
        self._sequence = 0
        self._closed = False
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f'lineage_{self.lineage}_ordinary_events.jsonl'
        self._stream = self.path.open('x', encoding='utf-8')
        self._write('header', schema='scene008.ordinary_lineage_probe.v1',
                    lineage=self.lineage, initial_source_row=65911 if self.lineage == 64990 else None,
                    semantics='Persistent initial lineage, not current compacted row or object ownership',
                    selection_authority='Actual postfix parent indices and actual prune masks',
                    gradient_semantics='grads are native normalized inputs; split gt is signed, clone uses its norm',
                    scope='Ordinary static densification only; SGCM wrappers and methods unchanged')
        try:
            self._install()
        except BaseException:
            self.close()
            raise

    def _write(self, kind, **fields):
        self._sequence += 1
        record = dict(event=self._sequence, kind=kind, lineage=self.lineage)
        if self._active is not None:
            record.update(self._active['context'])
        record.update(fields)
        self._stream.write(json.dumps(_json(record), ensure_ascii=False, allow_nan=False) + '\n')
        self._stream.flush()

    def _safe(self, where, function, *args, **kwargs):
        try:
            with torch.no_grad():
                return function(*args, **kwargs)
        except Exception as error:
            self.errors.append(dict(where=where, error=repr(error)))
            return None

    def assert_healthy(self):
        if self.errors:
            raise RuntimeError('Lineage probe failed: ' + json.dumps(self.errors[-5:]))

    def set_context(self, *, camera=None, time_shift=None, iteration=None):
        """Read scalar camera context; never keep renderer tensors or graphs."""
        context = dict(time_shift=time_shift)
        if camera is not None:
            if isinstance(camera, dict):
                context.update(camera)
            else:
                name = str(getattr(camera, 'image_name', getattr(camera, 'image_path', '')))
                context.update(camera=name, camera_timestamp=getattr(camera, 'timestamp', None))
                tail = Path(name).stem[-3:]
                if tail.isdigit():
                    context['raw'] = int(tail)
        if iteration is not None:
            context['iteration'] = int(iteration)
        self._context = _json(context)

    def _observing(self, model):
        return (not self._closed and self._active is not None
                and self._active['model'] is model)

    def _rows(self, model):
        ids = getattr(model, '_sgcm_lineage_ids', None)
        if ids is None:
            raise RuntimeError('Static checkpoint lacks persistent lineage metadata')
        return torch.nonzero(ids == self.lineage, as_tuple=False).flatten()

    def _family(self, model):
        rows = self._rows(model)
        return dict(count=len(rows), rows=rows.detach().cpu().tolist())

    @staticmethod
    def _row_values(value, rows, padded=False):
        """Gather only watched rows; ordinary split pads new clone rows with zero."""
        if value is None:
            return None
        if padded:
            result = value.new_zeros((len(rows), *value.shape[1:]))
            valid = rows < len(value)
            result[valid] = value.detach()[rows[valid]]
            return result
        return value.detach()[rows]

    def _statistics(self, model):
        rows = self._rows(model)
        denom = model.denom.detach()[rows].reshape(-1)
        xy = model.xyz_gradient_accum.detach()[rows].reshape(-1)
        gt = model.t_gradient_accum.detach()[rows].reshape(-1)
        # Match native NaN->0 normalization without editing its accumulators.
        norm_xy = xy / denom
        norm_gt = gt / denom
        norm_xy = torch.where(torch.isnan(norm_xy), 0., norm_xy)
        norm_gt = torch.where(torch.isnan(norm_gt), 0., norm_gt)
        result = []
        values = zip(rows.tolist(), denom.tolist(), xy.tolist(), gt.tolist(),
                     norm_xy.tolist(), norm_gt.tolist(), model.max_radii2D.detach()[rows].tolist())
        for row, den, xys, gts, gxy, signed_gt, radius in values:
            result.append(dict(row=row, denom=den, gxy_accum=xys, signed_gt_accum=gts,
                               normalized_gxy=gxy, normalized_signed_gt=signed_gt,
                               max_radius_before_reset=radius))
        return result

    def _rgb_vectors(self, model):
        latest = getattr(self.growth_probe, 'latest', None)
        if not latest or latest.get('iteration') != self._active['context']['iteration']:
            return None
        rows = self._rows(model)
        return dict(mode=latest.get('mode'), rows=rows.tolist(),
                    vectors={key: latest[key].detach()[rows, :2].cpu().tolist()
                             for key in ('g_total', 'g_static_RGB', 'g_joint_RGB', 'g_mixed', 'g_applied')
                             if key in latest})

    def _gates(self, model, arguments, operation):
        rows = self._rows(model)
        if not len(rows):
            return dict(rows=[], grad_threshold=arguments['grad_threshold'],
                        grad_t_threshold=arguments.get('grad_t_threshold'),
                        no_time_split=bool(model.no_time_split))
        grads = self._row_values(arguments['grads'], rows, padded=True).reshape(len(rows), -1)
        grads_t = self._row_values(arguments.get('grads_t'), rows, padded=True)
        gxy = grads.norm(dim=-1) if operation == 'ordinary_clone' else grads[:, 0]
        signed_gt = None if grads_t is None else grads_t.reshape(len(rows), -1)[:, 0]
        extent = float(arguments['scene_extent'])
        xyz = model._xyz.detach()[rows]
        scale = model.get_scaling.detach()[rows]
        sigma_t = model.get_scaling_t.detach()[rows].max(dim=-1).values
        if model.contract:
            factor = xyz.norm(dim=-1) * extent - 1
            factor = torch.where(factor <= 1, 1., factor) / extent
        else:
            factor = torch.ones_like(gxy) / extent
        scale_threshold = model.percent_dense * extent * factor
        grad_threshold = arguments['grad_threshold']
        spatial = ((gxy >= grad_threshold)
                   & ((scale.max(dim=-1).values <= scale_threshold) if operation == 'ordinary_clone'
                      else (scale.max(dim=-1).values > scale_threshold)))
        temporal_on = bool(arguments.get('time_clone' if operation == 'ordinary_clone' else 'time_split', False))
        temporal = torch.zeros_like(spatial)
        if temporal_on:
            used_gt = grads_t.reshape(len(rows), -1).norm(dim=-1) if operation == 'ordinary_clone' else signed_gt
            temporal = ((used_gt >= arguments['grad_t_threshold'])
                        & ((sigma_t <= model.percent_dense) if operation == 'ordinary_clone'
                           else (sigma_t > model.percent_dense)))
        selected = spatial | temporal if (operation == 'ordinary_clone' or arguments.get('joint_sample', True)) else spatial
        pre = {} if self._cycle is None else self._cycle['pre_stats']
        result = []
        for index, row in enumerate(rows.tolist()):
            result.append(dict(row=row, normalized_gxy=float(gxy[index]),
                               normalized_signed_gt=None if signed_gt is None else float(signed_gt[index]),
                               denom_before_cycle=pre.get(row, {}).get('denom', 0.),
                               accumulator_before_cycle=pre.get(row),
                               gradient_padded_after_clone=row >= len(arguments['grads']),
                               scale=scale[index].tolist(), sigma_t=float(sigma_t[index]),
                               scale_factor=float(factor[index]), scale_threshold=float(scale_threshold[index]),
                               spatial_selected=bool(spatial[index]), temporal_selected=bool(temporal[index]),
                               predicted_selected=bool(selected[index])))
        return dict(rows=result, grad_threshold=grad_threshold,
                    grad_t_threshold=arguments.get('grad_t_threshold'), scene_extent=extent,
                    percent_dense=model.percent_dense, temporal_enabled=temporal_on,
                    temporal_comparison='abs(norm signed_gt)' if operation == 'ordinary_clone' else 'signed_gt',
                    no_time_split=bool(model.no_time_split), joint_sample=arguments.get('joint_sample'),
                    requested_N=arguments.get('N'),
                    effective_N=(int(arguments.get('N', 2)) + int(not model.no_time_split)) if operation == 'ordinary_split' else 1)

    def _patch(self, owner, name, factory):
        original = getattr(owner, name)
        wrapped = factory(original, inspect.signature(original))
        setattr(owner, name, wrapped)
        self._patches.append((owner, name, original, wrapped))

    def _install(self):
        def statistics_factory(original, signature):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                model = bound.arguments['gaussians']
                if self._closed or bound.arguments.get('dynamic', False):
                    return original(*args, **kwargs)
                previous = self._active
                if self.camera_getter is not None:
                    self._safe('camera_getter', lambda: self.set_context(camera=self.camera_getter()))
                self._active = dict(model=model, context=dict(self._context, iteration=int(bound.arguments['iteration'])))
                try:
                    return original(*args, **kwargs)
                finally:
                    self._active = previous
            return wrapped

        def cycle_factory(original, signature):
            @functools.wraps(original)
            def wrapped(model, *args, **kwargs):
                if not self._observing(model):
                    return original(model, *args, **kwargs)
                bound = signature.bind(model, *args, **kwargs)
                bound.apply_defaults()
                previous = self._cycle
                stats = self._safe('cycle_statistics', self._statistics, model) or []
                before = self._safe('cycle_family', self._family, model)
                config = {key: value for key, value in bound.arguments.items() if key != 'self'}
                self._cycle = dict(pre_stats={row['row']: row for row in stats}, config=config)
                self._safe('cycle_before', self._write, 'ordinary_cycle_before', family=before,
                           statistics_before_reset=stats, config=config,
                           rgb_vectors=self._safe('rgb_vectors', self._rgb_vectors, model))
                try:
                    return original(model, *args, **kwargs)
                finally:
                    self._safe('cycle_after', lambda: self._write('ordinary_cycle_after', family=self._family(model)))
                    self._cycle = previous
            return wrapped

        def decision_factory(operation):
            def factory(original, signature):
                @functools.wraps(original)
                def wrapped(model, *args, **kwargs):
                    if not self._observing(model):
                        return original(model, *args, **kwargs)
                    bound = signature.bind(model, *args, **kwargs)
                    bound.apply_defaults()
                    previous = self._operation
                    self._operation = operation
                    before = self._safe(operation + ':family', self._family, model)
                    gates = self._safe(operation + ':gates', self._gates, model, bound.arguments, operation)
                    self._safe(operation + ':before', self._write, operation + '_before', family=before, gates=gates)
                    try:
                        return original(model, *args, **kwargs)
                    finally:
                        self._safe(operation + ':after', lambda: self._write(operation + '_after',
                                   family_before=before, family_after=self._family(model)))
                        self._operation = previous
                return wrapped
            return factory

        def postfix_factory(original, signature):
            @functools.wraps(original)
            def wrapped(model, *args, **kwargs):
                if not self._observing(model):
                    return original(model, *args, **kwargs)
                def before_append():
                    bound = signature.bind(model, *args, **kwargs)
                    parents = bound.arguments.get('sgcm_parent_indices')
                    if parents is None:
                        raise RuntimeError('Native postfix omitted persistent source-parent indices')
                    selected = torch.nonzero(model._sgcm_lineage_ids[parents] == self.lineage, as_tuple=False).flatten()
                    return dict(family_before=self._family(model),
                                parent_rows=parents[selected].detach().cpu().tolist(),
                                child_rows=(selected + len(model.get_xyz)).detach().cpu().tolist())
                event = self._safe('actual_append_before', before_append)
                result = original(model, *args, **kwargs)
                if event is not None:
                    self._safe('actual_append_after', lambda: self._write('actual_append',
                               operation=self._operation, **event, family_after=self._family(model)))
                return result
            return wrapped

        def prune_factory(original, signature):
            @functools.wraps(original)
            def wrapped(model, *args, **kwargs):
                if not self._observing(model):
                    return original(model, *args, **kwargs)
                def before_prune():
                    bound = signature.bind(model, *args, **kwargs)
                    mask = bound.arguments.get('mask')
                    if mask is None:
                        # Preserve compatibility with the native positional mask name.
                        mask = args[0] if args else next(v for k, v in bound.arguments.items() if k != 'self')
                    rows = self._rows(model)
                    removed = rows[mask[rows]]
                    event = dict(family_before=self._family(model), removed_rows=removed.tolist())
                    if self._operation != 'ordinary_split' and self._cycle is not None:
                        cfg = self._cycle['config']
                        opacity = model.get_opacity.detach()[rows].reshape(-1)
                        scales = model.get_scaling.detach()[rows].max(dim=-1).values
                        radius = model.max_radii2D.detach()[rows]
                        extent = float(cfg['extent'])
                        factor = model._xyz.detach()[rows].norm(dim=-1) * extent - 1
                        factor = (torch.where(factor <= 1, 1., factor) / extent
                                  if model.contract else torch.ones_like(scales) / extent)
                        world_limit = model.big_point_threshold * extent * factor
                        event['prune_inputs'] = [dict(row=int(row), actually_pruned=bool(mask[row]),
                            opacity=float(opacity[i]), opacity_below_min=bool(opacity[i] < cfg['min_opacity']),
                            radius=float(radius[i]), max_scale=float(scales[i]),
                            big_scale_threshold=float(world_limit[i]),
                            screen_size_enabled=bool(cfg['max_screen_size']),
                            screen_too_big=bool(cfg['max_screen_size'] and radius[i] > cfg['max_screen_size']),
                            world_too_big=bool(cfg['max_screen_size'] and scales[i] > world_limit[i]))
                            for i, row in enumerate(rows.tolist())]
                    return event
                event = self._safe('actual_prune_before', before_prune)
                result = original(model, *args, **kwargs)
                if event is not None:
                    operation = 'ordinary_split_parent_remove' if self._operation == 'ordinary_split' else 'ordinary_prune'
                    def after_prune():
                        after = self._family(model)
                        removed = set(event['removed_rows'])
                        survivors = [row for row in event['family_before']['rows'] if row not in removed]
                        if len(survivors) != after['count']:
                            raise RuntimeError('Native prune did not preserve the expected family survivors')
                        self._write('actual_prune', operation=operation, **event, family_after=after,
                                    survivor_row_map=list(zip(survivors, after['rows'])))
                    self._safe('actual_prune_after', after_prune)
                return result
            return wrapped

        self._patch(self.train, 'densification_and_optimization', statistics_factory)
        self._patch(self.model_class, 'densify_and_prune', cycle_factory)
        self._patch(self.model_class, 'densify_and_clone', decision_factory('ordinary_clone'))
        self._patch(self.model_class, 'densify_and_split', decision_factory('ordinary_split'))
        self._patch(self.model_class, 'densification_postfix', postfix_factory)
        self._patch(self.model_class, 'prune_points', prune_factory)

    def close(self):
        if self._closed:
            return
        self._safe('footer', self._write, 'footer', errors=self.errors)
        self._closed = True
        for owner, name, original, wrapped in reversed(self._patches):
            if getattr(owner, name) is wrapped:
                setattr(owner, name, original)
        self._stream.close()
        _INSTALLED.pop((id(self.train), id(self.model_class)), None)


def install(train, GaussianModel, output_dir, lineage=64990, camera_getter=None):
    """Wrap originals without changing call arguments, results, or SGCM hooks."""
    key = (id(train), id(GaussianModel))
    if key in _INSTALLED:
        raise RuntimeError('A lineage probe is already installed for this training module')
    handle = _LineageProbe(train, GaussianModel, output_dir, lineage, camera_getter)
    _INSTALLED[key] = handle
    return handle
