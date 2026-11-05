"""Read-only Gaussian topology diagnostics; no tensor or RNG mutations.

Usage: handle = install_hooks(GaussianModel, run / 'topology', lambda: iteration)
Call handle.assert_healthy() at checkpoints and handle.close() at shutdown.
NPZ files contain per-initial-lineage counts, never per-row geometry.  A lineage
is an initialization family, not an object label or a Gaussian birth step.
"""
from __future__ import annotations

import functools
import inspect
import json
import sys
import weakref
from pathlib import Path

import numpy as np


FOCUS_LINEAGES = (528407, 132314, 520617, 192988, 42783)
_INSTALLATIONS = {}


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _unique_indices(value):
    # Tensor.unique does not change the input or consume RNG.
    return value.unique() if hasattr(value, 'detach') else np.unique(value)


def _lineages(model, indices):
    lineage = getattr(model, '_sgcm_lineage_ids', None)
    if lineage is None:
        return None
    return _numpy(lineage[indices]).astype(np.int64, copy=False).reshape(-1)


def _hist(values):
    if values is None:
        return None
    ids, counts = np.unique(values, return_counts=True)
    return ids.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def _hist_summary(hist):
    if hist is None:
        return {'available': False}
    ids, counts = hist
    # Stable tie order: np.unique has already sorted lineage IDs.
    order = np.argsort(-counts, kind='stable')[:10]
    focus = {}
    for key in FOCUS_LINEAGES:
        at = int(np.searchsorted(ids, key))
        focus[str(key)] = int(counts[at]) if at < len(ids) and ids[at] == key else 0
    return dict(available=True, rows=int(counts.sum()), families=len(ids),
                top10=[{'initial_lineage': int(ids[i]), 'rows': int(counts[i])}
                       for i in order],
                focus=focus)


class _Hooks:
    def __init__(self, model_class, out_dir, get_step):
        self.model_class = model_class
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.get_step = get_step
        self.errors = []
        self._originals = {}
        self._models = weakref.WeakKeyDictionary()
        self._seq = 0
        self._closed = False
        self._stream = (self.out_dir / 'events.jsonl').open('x', encoding='utf-8')
        self._stream.write(json.dumps(dict(schema='gaussian_topology_counts.v1',
            kind='header', focus_lineages=FOCUS_LINEAGES,
            semantics='initial lineage counts; not object ownership; no RNG calls',
            histogram_keys='NPZ prefix + _ids/_counts; parents count distinct source rows; children count appended rows',
            logging_error_policy='preserve original method return; assert_healthy must be checked')) + '\n')
        self._stream.flush()

    def _error(self, where, error):
        item = dict(where=where, error=repr(error))
        self.errors.append(item)
        # A broken diagnostic must be visible, while leaving training math alone.
        try:
            print('DIAG_HOOK_ERROR ' + json.dumps(item), file=sys.stderr, flush=True)
        except Exception:
            pass

    def _context(self):
        result = {}
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None:
                name = frame.f_code.co_name
                local = frame.f_locals
                if name in ('densify_and_clone', 'densify_and_split', 'densify_and_prune'):
                    if 'op' not in result:
                        result['op'] = {'densify_and_clone': 'ordinary_clone',
                                        'densify_and_split': 'ordinary_split',
                                        'densify_and_prune': 'ordinary_prune'}[name]
                    for key in ('time_clone', 'time_split', 'N', 'joint_sample', 'prune_only'):
                        if key in local and key not in result:
                            result[key] = local[key]
                    if name == 'densify_and_split' and '_split_inputs' not in result:
                        # References are used only before postfix mutates topology;
                        # no caller frame or tensor is retained in the event log.
                        result['_split_inputs'] = {key: local.get(key) for key in (
                            'padded_grad', 'grad_threshold', 'scale_factor',
                            'scene_extent', 'selected_time_mask', 'selected_pts_mask')}
                if name == 'refine' and Path(frame.f_code.co_filename).name == 'paper_sgcm.py':
                    result.update(op='sgcm', clone_ids=local.get('clone_ids'),
                                  split_ids=local.get('split_ids'), prune_mask=local.get('prune'))
                    break
                frame = frame.f_back
        finally:
            del frame
        result.setdefault('op', 'other')
        return result

    def _split_triggers(self, model, context, record, histograms):
        inputs = context.get('_split_inputs', {})
        required = ('padded_grad', 'grad_threshold', 'scale_factor',
                    'scene_extent', 'selected_pts_mask')
        missing = [key for key in required if inputs.get(key) is None]
        if missing:
            raise RuntimeError('ordinary split trigger locals missing: ' + ','.join(missing))
        selected = inputs['selected_pts_mask']
        scales = model.get_scaling
        maximum = (scales.max(dim=1).values if hasattr(scales, 'detach')
                   else np.max(scales, axis=1))
        # Read-only repetition of GaussianModel's original spatial criterion
        # (:583 and :591-593). Do not use children scales or mutate caller masks.
        spatial = ((inputs['padded_grad'] >= inputs['grad_threshold']) &
                   (maximum > model.percent_dense * inputs['scene_extent'] *
                    inputs['scale_factor']))
        temporal = inputs.get('selected_time_mask')
        if temporal is None:
            temporal = selected & ~selected
        if len(selected) != len(model.get_xyz) or len(spatial) != len(selected) or len(temporal) != len(selected):
            raise RuntimeError('ordinary split trigger masks do not match pre-append rows')
        masks = dict(spatial=selected & spatial,
                     temporal=selected & temporal,
                     both=selected & spatial & temporal,
                     spatial_only=selected & spatial & ~temporal,
                     temporal_only=selected & temporal & ~spatial,
                     neither=selected & ~(spatial | temporal))
        for name, mask in masks.items():
            histograms['split_' + name + '_parents'] = _hist(_lineages(model, mask))
        record['split_trigger_diagnostic'] = dict(
            available=True, counts_are='final selected parent rows; spatial and temporal overlap in both',
            spatial_rule='padded_grad>=grad_threshold AND max(parent_scale)>percent_dense*scene_extent*scale_factor',
            grad_threshold=float(inputs['grad_threshold']),
            temporal_mask_present=inputs.get('selected_time_mask') is not None,
            time_center_sampling='ordinary split always draws samples_t and sets new_t; no_time_split only restores sigma_t',
            no_extra_random_draws=True)

    def _before(self, model, kind, values):
        context = self._context()
        if model not in self._models:
            self._models[model] = '%s-%03d' % (type(model).__name__, len(self._models) + 1)
        record = dict(step=int(self.get_step()), model=self._models[model], kind=kind,
                      op=context['op'], rows_before=int(len(model.get_xyz)),
                      lineage_available=getattr(model, '_sgcm_lineage_ids', None) is not None)
        for key in ('time_clone', 'time_split', 'N', 'joint_sample', 'prune_only'):
            if key in context:
                record[key] = context[key]
        for key in ('no_time_split', 't_grad'):
            if hasattr(model, key):
                record[key] = bool(getattr(model, key))
        histograms = {}
        if kind == 'append':
            parents, child_count = values
            record['appended_rows'] = int(child_count)
            if parents is not None:
                histograms['all_children'] = _hist(_lineages(model, parents))
                distinct = _unique_indices(parents)
                histograms['all_parents'] = _hist(_lineages(model, distinct))
                record['distinct_parent_rows'] = int(len(distinct))
            if context['op'] == 'ordinary_split':
                try:
                    self._split_triggers(model, context, record, histograms)
                except Exception as error:
                    record['split_trigger_diagnostic'] = dict(available=False, error=repr(error))
                    self._error('ordinary_split:triggers', error)
            if context['op'] == 'sgcm':
                for label, repeats in (('clone', 1), ('split', 2)):
                    ids = context[label + '_ids']
                    if ids is not None:
                        h = _hist(_lineages(model, ids))
                        histograms['sgcm_' + label + '_parents'] = h
                        histograms['sgcm_' + label + '_children'] = (
                            None if h is None else (h[0], h[1] * repeats))
                        record['sgcm_' + label + '_parent_rows'] = int(len(ids))
        else:
            mask = values
            record['removed_rows'] = int(_numpy(mask).astype(bool, copy=False).sum())
            histograms['all_removed'] = _hist(_lineages(model, mask))
            if context['op'] == 'ordinary_split':
                record['op'] = 'ordinary_split_parent_remove'
            elif context['op'] == 'sgcm':
                # SGCM's final mask combines distance pruning and split-parent
                # replacement. These disjoint reasons are recorded separately.
                split_ids = context['split_ids']
                prune_mask = context['prune_mask']
                if split_ids is not None:
                    histograms['sgcm_split_parents_removed'] = _hist(_lineages(model, split_ids))
                if prune_mask is not None:
                    prefix = getattr(model, '_sgcm_lineage_ids', None)
                    histograms['sgcm_distance_pruned'] = _hist(
                        None if prefix is None else _numpy(prefix[:len(prune_mask)][prune_mask]))
        return record, histograms

    def _after(self, model, observation):
        if observation is None:
            return
        record, histograms = observation
        record['rows_after'] = int(len(model.get_xyz))
        record['row_delta'] = record['rows_after'] - record['rows_before']
        self._seq += 1
        record['event'] = self._seq
        arrays = {}
        record['histograms'] = {}
        for name, hist in histograms.items():
            record['histograms'][name] = _hist_summary(hist)
            if hist is not None:
                arrays[name + '_ids'], arrays[name + '_counts'] = hist
        if arrays:
            filename = 'event_%06d.npz' % self._seq
            np.savez_compressed(self.out_dir / filename, **arrays)
            record['family_counts_file'] = filename
        self._stream.write(json.dumps(record, sort_keys=True) + '\n')
        self._stream.flush()

    def _patch(self, name, kind):
        original = getattr(self.model_class, name)
        signature = inspect.signature(original)
        # Keep whether the attribute was inherited, so uninstall restores the
        # original class dictionary rather than leaving a copied base method.
        self._originals[name] = (name in self.model_class.__dict__, original)

        @functools.wraps(original)
        def wrapped(model, *args, **kwargs):
            observation = None
            try:
                bound = signature.bind(model, *args, **kwargs)
                bound.apply_defaults()
                if kind == 'append':
                    values = (bound.arguments.get('sgcm_parent_indices'),
                              len(bound.arguments['new_xyz']))
                else:
                    values = bound.arguments['mask']
                observation = self._before(model, kind, values)
            except Exception as error:
                self._error(name + ':before', error)
            # Do not catch, replace, retry, or otherwise alter original errors.
            result = original(model, *args, **kwargs)
            try:
                self._after(model, observation)
            except Exception as error:
                self._error(name + ':after', error)
            return result

        setattr(self.model_class, name, wrapped)

    def assert_healthy(self):
        if self.errors:
            raise RuntimeError('Topology diagnostic failed: ' + json.dumps(self.errors))

    def uninstall(self):
        for name, (owned, original) in self._originals.items():
            if owned:
                setattr(self.model_class, name, original)
            else:
                delattr(self.model_class, name)
        self._originals.clear()
        _INSTALLATIONS.pop(self.model_class, None)

    def close(self):
        if not self._closed:
            self.uninstall()
            self._stream.close()
            self._closed = True


def install_hooks(model_class, out_dir, get_step):
    """Patch append/prune only; get_step must be a side-effect-free callable.

    The ordinary/SGCM caller stack supplies operation names and actual split
    flags. No additional model buffers, optimizer fields or random draws are
    introduced. Each call's original return value and original exceptions pass
    through unchanged. Diagnostic failures are recorded in handle.errors.
    """
    if model_class in _INSTALLATIONS:
        raise RuntimeError('Topology hooks already installed on this class')
    for name in ('densification_postfix', 'prune_points'):
        if not callable(getattr(model_class, name, None)):
            raise TypeError('Gaussian model is missing ' + name)
    if not callable(get_step):
        raise TypeError('get_step must be callable')
    handle = _Hooks(model_class, out_dir, get_step)
    try:
        handle._patch('densification_postfix', 'append')
        handle._patch('prune_points', 'prune')
    except Exception:
        handle.close()
        raise
    _INSTALLATIONS[model_class] = handle
    return handle
