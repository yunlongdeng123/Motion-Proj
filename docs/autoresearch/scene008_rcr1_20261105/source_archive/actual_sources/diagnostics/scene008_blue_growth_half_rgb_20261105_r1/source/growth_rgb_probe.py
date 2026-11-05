"""Halve only the RGB part of ordinary static XY growth statistics.

Both arms compute the same auxiliary gradients. They never enter main loss or
Adam. The candidate subtracts half the existing isolated static RGB vector;
the main static RGB weight2, image/domain, non-RGB XY and temporal source stay.
The joint RGB auxiliary vector is still calculated in both arms, but unused.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _retained_slots(result):
    unique = {}
    for tensor in _tensors(result):
        if tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad):
            unique[id(tensor)] = tensor
    return [(tensor, tensor.grad,
             None if tensor.grad is None else tensor.grad.detach().clone())
            for tensor in unique.values()]


def _restore_slots(slots):
    with torch.no_grad():
        for tensor, original, saved in slots:
            if original is not None:
                original.copy_(saved)
            tensor.grad = original


def _parameter_slots(models):
    unique = {}
    for model in models:
        optimizer = getattr(model, 'optimizer', None) if model is not None else None
        if optimizer is not None:
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    unique[id(parameter)] = parameter
    return [(parameter, parameter.grad,
             None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in unique.values()]


def _assert_parameter_slots(slots):
    for parameter, original, saved in slots:
        if parameter.grad is not original:
            raise RuntimeError('Auxiliary RGB gradients replaced an optimizer gradient slot')
        if original is not None and not torch.equal(parameter.grad, saved):
            raise RuntimeError('Auxiliary RGB gradients changed optimizer gradient values')


class GrowthRGBProbe:
    def __init__(self, mode, output, start_iteration, extra_updates):
        if mode not in ('control', 'half_rgb'):
            raise ValueError(mode)
        self.mode = mode
        self.output = Path(output)
        self.first_iteration = int(start_iteration) + 1
        self.needs_main_graph = True
        self.latest = None
        self._prepared_iteration = None
        self._static_rgb_gradient = None
        self._joint_rgb_gradient = None
        self._models = None
        contract = dict(
            schema='scene008.ordinary_static_xy_rgb_half_weight_probe.v1',
            native_arm='native_ownership', growth_rgb_mode=mode,
            source_checkpoint_iteration=int(start_iteration),
            extra_updates=int(extra_updates), stop_iteration=int(start_iteration) + int(extra_updates),
            applied_gradient=('g_total' if mode == 'control' else 'g_total - 0.5 * g_static_RGB'),
            main_static_RGB_weight='1 + hybrid_lambda_static_rgb (=2); unchanged',
            statistics_static_RGB_weight=('2 (original)' if mode == 'control' else '1 (half of original2)'),
            joint_aux_computed_but_not_applied=True,
            test_scope='ordinary static XY RGB weight2 to1; not restoration of the old loss, image source, or temporal signal',
            RGB_domain='same background = not dilated visible_reviewed union and dynamic union == 0',
            aux_in_both_arms=True, aux_added_to_main_loss=False,
            unchanged=['main loss and weights', 'RGB image/domain (isolated static)', 'non-RGB XY growth vector',
                       'temporal growth source', 'SGCM', 'densification thresholds and normalization',
                       'sampler and learning-rate schedule', 'saved resolved training config'],
        )
        (self.output / 'growth_probe_contract.json').write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def _auxiliary_gradients(self, terms, result, evidence, target, union, args):
        from utils.hybrid30k_losses import photometric_rgb_loss
        from utils.loss_utils import ssim

        collar = F.max_pool2d(evidence['visible_reviewed'].any(0)[None, None].float(), 5, 1, 2)[0].bool()
        background = ~collar & (union == 0)
        guarded = lambda value: torch.where(background, value, 0)
        joint_RGB = (1 + args.hybrid_lambda_static_rgb) * photometric_rgb_loss(
            guarded(result['render']), guarded(target), ssim, args.lambda_dssim)
        slots = _retained_slots(result)
        try:
            static_gradient, = torch.autograd.grad(
                terms['static_rgb'], result['static_growth'], retain_graph=True, allow_unused=False)
            joint_gradient, = torch.autograd.grad(
                joint_RGB, result['joint_growth'], retain_graph=True, allow_unused=False)
            static_gradient = static_gradient.detach().clone()
            joint_gradient = joint_gradient[:len(static_gradient)].detach().clone()
        finally:
            _restore_slots(slots)
        if not torch.isfinite(static_gradient).all() or not torch.isfinite(joint_gradient).all():
            raise RuntimeError('Nonfinite auxiliary RGB screen gradient')
        return static_gradient, joint_gradient

    def prepare(self, terms, result, evidence, target, union, args, iteration,
                static, dynamic, environment):
        iteration = int(iteration)
        self._models = (static, dynamic, environment)
        if self._prepared_iteration is not None and iteration != self._prepared_iteration + 1:
            raise RuntimeError('Growth RGB probe received non-consecutive updates')
        before = _parameter_slots(self._models) if iteration == self.first_iteration else None
        static_rgb, joint_rgb = self._auxiliary_gradients(terms, result, evidence, target, union, args)
        if before is not None:
            _assert_parameter_slots(before)
        self._static_rgb_gradient = static_rgb
        self._joint_rgb_gradient = joint_rgb
        self._prepared_iteration = iteration
        self.needs_main_graph = iteration == self.first_iteration

    def after_backward(self, terms, result, evidence, target, union, args, iteration):
        iteration = int(iteration)
        if iteration != self._prepared_iteration:
            raise RuntimeError('Main backward did not match prepared RGB gradients')
        if iteration != self.first_iteration:
            return
        # The first main backward retains its graph only for this audit. Re-run
        # auxiliary reads while real Adam gradients exist, then prove their
        # identities and every value are unchanged. All retained slots restore.
        before = _parameter_slots(self._models)
        retained_before = _retained_slots(result)
        self._auxiliary_gradients(terms, result, evidence, target, union, args)
        _assert_parameter_slots(before)
        for tensor, original, saved in retained_before:
            if tensor.grad is not original or (original is not None and not torch.equal(tensor.grad, saved)):
                raise RuntimeError('Auxiliary reads failed to restore a retained gradient slot')
        audit = dict(status='PASS_EXACT_OPTIMIZER_GRADIENT_NEUTRALITY', iteration=iteration,
                     growth_rgb_mode=self.mode, optimizer_parameter_slots=len(before),
                     optimizer_slots_with_gradient=sum(original is not None for _, original, _ in before),
                     retained_slots_verified=len(retained_before),
                     optimizer_gradient_max_abs_difference=0.0,
                     optimizer_gradient_slot_identity_unchanged=True,
                     aux_added_to_main_loss=False)
        (self.output / 'growth_first_step_optimizer_gradient_audit.json').write_text(
            json.dumps(audit, indent=2, allow_nan=False) + '\n')
        self.needs_main_graph = False

    def statistics_tensor(self, result, iteration):
        if int(iteration) != self._prepared_iteration:
            raise RuntimeError('Growth statistics requested without matching auxiliary gradients')
        source = result['static_growth']
        total = source.grad
        if total is None:
            raise RuntimeError('Main static screen gradient is missing')
        if total.shape != self._static_rgb_gradient.shape or total.shape != self._joint_rgb_gradient.shape:
            raise RuntimeError('Static RGB growth vectors do not align with native row order')
        mixed = (total - 0.5 * self._static_rgb_gradient).detach()
        applied = total.detach() if self.mode == 'control' else mixed
        if not torch.isfinite(applied).all():
            raise RuntimeError('Nonfinite applied static XY growth gradient')
        # Both arms allocate the same proxy. Its data are the same zero screen
        # tensor; only .grad is read by the unchanged native accumulator.
        proxy = source.detach().clone().requires_grad_(True)
        proxy.grad = applied
        self.latest = dict(iteration=int(iteration), mode=self.mode,
                           g_total=total.detach(), g_static_RGB=self._static_rgb_gradient,
                           g_joint_RGB=self._joint_rgb_gradient,
                           g_mixed=mixed, g_applied=applied)
        return proxy
