"""Opt-in separation of conservative input hiding and caller-authorized output.

H_input may expand the existing appearance mask. W_output is an independent,
explicit caller authorization. H_input outside W_output is input-only context,
unless the caller separately marks pending_output. Remaining predicted effects
are diagnostics, not implicit authorization, pending edits, or removal claims.
"""
from __future__ import annotations

import copy
import numpy as np
from .shadow_probability import rgb_sha256


REGION_MODE = 'independent'
BUNDLE_SCHEMA = 'independent_region_input_bundle.v1'


def _explicit_mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != bool:
        raise ValueError(f'{name} must be an explicit Boolean mask on the native/output camera grid')
    return array.copy()


def validate_independent_regions(masks, shape):
    """Validate canonical roles and old aliases; never infer pending from H-W."""
    names = ['H_input', 'M_material', 'W_output', 'pending_output', 'input_only',
             'remaining_effect', 'K', 'target_unknown', 'W_edit', 'shadow_pending', 'C_mat']
    roles = {}
    for name in names:
        if name not in masks:
            raise ValueError(f'independent region contract is missing {name}')
        roles[name] = _explicit_mask(masks[name], shape, name)
    h, w, k, pending = [roles[name] for name in ['H_input', 'W_output', 'K', 'pending_output']]
    if (w & ~h).any():
        raise ValueError('W_output must be a subset of H_input')
    if (w & k).any() or (h & k).any():
        raise ValueError('independent input/write masks conflict with protected K')
    if (w & pending).any():
        raise ValueError('authorized W_output and pending_output must be disjoint')
    if (w & roles['remaining_effect']).any():
        raise ValueError('remaining_effect describes effects outside authorized W_output')
    if (h & ~roles['M_material']).any():
        raise ValueError('hidden RGB cannot remain valid material context')
    expected = {'target_unknown': h, 'W_edit': w, 'shadow_pending': pending,
                'C_mat': ~roles['M_material'], 'input_only': h & ~w & ~pending}
    for name, value in expected.items():
        if not np.array_equal(roles[name], value):
            raise ValueError(f'independent role alias or partition differs: {name}')
    return roles


def apply_independent_regions(layers, *, H_input, W_output, pending_output,
                              M_material=None, remaining_effect=None):
    """Return new NativeLayers with independently authorized output regions.

    H_input is the full requested hiding region and may only expand the existing
    target_unknown. M_material adds exclusions to the existing material policy;
    old forbidden material can never be restored. pending_output is mandatory,
    including an explicit all-false mask when no output edit remains pending.
    W_output may be smaller than H_input without blocking authorized completion.
    """
    observed = np.asarray(layers['rgb'])
    if observed.ndim != 3 or observed.shape[2] != 3 or observed.dtype != np.uint8:
        raise ValueError('independent regions require actual native uint8 RGB layers')
    shape = observed.shape[:2]
    old = layers['masks']
    h = _explicit_mask(H_input, shape, 'H_input')
    w = _explicit_mask(W_output, shape, 'W_output')
    pending = _explicit_mask(pending_output, shape, 'pending_output')
    original_h = _explicit_mask(old['target_unknown'], shape, 'original target_unknown')
    if (original_h & ~h).any():
        raise ValueError('H_input may expand the existing hiding mask, never shrink it')
    requested_material = np.zeros(shape, bool) if M_material is None else _explicit_mask(M_material, shape, 'M_material')
    remaining = np.zeros(shape, bool) if remaining_effect is None else _explicit_mask(remaining_effect, shape, 'remaining_effect')
    material_forbidden = ~_explicit_mask(old['C_mat'], shape, 'original C_mat') | requested_material | h
    result = copy.deepcopy(layers)
    masks = result['masks']
    masks.update(H_input=h, M_material=material_forbidden, W_output=w,
                 pending_output=pending, remaining_effect=remaining,
                 input_only=h & ~w & ~pending, target_unknown=h.copy(), W_edit=w.copy(),
                 shadow_pending=pending.copy(), C_mat=~material_forbidden,
                 visible_support=~material_forbidden, retained_context=~h)
    validate_independent_regions(masks, shape)
    receipt = dict(mode=REGION_MODE, schema='independent_input_output_regions.v1',
        completion_scope='authorized_output_only', caller_explicit_write_authorization=True,
        pending_output_explicit=True, input_only_is_not_pending=True,
        H_input_pixels=int(h.sum()), H_input_added_pixels=int((h & ~original_h).sum()),
        M_material_pixels=int(material_forbidden.sum()),
        W_output_pixels=int(w.sum()), input_only_pixels=int(masks['input_only'].sum()),
        pending_output_pixels=int(pending.sum()), remaining_effect_pixels=int(remaining.sum()),
        remaining_effect_is_prediction_evidence_only=True,
        all_target_effects_removed_claim=False, geometry_and_depth_roles_unchanged=True,
        existing_material_exclusions_preserved=True, loss_support_unchanged=True,
        neutral_rgb=np.asarray(layers.get('neutral_rgb', (127, 127, 127)), np.uint8).tolist())
    result.setdefault('receipt', {})['independent_regions'] = receipt
    result['receipt']['observation_sha256'] = rgb_sha256(observed)
    if 'object_shadow' in result['receipt']:
        result['receipt']['object_shadow'].update(
            region_mode=REGION_MODE, authorized_write_pixels=int(w.sum()),
            input_hiding_pixels=int(h.sum()), pending_pixels=int(pending.sum()),
            write_region_source='explicit caller W_output; not inferred from H_input',
            completion_scope='authorized_output_only', all_target_effects_removed_claim=False,
            completion_status='pending_output_edits' if pending.any() else 'authorized_regions_ready_for_model')
    return result
