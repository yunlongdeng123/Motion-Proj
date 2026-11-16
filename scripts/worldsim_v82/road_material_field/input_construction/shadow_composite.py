"""Strict CPU composition with explicit authorized write and pending regions."""
from __future__ import annotations

import numpy as np
try:
    from .shadow_probability import rgb_sha256
except ImportError:
    from shadow_probability import rgb_sha256


def compose_shadow_prediction(base_original_rgb,raw_prediction,maskdict,*,
                              expected_observation_sha256,
                              allow_diagnostic_pending=False,region_mode='strict'):
    """Use actual observed RGB as the copy source, never a neutralized I_geo.

    ``expected_observation_sha256`` is mandatory and comes from the input bundle
    receipt for this exact grid. Pending pixels normally refuse final output;
    diagnostic mode returns an explicitly unfinished composite for inspection.
    """
    base=np.asarray(base_original_rgb);raw=np.asarray(raw_prediction)
    if base.ndim!=3 or base.shape[2]!=3 or base.dtype!=np.uint8:
        raise ValueError('base_original_rgb must be actual uint8 HxWx3 observation')
    if raw.shape!=base.shape or raw.dtype!=np.uint8:
        raise ValueError('raw prediction must be uint8 RGB on exactly the same grid')
    if not expected_observation_sha256 or rgb_sha256(base)!=expected_observation_sha256:
        raise ValueError('copy source is not the pinned actual observation; do not use sanitized I_geo')
    if region_mode not in ('strict','independent'):
        raise ValueError('region_mode must be strict or explicitly independent')
    shape=base.shape[:2]
    def mask(name,default=None):
        value=maskdict.get(name,default)
        if value is None:raise ValueError(f'required explicit mask is missing: {name}')
        array=np.asarray(value)
        if array.shape!=shape or array.dtype!=bool:
            raise ValueError(f'{name} must be a Boolean mask on the output camera grid')
        return array
    f=mask('target_unknown');w=mask('W_edit');k=mask('K')
    explicit_pending=mask('shadow_pending')
    if region_mode=='independent':
        from .independent_regions import validate_independent_regions
        roles=validate_independent_regions(maskdict,shape)
        pending=roles['pending_output']
    else:
        pending=explicit_pending|(f&~w)
    if (w&~f).any():raise ValueError('write region must be a subset of appearance-context unknown')
    if (w&k).any():raise ValueError('write region conflicts with protected retained objects')
    if (f&k).any():raise ValueError('appearance unknown conflicts with protected retained objects')
    if pending.any() and not allow_diagnostic_pending:
        raise RuntimeError(f'refusing final composite: {int(pending.sum())} unresolved shadow/context pixels')
    out=base.copy();out[w]=raw[w]
    copied=(~w)&(~pending|k)
    receipt=dict(schema='object_shadow_composite.v1',status='unfinished_diagnostic' if pending.any() else 'complete',
                 final_eligible=not bool(pending.any()),remaining_pending=int(pending.sum()),
                 write_pixels=int(w.sum()),protected_pixels=int(k.sum()),
                 eligible_copy_pixels=int(copied.sum()),diagnostic_pending_copy_pixels=int((pending&~w).sum()),
                 copy_source_observation_sha256=expected_observation_sha256,
                 copy_source_is_actual_observation=True,raw_decoder_and_composite_separate=True,
                 calibrated_shadow_claim=False,detection_accuracy_validated=False,
                 pending_copy_policy='not a valid final result' if pending.any() else 'no pending pixels')
    assert np.array_equal(out[k],base[k])
    assert np.array_equal(out[w],raw[w])
    assert np.array_equal(out[~w],base[~w])
    if region_mode=='independent':
        receipt.update(schema='independent_region_composite.v1',region_mode=region_mode,
            completion_scope='authorized_output_only',authorized_output_complete=not bool(pending.any()),
            H_input_pixels=int(roles['H_input'].sum()),input_only_pixels=int(roles['input_only'].sum()),
            remaining_effect_pixels=int(roles['remaining_effect'].sum()),
            remaining_effect_is_prediction_evidence_only=True,all_target_effects_removed_claim=False,
            input_only_does_not_block_composition=True,
            W_outside_copied_from_actual_observation=True,W_inside_uses_raw_decoder_without_base_restore=True)
    return out,receipt
