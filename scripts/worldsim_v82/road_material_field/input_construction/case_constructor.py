"""One assembly entry for cached, training, real-query, and online inputs."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.ndimage import binary_dilation
from .proposal_policy import qualify_ensemble,POLICY
from .input_builder import prepare_native_layers,render_view,_paint_visible
from .adaptive_view import plan_views
from .effects import conservative_effect_candidates
from .input_utility import choose_view
from .shadow_regions import apply_shadow_regions


def make_proposal_reducer(protected_core,policy=None):
    """FastSAM callback: fn(prompt_dict,candidates_KHW,scores_K)->qualified dict.

    The fixed same policy is used on native RAM masks and decoded ROI archives.
    No point support is required when a prompt has no points. Accepted masks
    remain an independent provider layer and cannot be invalidated here.
    """
    protected=np.asarray(protected_core,bool).copy()
    cfg=dict(POLICY if policy is None else policy)
    def reducer(prompt,candidates,scores):
        return qualify_ensemble(candidates,scores,positive_uv=prompt.get('positive_uv',()),
             negative_uv=prompt.get('negative_uv',()),protected_mask=protected,policy=cfg)
    mask_digest=hashlib.sha256(np.asarray(protected.shape,np.int64).tobytes()+np.packbits(protected,bitorder='little').tobytes()).hexdigest()
    reducer.context=dict(policy_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                         protected_sha256=mask_digest,core_sha256=mask_digest)
    return reducer


def construct_case_inputs(rgb,corrupted_semantic,target_core,provider,*,retained_core=None,
                          target_uncertainty=None,view_config=None,selected_view_id=None,
                          selector=None,render_all=False,provenance=None,
                          shadow_mode='legacy',shadow_field=None,shadow_config=None,
                          independent_regions=None):
    """Typed proposals -> roles/effect -> plans -> chosen local/global tensors.

    Semantic labels are from the actual corrupted RGB, Cityscapes mapping.
    ``retained_core`` is optional certified nonedited foreground; if omitted,
    corrupted semantic foreground outside the known target supplies a pixel
    preservation prior. No identity lookup or bounding-box fill is performed.
    No clean GT, true shadow alpha, or supervision field is accepted.
    """
    if shadow_mode not in ('legacy','object_probability'):
        raise ValueError('shadow_mode must be legacy or object_probability')
    if shadow_mode=='object_probability' and shadow_field is None:
        raise ValueError('object_probability requires an actual target-associated ShadowField; no dark-rule fallback')
    if shadow_mode=='legacy' and shadow_field is not None:
        raise ValueError('a ShadowField requires explicit object_probability mode')
    rgb=np.asarray(rgb,np.uint8);core=np.asarray(target_core,bool)
    if shadow_mode=='object_probability' and np.all(rgb[core]==127):
        raise ValueError('shadow predictor needs observed object RGB, not a pre-erased gray core')
    sem=np.asarray(corrupted_semantic)
    if sem.shape!=core.shape or rgb.shape!=(*core.shape,3):raise ValueError('native grids differ')
    road=sem==0;semfg=(sem>=11)&~core
    def field(*names):
        for name in names:
            if name in provider:return np.asarray(provider[name],bool).copy()
        return np.zeros(core.shape,bool)
    qualified=field('qualified_union','qualified')
    disagreement=field('qualified_disagreement','disagreement')
    accepted=field('accepted_union','accepted')&~core
    foreground=semfg|qualified|accepted
    if retained_core is None:
        retained=(semfg|accepted)&~binary_dilation(core,iterations=2)
    else:
        retained=np.asarray(retained_core,bool).copy()
    if shadow_mode=='object_probability' and (retained&core).any():
        raise ValueError('target and certified retained object overlap: resolve identity before editing')
    k=binary_dilation(retained,iterations=4)&~core
    body=binary_dilation(core,iterations=2)&~k
    if shadow_mode=='legacy':
        effect,effect_receipt,effect_parts=conservative_effect_candidates(rgb,core,body,retained,k,foreground|core,road)
    else:
        effect=np.zeros_like(core)
        effect_receipt=dict(mode='object_probability',legacy_luminance_rule_called=False)
        effect_parts=dict(confidence=np.zeros(core.shape,np.float32))
    confidence=provider.get('qualified_confidence',provider.get('confidence'))
    layers=prepare_native_layers(rgb,road,core,semantic_foreground=semfg,accepted_foreground=accepted,
         candidate_foreground=qualified,foreground_uncertainty=disagreement,target_uncertainty=target_uncertainty,
         effect=effect,retained_core=retained,K=k,sam_confidence=confidence,
         effect_confidence=effect_parts['confidence'],provenance=provenance)
    if shadow_mode=='object_probability':
        layers=apply_shadow_regions(layers,shadow_field,observation_rgb=rgb,config=shadow_config)
        effect_receipt=layers['receipt']['object_shadow']
    if independent_regions is not None:
        from .independent_regions import apply_independent_regions
        layers=apply_independent_regions(layers,**independent_regions)
    observed=rgb.copy();observed[layers['masks']['rgb_unknown_core']]=127
    white,yellow=_paint_visible(observed,layers['masks']['geometry_road_visible'])
    plans=plan_views(core.shape,layers['masks'],lane_support=(white|yellow).astype(np.float32),config=view_config)
    if selected_view_id is None:
        if selector is not None:
            chosen=selector(plans)
            selected_view_id=chosen['id'] if isinstance(chosen,dict) else chosen
        else:
            # One shared fallback: prefer empirical quality-feasible candidates,
            # otherwise minimum quality deficit, then compute cost and ID.
            selected_view_id=choose_view(plans['candidates'],model=None)[0]['id']
    selected=next(p for p in plans['candidates'] if p['id']==selected_view_id)
    if not selected['hard_valid']:raise ValueError('selected view is not construction-valid')
    local=render_view(layers,selected)
    global_geometry=render_view(layers,plans['global_geometry'])
    rendered={p['id']:render_view(layers,p) for p in plans['candidates']} if render_all else None
    return dict(layers=layers,plans=plans,selected_plan=selected,local=local,global_geometry=global_geometry,
                all_views=rendered,effect_receipt=effect_receipt,
                provenance=dict(clean_GT_consumed=False,true_shadow_consumed=False,
                                bbox_filled=False,material_owner_matching_required=False,
                                shared_constructor=True,planner_selection_diagnostic_only=True,
                                shadow_mode=shadow_mode))
