"""CPU policy: object-shadow probability -> appearance, material, and write roles."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

try:
    from .shadow_probability import ShadowField, rgb_sha256, mask_sha256
except ImportError:
    from shadow_probability import ShadowField, rgb_sha256, mask_sha256


@dataclass(frozen=True)
class ShadowRegionConfig:
    low: float = .1
    high: float = .6
    max_write_uncertainty: float = .35
    uncertain_threshold: float = .35
    boundary_guard_px: int = 2

    def __post_init__(self):
        if not 0 <= self.low < self.high <= 1:
            raise ValueError('require 0 <= low < high <= 1')
        if not 0 <= self.max_write_uncertainty <= 1 or not 0 <= self.uncertain_threshold <= 1:
            raise ValueError('uncertainty policy values must lie in [0,1]')
        if isinstance(self.boundary_guard_px,bool) or int(self.boundary_guard_px)!=self.boundary_guard_px or not 0<=self.boundary_guard_px<=4:
            raise ValueError('boundary guard is a small integer sampling guard, not a shadow search range')


def _mask(masks,name,shape,default=None):
    value=masks.get(name,default)
    if value is None:return np.zeros(shape,bool)
    result=np.asarray(value,bool)
    if result.shape!=shape:raise ValueError(f'{name} must share native grid')
    return result.copy()


def apply_shadow_regions(layers,field,observation_rgb,config=None,*,independent_regions=None):
    """Return new layers without changing the original or using supervision.

    Probability-derived appearance uncertainty does not alter the physical
    unknown mask, valid geometry observations, or existing depth values.
    Low/high defaults are provisional engineering thresholds, not tuned claims.
    """
    if layers.get('receipt',{}).get('independent_regions'):
        raise ValueError('apply shadow routing before independent regions; do not silently replace an independent contract')
    if not isinstance(field,ShadowField):raise TypeError('field must be a typed ShadowField')
    field.require_ready()
    cfg=ShadowRegionConfig(**config) if isinstance(config,dict) else config or ShadowRegionConfig()
    if not isinstance(cfg,ShadowRegionConfig):raise TypeError('config must be ShadowRegionConfig or dict')
    observed=np.asarray(observation_rgb)
    if observed.ndim!=3 or observed.shape[2]!=3 or observed.dtype!=np.uint8:
        raise ValueError('observation_rgb must be actual native uint8 RGB')
    shape=observed.shape[:2]
    if field.probability.shape!=shape or field.observation_sha256!=rgb_sha256(observed):
        raise ValueError('shadow prediction belongs to a different observation or camera grid')
    if 'rgb' not in layers or rgb_sha256(layers['rgb'])!=field.observation_sha256:
        raise ValueError('native layers belong to a different observation than the shadow prediction')
    old=layers['masks'];target=_mask(old,'target_core',shape)
    if not target.any() or field.target_sha256!=mask_sha256(target):
        raise ValueError('shadow prediction belongs to a different target instance')
    k=_mask(old,'K',shape);retained=_mask(old,'retained_core',shape)
    physical=_mask(old,'physical_unknown',shape,old.get('rgb_unknown_core'))
    if (target&(k|retained)).any() or (physical&k).any():
        raise ValueError('target/physical core conflicts with a protected retained object')
    foreground=_mask(old,'foreground_possible',shape)
    foreground_uncertainty=_mask(old,'foreground_uncertainty',shape)
    observed_road=_mask(old,'road_geometry_observed',shape,
                        old.get('geometry_road_visible',old.get('road_observed')))
    legal_road=observed_road&~foreground&~k&~physical
    p=np.asarray(field.probability);u=np.asarray(field.uncertainty);valid=np.asarray(field.valid)
    raw_possible=valid&(p>=cfg.low)
    possible=raw_possible&legal_road
    high=valid&(p>=cfg.high)&legal_road
    probability_uncertain=valid&(((p>=cfg.low)&(p<cfg.high))|(u>cfg.uncertain_threshold))&legal_road
    invalid_road=~valid&legal_road
    radius=int(cfg.boundary_guard_px)
    boundary=(binary_dilation(possible,iterations=radius)&~binary_erosion(possible,iterations=radius))&legal_road if radius else np.zeros(shape,bool)
    uncertain=probability_uncertain|boundary|invalid_road
    effect=possible|uncertain
    appearance=(physical|effect)&~k
    association_u=float(field.diagnostics.get('association_uncertainty',0.))
    if not np.isfinite(association_u) or not 0<=association_u<=1:
        raise ValueError('instance association uncertainty must lie in [0,1]')
    association_can_write=association_u<=cfg.max_write_uncertainty
    auto_write=(physical|(high&(u<=cfg.max_write_uncertainty)&~boundary&association_can_write))&appearance&~k
    protection_conflict=raw_possible&k
    pending=(appearance&~auto_write)|protection_conflict
    cmat=observed_road&valid&~foreground&~foreground_uncertainty&~appearance&~uncertain&~raw_possible
    result=copy.deepcopy(layers)
    result['rgb']=observed.copy()
    masks=result['masks']
    masks.update(target_unknown=appearance,rgb_unknown_core=physical.copy(),physical_unknown=physical.copy(),
                 material_contaminated_effect=effect,effect_inferred=possible,
                 shadow_possible=possible,shadow_high=high,shadow_uncertain=uncertain,
                 shadow_boundary_guard=boundary,shadow_prediction_unavailable=invalid_road,
                 shadow_protection_conflict=protection_conflict,shadow_pending=pending,
                 W_edit=auto_write,C_mat=cmat,visible_support=cmat.copy(),retained_context=~appearance)
    masks['uncertainty']=_mask(old,'uncertainty',shape)|uncertain|protection_conflict
    confidence=result.setdefault('confidence',{})
    confidence['shadow_probability']=p.copy().astype(np.float32)
    confidence['shadow_uncertainty']=u.copy().astype(np.float32)
    confidence['shadow_valid']=valid.astype(np.float32)
    confidence['shadow_association_uncertainty']=np.full(shape,association_u,np.float32)
    # Retain old independent supervision; never expand its loss support from GT.
    if result.get('loss_mask') is not None:
        result['loss_mask']=np.asarray(result['loss_mask'],bool)&appearance&~k
    metadata=field.metadata()
    receipt=dict(metadata,policy=asdict(cfg),policy_provisional=True,calibrated=False,
                 detection_accuracy_validated=False,
                 probability_role='object-associated shadow prediction; not shadow alpha or GT',
                 bbox_search_range_used=False,brightness_shadow_rule_used=False,
                 geometry_and_depth_roles_unchanged=True,
                 boundary_guard_role='camera sampling protection only, not shadow detection',
                 appearance_pixels=int(appearance.sum()),possible_shadow_pixels=int(possible.sum()),
                 high_shadow_pixels=int(high.sum()),uncertain_shadow_pixels=int(uncertain.sum()),
                 automatic_write_pixels=int(auto_write.sum()),pending_pixels=int(pending.sum()),
                 protected_shadow_conflict_pixels=int(protection_conflict.sum()),
                 association_uncertainty=association_u,association_allows_shadow_write=association_can_write,
                 association_uncertainty_is_not_a_global_pixel_floor=True,
                 material_visible_pixels=int(cmat.sum()),original_layers_mutated=False,
                 completion_status='pending_regions' if pending.any() else 'regions_ready_for_model',
                 new_model_inference_executed=False,synthetic_truth_consumed=False)
    result.setdefault('receipt',{})['object_shadow']=receipt
    result['receipt']['observation_sha256']=field.observation_sha256
    result['receipt']['geometry_policy']='physical core unknown; actual shadowed structure remains observable'
    result['receipt']['material_policy']='exclude possible shadow, boundary uncertainty, foreground and unavailable predictions'
    assert not (auto_write&~appearance).any() and not (auto_write&k).any()
    assert not (cmat&(raw_possible|uncertain|appearance|foreground|foreground_uncertainty)).any()
    assert np.array_equal(result['masks']['physical_unknown'],np.asarray(old['physical_unknown'],bool))
    if independent_regions is not None:
        from .independent_regions import apply_independent_regions
        result=apply_independent_regions(result,**independent_regions)
    return result
