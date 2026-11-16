"""Typed object-conditioned shadow fields and CPU SSIS association adapter.

This module performs no model inference. Soft maps are never represented as
Boolean masks. Association and ambiguity scores are provisional, uncalibrated
engineering evidence, not detection accuracy or a guarantee of no leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np


def _rgb(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError('observation RGB must be native uint8 HxWx3')
    return array


def rgb_sha256(value):
    """Array identity, independent of PNG compression, with explicit grid shape."""
    array = np.ascontiguousarray(_rgb(value))
    return hashlib.sha256(b'rgb.uint8.v1\0' + np.asarray(array.shape, '<i8').tobytes()
                          + array.tobytes()).hexdigest()


def mask_sha256(value):
    array = np.asarray(value, bool)
    if array.ndim != 2:
        raise ValueError('target mask must be HxW')
    return hashlib.sha256(b'mask.bool.v1\0' + np.asarray(array.shape, '<i8').tobytes()
                          + np.packbits(array, bitorder='little').tobytes()).hexdigest()


def _unit_map(value, name, shape=None):
    array = np.asarray(value, np.float32)
    if array.ndim != 2 or (shape is not None and array.shape != shape):
        raise ValueError(f'{name} must share the native HxW camera grid')
    if not np.isfinite(array).all() or array.min(initial=0) < 0 or array.max(initial=0) > 1:
        raise ValueError(f'{name} must contain finite values in [0,1]')
    return array.copy()


def _source_is_fixture(source_kind):
    return source_kind.lower() in ('fixture_contract_test', 'contract_test_fixture')


def _validate_source(source_kind):
    text = str(source_kind).strip().lower()
    if not text:
        raise ValueError('source_kind is required')
    if _source_is_fixture(text):
        return
    compact = re.sub(r'[^a-z0-9]+', '', text)
    if ('oracle' in compact or 'groundtruth' in compact or 'synthetictruth' in compact
            or 'supervision' in compact or re.search(r'(^|[^a-z])gt([^a-z]|$)', text)):
        raise ValueError('GT/oracle/synthetic truth cannot be used as a shadow prediction source')


@dataclass(frozen=True)
class ShadowField:
    probability: np.ndarray
    uncertainty: np.ndarray
    valid: np.ndarray
    observation_sha256: str
    target_sha256: str
    model_id: str
    checkpoint_sha256: str
    source_kind: str
    status: str
    diagnostics: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        probability = _unit_map(self.probability, 'shadow probability')
        uncertainty = _unit_map(self.uncertainty, 'shadow uncertainty', probability.shape)
        valid = np.asarray(self.valid, bool)
        if valid.shape != probability.shape:
            raise ValueError('prediction validity must share the native HxW camera grid')
        _validate_source(self.source_kind)
        for name in ('observation_sha256', 'target_sha256'):
            if not re.fullmatch(r'[0-9a-fA-F]{64}', str(getattr(self, name))):
                raise ValueError(f'{name} must be an array-identity SHA256')
        if not self.model_id or not self.status:
            raise ValueError('model_id and explicit prediction status are required')
        if self.status == 'ready' and not valid.any():
            raise ValueError('a ready prediction must have actual valid coverage')
        if self.status == 'ready' and not _source_is_fixture(self.source_kind):
            if not re.fullmatch(r'[0-9a-fA-F]{64}', str(self.checkpoint_sha256)):
                raise ValueError('ready model predictions must pin their checkpoint SHA256')
        for name, array in [('probability', probability), ('uncertainty', uncertainty),
                            ('valid', valid.copy())]:
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    def require_ready(self):
        if self.status != 'ready':
            raise RuntimeError(f'shadow prediction is not ready: {self.status}')
        return self

    def metadata(self):
        return dict(schema='object_conditioned_shadow_field.v1', status=self.status,
                    source_kind=self.source_kind, model_id=self.model_id,
                    checkpoint_sha256=self.checkpoint_sha256,
                    observation_sha256=self.observation_sha256, target_sha256=self.target_sha256,
                    native_hw=list(self.probability.shape), valid_pixels=int(self.valid.sum()),
                    calibrated=False, detection_accuracy_validated=False,
                    fixture_only=_source_is_fixture(self.source_kind), diagnostics=dict(self.diagnostics))


def from_ssis_associations(observation_rgb, target_mask, object_soft_masks, shadow_soft_masks, *,
                           model_id, checkpoint_sha256, associations=None,
                           object_scores=None, association_scores=None,
                           pixel_uncertainties=None, valid=None,
                           min_iou=.5, min_margin=.1, min_association_score=.1,
                           source_kind='ssis_model_output'):
    """Match SSIS object-shadow pairs to a target without hardening soft maps.

    Upstream must decode actual detector association indices and resize soft maps
    into the native camera grid. ``associations`` is an Nx2 object/shadow index
    array; omitted means maps are already paired row-by-row. This adapter does
    not invent associations from shadow proximity or a vehicle bounding box.

    On no match, ambiguity, or unavailable associated shadow, status is nonready.
    Its placeholder zeros are never evidence of an observed shadow-free road.
    """
    observation = _rgb(observation_rgb)
    target = np.asarray(target_mask, bool)
    shape = observation.shape[:2]
    if target.shape != shape or not target.any():
        raise ValueError('target must be a nonempty native HxW instance mask')
    if not 0 <= min_iou <= 1 or not 0 <= min_margin <= 1:
        raise ValueError('association IoU and margin thresholds must lie in [0,1]')
    objects = np.asarray(object_soft_masks, np.float32)
    shadows = np.asarray(shadow_soft_masks, np.float32)
    if objects.size == 0:objects = np.empty((0, *shape), np.float32)
    if shadows.size == 0:shadows = np.empty((0, *shape), np.float32)
    for name, array in [('object soft masks', objects), ('shadow soft masks', shadows)]:
        if array.ndim != 3 or array.shape[1:] != shape:
            raise ValueError(f'{name} must be NxHxW on the native camera grid')
        if not np.isfinite(array).all() or array.min(initial=0) < 0 or array.max(initial=0) > 1:
            raise ValueError(f'{name} must contain finite soft values in [0,1]')
    if associations is None:
        if len(objects) != len(shadows):
            raise ValueError('unpaired SSIS outputs require explicit object/shadow association indices')
        pairs = np.stack([np.arange(len(objects)), np.arange(len(objects))], axis=1)
    else:
        pairs = np.asarray(associations, np.int64)
        if pairs.size == 0:pairs = np.empty((0, 2), np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError('associations must contain object/shadow index pairs')
        if pairs.size and ((pairs < 0).any() or (pairs[:,0] >= len(objects)).any()
                           or (pairs[:,1] >= len(shadows)).any()):
            raise ValueError('SSIS association index lies outside available soft maps')
    def scores(value, length, name):
        array = np.ones(length, np.float32) if value is None else np.asarray(value, np.float32)
        if array.shape != (length,) or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError(f'{name} must contain one finite [0,1] score per entry')
        return array
    oscores=scores(object_scores,len(objects),'object scores')
    ascores=scores(association_scores,len(pairs),'association scores')
    metadata=dict(provisional_association_thresholds=True, min_soft_iou=float(min_iou),
                  min_soft_iou_margin=float(min_margin), pair_count=len(pairs),
                  calibration_validated=False, detector_inference_executed_by_adapter=False)
    identity=dict(observation_sha256=rgb_sha256(observation), target_sha256=mask_sha256(target),
                  model_id=model_id, checkpoint_sha256=checkpoint_sha256, source_kind=source_kind)
    def unavailable(status, extra=None):
        return ShadowField(np.zeros(shape,np.float32),np.ones(shape,np.float32),np.zeros(shape,bool),
                           **identity,status=status,diagnostics=dict(metadata,**(extra or {})))
    if not len(pairs):return unavailable('no_associated_pairs')
    overlap=np.minimum(objects,target.astype(np.float32)[None]).sum(axis=(1,2),dtype=np.float64)
    union=objects.sum(axis=(1,2),dtype=np.float64)+target.sum()-overlap
    ious=overlap/np.maximum(union,1e-12)
    pair_iou=ious[pairs[:,0]]
    order=sorted(range(len(pairs)),key=lambda i:(-float(pair_iou[i]),int(pairs[i,0]),int(pairs[i,1])))
    selected=order[0];best=float(pair_iou[selected])
    runner_up=float(pair_iou[order[1]]) if len(order)>1 else 0.
    match=dict(selected_pair_index=int(selected),object_index=int(pairs[selected,0]),
               shadow_index=int(pairs[selected,1]),best_soft_iou=best,
               runner_up_soft_iou=runner_up,soft_iou_margin=best-runner_up)
    if best < min_iou:return unavailable('no_target_match',match)
    if len(order)>1 and best-runner_up < min_margin:return unavailable('ambiguous_target_association',match)
    if float(ascores[selected]) < min_association_score:return unavailable('low_association_score',match)
    object_index,shadow_index=map(int,pairs[selected])
    probability=shadows[shadow_index].copy()
    if not np.any(probability>0):return unavailable('associated_shadow_unavailable',match)
    # A provisional ambiguity score, explicitly not calibrated model confidence.
    association_uncertainty=max(1-best,1-float(oscores[object_index]),1-float(ascores[selected]))
    if pixel_uncertainties is None:
        local_uncertainty=4*probability*(1-probability)
        uncertainty_source='local Bernoulli ambiguity 4p(1-p); association uncertainty is a separate instance-level gate; uncalibrated'
    else:
        values=np.asarray(pixel_uncertainties,np.float32)
        if values.shape!=shadows.shape:raise ValueError('pixel uncertainty must match shadow soft-map stack')
        local_uncertainty=_unit_map(values[shadow_index],'pixel uncertainty',shape)
        uncertainty_source='provided local model uncertainty; association uncertainty is a separate instance-level gate; calibration unverified'
    # Association confidence concerns this instance's ownership. Broadcasting
    # it over the whole image would erase unrelated visible road when low.
    uncertainty=local_uncertainty.astype(np.float32)
    coverage=np.ones(shape,bool) if valid is None else np.asarray(valid,bool)
    if coverage.shape!=shape:raise ValueError('prediction valid coverage must share native grid')
    metadata.update(match,association_uncertainty=float(association_uncertainty),
                    uncertainty_source=uncertainty_source, probability_preserved_without_score_scaling=True)
    return ShadowField(probability,uncertainty,coverage,**identity,status='ready',diagnostics=metadata)


shadow_field_from_ssis = from_ssis_associations
hash_observation_rgb = rgb_sha256
hash_target_mask = mask_sha256


def save_shadow_field(field, path):
    """Save unthresholded arrays and provenance without pickle or GT fields."""
    if not isinstance(field,ShadowField):raise TypeError('expected ShadowField')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,probability=field.probability,uncertainty=field.uncertainty,
                        valid=field.valid,metadata_json=np.asarray(json.dumps(field.metadata(),sort_keys=True)))
    return str(path)


def load_shadow_field(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata_json'].item()))
        return ShadowField(data['probability'],data['uncertainty'],data['valid'],
            observation_sha256=meta['observation_sha256'],target_sha256=meta['target_sha256'],
            model_id=meta['model_id'],checkpoint_sha256=meta['checkpoint_sha256'],
            source_kind=meta['source_kind'],status=meta['status'],diagnostics=meta.get('diagnostics',{}))


def from_ssis_instance_packet(observation_rgb,target_mask,soft_masks,classes,association_ids,scores,*,
                              model_id,checkpoint_sha256,min_member_margin=.1,
                              object_class_id=None,shadow_class_id=None,**matching_policy):
    """Decode official SSIS paired Instances without guessing class 0/1 names.

    This is an instance-association BASELINE, not MetaShadow inference and not a
    directly mask-conditioned neural forward. Matching an already paired member
    to the supplied target is the explicit conditioning step. Both pair members
    must have different detector class IDs, and ambiguous member matches abstain.
    """
    observed=_rgb(observation_rgb);target=np.asarray(target_mask,bool)
    shape=target.shape;maps=np.asarray(soft_masks,np.float32)
    cls=np.asarray(classes);ids=np.asarray(association_ids);sc=np.asarray(scores,np.float32)
    if maps.ndim!=3 or maps.shape[1:]!=shape or observed.shape[:2]!=shape:
        raise ValueError('SSIS soft maps must be mapped to the full observation camera grid')
    if any(x.shape!=(len(maps),) for x in [cls,ids,sc]):raise ValueError('SSIS instance fields have different lengths')
    if (not np.isfinite(maps).all() or maps.min(initial=0)<0 or maps.max(initial=0)>1
            or not np.isfinite(sc).all() or sc.min(initial=0)<0 or sc.max(initial=0)>1):
        raise ValueError('SSIS must export finite soft membership maps and scores in [0,1]')
    if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(cls.dtype,np.integer):
        raise ValueError('SSIS class and association identities must remain integer labels')
    overlap=np.minimum(maps,target.astype(np.float32)[None]).sum((1,2),dtype=np.float64)
    union=maps.sum((1,2),dtype=np.float64)+target.sum()-overlap
    iou=overlap/np.maximum(union,1e-12)
    if (object_class_id is None)!=(shadow_class_id is None):
        raise ValueError('supply both verified object/shadow class IDs or neither')
    objects=[];shadows=[];obj_scores=[];pair_scores=[];records=[];rejected=[]
    for pair_id in sorted(set(ids.tolist())):
        if pair_id<=0:continue
        members=np.flatnonzero(ids==pair_id)
        if len(members)!=2 or cls[members[0]]==cls[members[1]]:
            rejected.append(dict(association_id=int(pair_id),members=members.tolist(),reason='not_two_different_classes'))
            continue  # No fabricated proximity association.
        if object_class_id is not None:
            oi=members[cls[members]==object_class_id];si=members[cls[members]==shadow_class_id]
            if len(oi)!=1 or len(si)!=1:
                rejected.append(dict(association_id=int(pair_id),reason='verified_semantic_class_mismatch'));continue
            a,b=int(oi[0]),int(si[0])
        else:
            a,b=sorted(members,key=lambda i:(-float(iou[i]),int(i)))
            if float(iou[a]-iou[b])<min_member_margin:
                rejected.append(dict(association_id=int(pair_id),reason='ambiguous_pair_member_role'));continue
        objects.append(maps[a]);shadows.append(maps[b]);obj_scores.append(sc[a])
        pair_scores.append(min(float(sc[a]),float(sc[b])))
        records.append(dict(association_id=int(pair_id),matched_target_member=int(a),shadow_member=int(b),
                            target_class_id=int(cls[a]),shadow_class_id=int(cls[b]),
                            semantic_class_names_not_assumed=object_class_id is None))
    empty=np.empty((0,*shape),np.float32)
    field=from_ssis_associations(observed,target,np.stack(objects) if objects else empty,
        np.stack(shadows) if shadows else empty,model_id=model_id,checkpoint_sha256=checkpoint_sha256,
        object_scores=np.asarray(obj_scores,np.float32),association_scores=np.asarray(pair_scores,np.float32),
        source_kind='ssis_instance_association_baseline',**matching_policy)
    extra=dict(field.diagnostics,official_instance_pairs=records,
               rejected_official_pairs=rejected,verified_object_class_id=object_class_id,
               verified_shadow_class_id=shadow_class_id,
               directly_mask_conditioned_network=False,object_conditioning='explicit target-mask to paired-instance matching',
               native_from_prediction=np.eye(3).tolist(),full_native_probability_grid=True)
    return ShadowField(field.probability,field.uncertainty,field.valid,field.observation_sha256,
        field.target_sha256,field.model_id,field.checkpoint_sha256,field.source_kind,field.status,extra)
