"""Optional SAM2 image adapter: measured prompts in, certified observations out.

Model files and device are explicit. A prompt does not establish identity. All
returned candidates and certification reasons are retained for visual review.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
from .v44_occluders import propose_occluders, certify_occluder


class SAM2ImageAdapter:
    def __init__(self, source, checkpoint, config, *, device='cuda'):
        source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
        if not source.is_dir() or not checkpoint.is_file():
            raise ValueError('Explicit SAM2 source/checkpoint is missing')
        sys.path.insert(0, str(source))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device=device))
        self.provenance = dict(source=str(source), checkpoint=str(checkpoint), config=config,
                               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), device=device)
        self.image_shape = None

    def set_image(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('SAM2 source must be uint8 H,W,3 RGB')
        self.predictor.set_image(rgb)
        self.image_shape = rgb.shape[:2]

    def predict(self, positive_uv, negative_uv=()):
        if self.image_shape is None:
            raise RuntimeError('set_image must precede predict')
        positive = np.asarray(positive_uv, float).reshape(-1, 2)
        negative = np.asarray(negative_uv, float).reshape(-1, 2)
        if not len(positive) or not np.isfinite(np.r_[positive, negative]).all():
            raise ValueError('Measured finite positive prompts are required')
        coordinates = np.r_[positive, negative]
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))].astype(np.int32)
        masks, scores, _ = self.predictor.predict(point_coords=coordinates, point_labels=labels,
                                                  multimask_output=True)
        return [(np.asarray(mask, bool), float(score)) for mask, score in zip(masks, scores)]


def infer_occluders(predictor, foreground_uv, foreground_depth, own_supported_uv, *,
                    semantic_nonvehicle, config=None):
    """Certification is independent of queried deletion and never edits 3-D rows."""
    if predictor.image_shape is None:
        raise RuntimeError('Predictor has no source image')
    proposals = propose_occluders(foreground_uv, foreground_depth, own_supported_uv,
        predictor.image_shape, semantic_nonvehicle=semantic_nonvehicle, config=config)
    records, accepted = [], np.zeros(predictor.image_shape, bool)
    for proposal in proposals['proposals']:
        candidates, best = [], None
        def predict(request):
            nonlocal best
            for mask, score in predictor.predict(request['positive_uv'], request['negative_uv']):
                result = certify_occluder(request, mask, score,
                                      semantic_nonvehicle=semantic_nonvehicle, config=config)
                candidates.append(dict(mask=mask.copy(), score=score, certification=result,
                    prompt_kind=request.get('prompt_kind', 'measured_cluster'),
                    positive_indices=request['positive_indices']))
                if result['status'] == 'accepted_occluder' and (best is None or score > best['score']):
                    best = candidates[-1]
        predict(proposal)
        # A depth-continuous cluster may span several physical image objects.
        # Split only the prompt hypotheses, not the measured geometry. A single
        # prompt still needs >=3 distinct cluster pixels under the RGB mask.
        if best is None and len(proposal['positive_indices'])>1 and proposal.get('measured_uv'):
            for index in range(len(proposal['positive_indices'])):
                request=dict(proposal, prompt_kind='single_measured_point_hypothesis',
                    positive_indices=[proposal['positive_indices'][index]],
                    positive_uv=[proposal['positive_uv'][index]],
                    positive_depth=[proposal['positive_depth'][index]])
                predict(request)
        if best is not None:
            mask = best['mask']
            # Small enclosed segmentation holes (e.g. printed texture) are a
            # separate, recorded topology hypothesis. Never bridge open gaps,
            # and never fill any actual own-source sample. Predictor score is
            # retained as the raw candidate score, not an independently scored
            # claim about inferred hole pixels.
            from scipy.ndimage import binary_fill_holes
            filled = binary_fill_holes(mask); holes = filled & ~mask
            own = np.asarray(own_supported_uv, float).reshape(-1,2)
            xy = np.floor(own).astype(int)
            valid = (xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
            own_hit = bool(holes[xy[valid,1],xy[valid,0]].any())
            inferred = bool(holes.any()) and int(holes.sum()) <= .05*int(mask.sum()) and not own_hit
            best['topology'] = dict(enclosed_hole_pixels=int(holes.sum()),
                maximum_fraction=.05, actual_own_source_conflict=own_hit,
                inferred_pixels_accepted=inferred, inference='small enclosed RGB-mask holes, not measured surface truth')
            accepted |= filled if inferred else mask
        records.append(dict(proposal=proposal, candidates=candidates,
                            chosen_score=None if best is None else best['score']))
    return dict(accepted_mask=accepted, records=records, unknown=proposals['unknown'],
                model_provenance=getattr(predictor, 'provenance', None))
