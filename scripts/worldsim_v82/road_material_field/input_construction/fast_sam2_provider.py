"""Resident, RAM-first SAM2 pixel-proposal provider for an automatic InputBuilder.

No candidate PNG archival occurs here.  Bounding boxes remain model prompts;
accepted evidence is supplied independently and never gated by current points.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SAM_ROOT = Path('/root/autodl-tmp/third_party/worldsim_v32/sam2')
CHECKPOINT = SAM_ROOT/'checkpoints/sam2.1_hiera_large.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


def pack_roi(mask):
    """Optional sparse cache representation, without a full-frame PNG scan/write."""
    yy = np.flatnonzero(mask.any(axis=1))
    xx = np.flatnonzero(mask.any(axis=0))
    if not len(yy):
        return dict(roi_xyxy=[0, 0, 0, 0], shape_hw=[0, 0], bits=np.empty(0, np.uint8))
    x0, x1, y0, y1 = int(xx[0]), int(xx[-1]+1), int(yy[0]), int(yy[-1]+1)
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
    return dict(roi_xyxy=[x0, y0, x1, y1], shape_hw=list(crop.shape),
                bits=np.packbits(crop.reshape(-1), bitorder='little'))


def unpack_roi(packed, image_hw):
    result = np.zeros(image_hw, bool)
    x0, y0, x1, y1 = packed['roi_xyxy']
    if x1 > x0 and y1 > y0:
        count = (y1-y0)*(x1-x0)
        value = np.unpackbits(packed['bits'], count=count, bitorder='little').reshape(y1-y0, x1-x0)
        result[y0:y1, x0:x1] = value
    return result


class FastSAM2Provider:
    def __init__(self, *, source=SAM_ROOT, checkpoint=CHECKPOINT, config=CONFIG, batch_size=8):
        sys.path.insert(0, str(source))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.batch_size = batch_size
        torch.set_num_threads(2)
        start = time.perf_counter()
        self.predictor = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device='cuda'))
        torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter()-start
        self.provenance = dict(source=str(source), checkpoint=str(checkpoint), config=config,
                               precision='float32 existing-provider default', batch_size=batch_size)
        self._embedding_image_key = None
        self._proposal_cache_key = self._proposal_cache = None

    def infer_path(self, path, prompts, *, accepted_masks=(), retain_sparse=False,
                   proposal_reducer=None, reducer_context=None, require_qualified=True):
        before = time.perf_counter()
        rgb = np.asarray(Image.open(path).convert('RGB')).copy()
        read_seconds = time.perf_counter()-before
        result = self.infer(rgb, prompts, accepted_masks=accepted_masks, retain_sparse=retain_sparse,
                            proposal_reducer=proposal_reducer, reducer_context=reducer_context,
                            require_qualified=require_qualified)
        result['timing']['image_read_seconds'] = read_seconds
        result['timing']['provider_with_read_seconds'] = read_seconds+result['timing']['provider_seconds']
        return result

    def infer(self, rgb, prompts, *, accepted_masks=(), retain_sparse=False,
              proposal_reducer=None, reducer_context=None, require_qualified=True):
        torch = self.torch
        started = time.perf_counter()
        rgb = np.asarray(rgb)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        if require_qualified:
            if proposal_reducer is None:
                raise ValueError('Online InputBuilder requires the shared qualify_ensemble RAM reducer')
            required=['policy_sha256','protected_sha256','core_sha256']
            if reducer_context is None or any(not isinstance(reducer_context.get(k),str) or len(reducer_context[k])!=64 for k in required):
                raise ValueError('Reducer cache key must include policy, protected-mask and core-mask SHA256')
        accepted_union = np.zeros((height, width), bool)
        # This layer is deliberately independent of the prompt list and points.
        for mask in accepted_masks:
            mask = np.asarray(mask, bool)
            if mask.shape != (height, width):
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((width, height), Image.Resampling.NEAREST)) > 0
            accepted_union |= mask
        image_key = hashlib.sha256(rgb.tobytes()).hexdigest()
        serializable = [{key:value for key,value in p.items() if key in
                         ['id', 'box_xyxy_native', 'box', 'positive_uv', 'negative_uv', 'positive', 'negative']}
                        for p in prompts]
        cache_key = hashlib.sha256((image_key+json.dumps(serializable, sort_keys=True,
                            default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))
                            +json.dumps(reducer_context,sort_keys=True)+str(retain_sparse)+str(require_qualified)).encode()).hexdigest()
        if cache_key == self._proposal_cache_key:
            result = dict(self._proposal_cache)
            result['accepted_union'] = accepted_union
            result['timing'] = dict(provider_seconds=time.perf_counter()-started, encoder_seconds=0.,
                                   decoder_seconds=0., aggregate_seconds=0., encoder_calls=0,
                                   decoder_calls=0, proposal_cache_hit=True)
            return result
        encoder_seconds = decoder_seconds = aggregate_seconds = 0.
        encoder_calls = decoder_calls = 0
        with torch.inference_mode(), torch.autocast('cuda',enabled=False):
            if image_key != self._embedding_image_key:
                torch.cuda.synchronize()
                before = time.perf_counter()
                self.predictor.set_image(rgb.copy())
                torch.cuda.synchronize()
                encoder_seconds = time.perf_counter()-before
                encoder_calls = 1
                self._embedding_image_key = image_key
            top_union = np.zeros((height, width), bool)
            any_union = np.zeros_like(top_union)
            disagreement_union = np.zeros_like(top_union)
            qualified_union = np.zeros_like(top_union)
            qualified_disagreement = np.zeros_like(top_union)
            unqualified_possible = np.zeros_like(top_union)
            qualified_confidence = np.zeros((height,width),np.float32)
            records = []
            box_only, point_conditioned = [], []
            for p in prompts:
                box = np.clip(np.asarray(p.get('box_xyxy_native', p.get('box')), np.float32),
                              [0, 0, 0, 0], [width, height, width, height])
                if min(box[2:]-box[:2]) < 1:
                    records.append(dict(id=p['id'], status='invalid_prompt_unknown', certified=False))
                    continue
                positive = p.get('positive_uv', p.get('positive', []))
                negative = p.get('negative_uv', p.get('negative', []))
                row = dict(p=p, box=box, positive=positive, negative=negative)
                (point_conditioned if positive or negative else box_only).append(row)
            batches = [box_only[start:start+self.batch_size] for start in range(0, len(box_only), self.batch_size)]
            batches += [[row] for row in point_conditioned]
            for batch in batches:
                boxes = np.stack([row['box'] for row in batch])
                coords = labels = None
                if len(batch) == 1 and (batch[0]['positive'] or batch[0]['negative']):
                    positive, negative = batch[0]['positive'], batch[0]['negative']
                    coords = np.asarray(positive+negative, np.float32)
                    labels = np.asarray([1]*len(positive)+[0]*len(negative), np.int32)
                    boxes = boxes[0]
                torch.cuda.synchronize()
                before = time.perf_counter()
                masks, scores, _ = self.predictor.predict(box=boxes, point_coords=coords,
                                                         point_labels=labels, multimask_output=True)
                torch.cuda.synchronize()
                decoder_seconds += time.perf_counter()-before
                decoder_calls += 1
                before = time.perf_counter()
                masks, scores = np.asarray(masks, bool), np.asarray(scores, float)
                if masks.ndim == 3:
                    masks, scores = masks[None], scores[None]
                assert masks.shape[0] == len(batch) and masks.shape[2:] == (height, width)
                scores = scores.reshape(len(batch), -1)
                for row, candidates, predicted_iou in zip(batch, masks, scores):
                    best = int(predicted_iou.argmax())
                    any_mask, all_mask = candidates.any(axis=0), candidates.all(axis=0)
                    disagreement = any_mask & ~all_mask
                    top_union |= candidates[best]
                    any_union |= any_mask
                    disagreement_union |= disagreement
                    record = dict(id=row['p']['id'], type=row['p'].get('type'),
                                  sources=row['p'].get('sources'), track_id=row['p'].get('track_id'),
                                  box_xyxy_native=row['box'].tolist(), scores=predicted_iou.tolist(),
                                  selected_top_score_index=best, candidate_count=len(candidates),
                                  positive_points=len(row['positive']), negative_points=len(row['negative']),
                                  point_absence_is_not_negative_evidence=True,
                                  certified=False, role='uncertain SAM pixel proposal')
                    if proposal_reducer is not None:
                        reduction=proposal_reducer(row['p'],candidates,predicted_iou)
                        for key in ['qualified_union','qualified_disagreement','unqualified_possible']:
                            assert np.asarray(reduction[key]).shape==(height,width),key
                        qualified_union |= reduction['qualified_union']
                        qualified_disagreement |= reduction['qualified_disagreement']
                        unqualified_possible |= reduction['unqualified_possible']
                        if 'confidence_map' in reduction:
                            confidence=np.asarray(reduction['confidence_map'],np.float32)
                            assert confidence.shape==(height,width)
                            qualified_confidence=np.maximum(qualified_confidence,confidence)
                        else:
                            for candidate_info in reduction['receipt']['candidates']:
                                if candidate_info['qualified']:
                                    index=candidate_info['index']
                                    confidence=candidates[index].astype(np.float32)*float(np.clip(predicted_iou[index],0,1))
                                    qualified_confidence=np.maximum(qualified_confidence,confidence)
                        record['qualification_receipt']=reduction['receipt']
                    if retain_sparse:
                        record['packed_candidates'] = [pack_roi(candidate) for candidate in candidates]
                    records.append(record)
                aggregate_seconds += time.perf_counter()-before
        for value in [top_union, any_union, disagreement_union,qualified_union,
                      qualified_disagreement,unqualified_possible,qualified_confidence]:
            value.setflags(write=False)
        result = dict(top_candidate_union=top_union, any_candidate_union=any_union,
                      disagreement_union=disagreement_union, accepted_union=accepted_union,
                      qualified_union=qualified_union,qualified_disagreement=qualified_disagreement,
                      unqualified_possible=unqualified_possible,qualified_confidence=qualified_confidence,
                      shared_qualification_applied=proposal_reducer is not None,
                      reducer_context=reducer_context,
                      proposals=records, provenance=self.provenance,
                      accepted_retained_independently=True, full_proposal_PNGs_written=0,
                      returned_proposal_masks_are_read_only=True,
                      timing=dict(provider_seconds=time.perf_counter()-started,
                                  encoder_seconds=encoder_seconds, decoder_seconds=decoder_seconds,
                                  aggregate_seconds=aggregate_seconds, encoder_calls=encoder_calls,
                                  decoder_calls=decoder_calls, proposal_cache_hit=False))
        self._proposal_cache_key = cache_key
        self._proposal_cache = result
        return result
