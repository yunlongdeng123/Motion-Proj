"""Refine accepted input observations using measured, image-grounded occluders.

The input manifest supplies an existing accepted label bank and its provenance.
This is a reusable refinement stage, not a claim of automatic raw-video identity
discovery. Unaccepted vehicle fragments keep foreground with unknown identity.
All model proposals, decisions and original observations remain reproducible.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .v44_inputs import IdentityRegistry
from .v44_prepare import FrameMaskEvidence, prepare_frame_masks
from .v44_temporal import AcceptedObservation
from .v44_observation_io import load_waymo_evidence
from .v44_sam import SAM2ImageAdapter, infer_occluders


def _image(path, shape=None):
    image=Image.open(path)
    if shape is not None: image=image.resize((shape[1],shape[0]),Image.Resampling.NEAREST)
    return np.array(image, copy=True)


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def refine(manifest, output, frame_keys=None):
    import torch
    path=Path(manifest).resolve();data=json.loads(path.read_text());base=path.parent
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob('v44_*.py')}
    if data.get('schema')!='v44.input_refinement.v1':raise ValueError('Unknown refinement manifest')
    resolve=lambda p:Path(p) if Path(p).is_absolute() else base/p
    bank,labels,candidates,semantics=map(resolve,[data[k] for k in ('bank','accepted_masks','candidate_masks','semantics')])
    if not data.get('accepted_masks_provenance'):raise ValueError('Accepted observations require provenance')
    registry=IdentityRegistry(data['track_to_actor'])
    vehicle_classes=data['vehicle_semantic_classes'];nonvehicle_classes=data['nonvehicle_semantic_classes']
    if set(vehicle_classes)&set(nonvehicle_classes):raise ValueError('Semantic classes overlap')
    frames=data['frames']
    if frame_keys is not None:frames=[f for f in frames if f['key'] in frame_keys]
    if not frames:raise ValueError('No frames')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sam=data['sam'];predictor=SAM2ImageAdapter(resolve(sam['source']),resolve(sam['checkpoint']),sam['config'])
    records=[];started=time.monotonic()
    with torch.inference_mode():
        for frame in frames:
            key=frame['key'];directory=frame['directory']
            observation=resolve(frame['observation']);obs=json.loads(observation.read_text())
            rgb=_image(bank/obs['image_path']);native=rgb.shape[:2]
            known=_image(labels/directory/'labels.png');shape=known.shape
            if known.ndim!=2 or known.dtype.kind not in 'iu':raise ValueError('Expected integer accepted labels')
            semantic_native=_image(semantics/frame['semantic_file'],native)
            semantic=_image(semantics/frame['semantic_file'],shape)
            nonvehicle_native=np.isin(semantic_native,nonvehicle_classes)
            vehicle=np.isin(semantic,vehicle_classes)
            predictor.set_image(rgb)
            target_records=[];static=np.zeros(shape,bool);unknown=np.zeros(shape,bool)
            actor_evidence={};folder=out/directory;folder.mkdir()
            prompts=json.loads(resolve(frame['prompts']).read_text())
            for prompt in prompts:
                track=prompt['track_id'];actor=registry.track_to_actor[track]
                current=known==actor;candidate_path=candidates/directory/(prompt['alias']+'.png')
                candidate=_image(candidate_path,shape)>0 if candidate_path.exists() else np.zeros(shape,bool)
                # Independent vehicle semantics certify foreground, never identity.
                unknown|=candidate & vehicle & (known==0)
                nonvehicle_fraction=float((current & np.isin(semantic,nonvehicle_classes)).sum()/max(1,int(current.sum())))
                record=dict(track_id=track,actor_id=actor,accepted_pixels=int(current.sum()),
                    candidate_path=str(candidate_path),nonvehicle_fraction=nonvehicle_fraction,
                    proposals=[],removed_pixels=0)
                # Same fixed trigger for every actor and every scene; low semantic
                # purity is only a request for actual measured/RGB examination.
                if int(current.sum())>=12 and nonvehicle_fraction>=.35:
                    evidence=load_waymo_evidence(bank,key,prompt,observation_path=observation)
                    result=infer_occluders(predictor,evidence['foreground_uv'],evidence['foreground_depth'],
                        evidence['own_supported_uv'],semantic_nonvehicle=nonvehicle_native)
                    occluder=np.asarray(Image.fromarray(result['accepted_mask']).resize(
                        (shape[1],shape[0]),Image.Resampling.NEAREST),bool)
                    # Certification addresses this observation. It cannot overwrite
                    # another accepted object's visible observation.
                    removed=occluder & (current | ((known==0)&candidate))
                    static|=removed;record['removed_pixels']=int((removed & current).sum())
                    np.savez_compressed(folder/f'occluder_{actor}.npz',accepted_native=result['accepted_mask'])
                    for pi,r in enumerate(result['records']):
                        for ci,c in enumerate(r['candidates']):
                            Image.fromarray(np.uint8(c['mask'])*255).save(folder/f'proposal_{actor}_{pi}_{ci}.png')
                            c['certification'].pop('accepted_occluder',None);c.pop('mask')
                    record.update(proposals=result['records'],unknown_proposals=result['unknown'],
                        measured_foreground_returns=len(evidence['foreground_depth']),
                        own_supported_returns=len(evidence['own_supported_uv']))
                if current.any():
                    actor_evidence[actor]=FrameMaskEvidence(candidate=current,candidate_accepted=True,
                        candidate_provenance=dict(accepted_bank=str(labels),source=data['accepted_masks_provenance']))
                target_records.append(record)
            # Preserve existing unknown foreground independently of instance labels.
            old_unknown=labels/directory/'unknown_vehicle_identity.png'
            if old_unknown.exists():unknown|=_image(old_unknown,shape)>0
            static_observation=None if not static.any() else AcceptedObservation(static,
                provenance=dict(method='measured_points_plus_independent_SAM2_and_semantics',model=predictor.provenance))
            labels_out,trace=prepare_frame_masks(actor_evidence,registry,timestamp_s=obs['timestamp_micros']/1e6,
                shape=shape,frame_id=key,vehicle_foreground=unknown,certified_nonvehicle_occluder=static_observation)
            if not trace['training_ready']:raise RuntimeError('Unresolved semantic regions cannot train')
            owner=labels_out.owner
            Image.fromarray(np.uint16(np.maximum(owner,0))).save(folder/'labels.png')
            Image.fromarray(labels_out.export()['binary_dynamic_mask']).save(folder/'dynamic_union.png')
            Image.fromarray(labels_out.export()['unknown_identity']).save(folder/'unknown_vehicle_identity.png')
            for prompt in prompts:
                actor=registry.track_to_actor[prompt['track_id']]
                Image.fromarray(np.uint8(owner==actor)*255).save(folder/(prompt['alias']+'.png'))
            trace.pop('semantic_unresolved_regions')
            receipt=dict(key=key,targets=target_records,decision=trace,
                accepted_mask_sha256=hashlib.sha256((labels/directory/'labels.png').read_bytes()).hexdigest(),
                known_labels_changed=int((np.maximum(owner,0)!=known).sum()),unknown_pixels=int((owner==-1).sum()))
            (folder/'refinement.json').write_text(json.dumps(receipt,default=_json,indent=2)+'\n')
            records.append({k:receipt[k] for k in ('key','known_labels_changed','unknown_pixels')})
            print(json.dumps(records[-1]),flush=True)
    receipt=dict(status='INPUT_REFINEMENT_COMPLETE_AWAITING_VISUAL_QA',frames=records,
        input_manifest=str(path),model=predictor.provenance,seconds=time.monotonic()-started,
        input_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=source_hashes,
        same_rules_for_all_frames=True,training_updates=0,scene_specific_algorithm_branches=False)
    (out/'receipt.json').write_text(json.dumps(receipt,default=_json,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--frame-key',action='append',type=json.loads)
    a=p.parse_args();refine(a.manifest,a.output,a.frame_key)

if __name__=='__main__':main()
