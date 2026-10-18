"""CPU pixel-level check of GN11 dynamic targets versus five-ID stage targets."""
from pathlib import Path
import os,json,hashlib,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018';OUT=ROOT/'baseline_engineering_20261018'
def main():
    spec=json.loads((TASK/'spec5.json').read_text());scene=Path(spec['source_scene'])
    prep=json.loads((scene/'dynamic_multi_actor_preparation.json').read_text())
    tracks=[x['raw_track_id'] for x in prep['actors']]
    data=np.load(Path(spec['prepared'])/'instances.npz',allow_pickle=False)
    ids=data['instance_ids'].tolist();rows=[]
    for fid in data['frame_ids'].tolist():
        labels=np.asarray(Image.open(Path(spec['prepared'])/'instance_masks'/(fid+'.png')).resize((480,320),Image.Resampling.NEAREST))
        original=scene/'dynamic_image/dynamic_image_0'/(fid+'.png')
        rgb=np.asarray(Image.open(original).convert('RGB').resize((480,320),Image.Resampling.BILINEAR))
        # Match the original renderer-era dynamic-image mask convention on the
        # first RGB channel. Also record any-channel occupancy for transparency.
        first=rgb[...,0]>0;any_channel=(rgb>0).any(-1)
        canonical_path=scene/'dynamic_masks/dynamic_mask_0'/(fid+'.png')
        canonical=np.asarray(Image.open(canonical_path).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
        shadow_path=scene/'dynamic_masks/shadow_mask_0'/(fid+'.png')
        if shadow_path.exists():
            canonical|=np.asarray(Image.open(shadow_path).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
        new=[]
        for slot,actor in [(3,6),(4,37)]:
            target=labels==slot+1
            new.append(dict(actor=actor,track=ids[slot],body_pixels=int(target.sum()),
                GN11_dynamic_first_channel_pixels=int((target&first).sum()),
                GN11_dynamic_any_channel_pixels=int((target&any_channel).sum()),
                GN11_canonical_dynamic_mask_pixels=int((target&canonical).sum())))
        rows.append(dict(frame_id=fid,added_actors=new,dynamic_RGB_sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    added={a:dict(body_pixels=sum(x['body_pixels'] for r in rows for x in r['added_actors'] if x['actor']==a),
        GN11_dynamic_first_channel_pixels=sum(x['GN11_dynamic_first_channel_pixels'] for r in rows for x in r['added_actors'] if x['actor']==a),
        GN11_dynamic_any_channel_pixels=sum(x['GN11_dynamic_any_channel_pixels'] for r in rows for x in r['added_actors'] if x['actor']==a),
        GN11_canonical_dynamic_mask_pixels=sum(x['GN11_canonical_dynamic_mask_pixels'] for r in rows for x in r['added_actors'] if x['actor']==a)) for a in [6,37]}
    r=dict(status='CONFIRMED_STAGE_IDENTITY_SET_MISMATCH' if tracks==ids[:3] else 'NEEDS_REVIEW',
        GN11_dynamic_track_ids=tracks,instance_stage_track_ids=ids,added_tracks_absent_from_GN11=[x not in tracks for x in ids[3:]],
        frame_count=len(rows),totals=added,frames=rows,parameter_updates=0,original_files_modified=False,
        scope='GN11 was trained with the three-ID dynamic target set; added a6/a37 enter only the later instance stage. This input-contract change confounds a clean five-ID algorithm comparison; it is not a proof of every artifact cause.')
    r['actual_GN11_target_path']='utils.paper_masks.load_dynamic_mask: nearest binary dynamic_masks plus optional shadow mask'
    p=OUT/'stage_input_audit.json';backup=OUT/'stage_input_audit_rgb_only.json'
    if p.exists() and not backup.exists():shutil.copy2(p,backup)
    p.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='frames'}))
if __name__=='__main__':main()
