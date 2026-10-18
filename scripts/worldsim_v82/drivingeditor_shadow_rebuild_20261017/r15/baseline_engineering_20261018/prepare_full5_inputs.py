"""Build a separate, matched-five-ID GN11 input from original measurements.

The five categorical masks are the already reviewed legacy-stage inputs, not
R15 tri-state masks. Nearest expansion serves only the native 1920x1280 loader;
the exact 480x320 supervision must round-trip bitwise unchanged.
"""
from pathlib import Path
import os,sys,json,shutil,hashlib,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018';OUT=ROOT/'baseline_engineering_20261018'
NEW=OUT/'full5_pipeline';SID='scene0008_089_137'
def main():
    import numpy as np
    from PIL import Image
    if NEW.exists():raise FileExistsError('preserve any prior preparation; diagnose before restart')
    NEW.mkdir();src=ROOT/'inputs'/SID;dest=NEW/'inputs'/SID;dest.mkdir(parents=True)
    for n in ['image','calib','pose','velodyne','sky','timestamp','annotations','mask','points3d.ply']:
        os.symlink(src/n,dest/n,target_is_directory=(src/n).is_dir())
    prep=json.loads((src/'preparation_manifest.json').read_text())
    spec=json.loads((TASK/'spec5.json').read_text());bundle=np.load(Path(spec['prepared'])/'instances.npz',allow_pickle=False)
    tracks=bundle['instance_ids'].tolist();assert len(tracks)==5
    native=json.loads((TASK/'native_masks_for_review/manifest.json').read_text())
    assert native['root_engineering_review']['status']=='ALL49_IDENTITY_CHECKED'
    manifest=dict(prep);manifest['selected_real_track_ids']=tracks
    manifest['scope']='Same49 raw observations; five selected identities at both GN11 and rigid-instance stages'
    manifest['parent_source_scene_read_only']=str(src)
    (dest/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'matched_training_labels').mkdir();frames=bundle['frame_ids'].tolist();records=[]
    for fid in frames:
        p=Path(spec['prepared'])/'instance_masks'/(fid+'.png');a=np.asarray(Image.open(p))
        assert a.shape==(320,480) and set(np.unique(a)).issubset(set(range(6)))
        shutil.copy2(p,dest/'matched_training_labels'/p.name)
        records.append(dict(frame_id=fid,label_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (NEW/'reviewed_selections').mkdir()
    sel=dict(status='assembled_from_previously_reviewed_identity_masks',GT_boxes_used_to_create_masks=False,
        source_legacy_prepared=str(Path(spec['prepared'])),native_two_mask_review=str(TASK/'native_masks_for_review/manifest.json'),
        actor_order=[5,26,18,6,37],human_verdict=None,actors=[dict(raw_track_id=t,description=f'physical a{a}',
            frames=[dict(frame_id=fid,training_slot=i) for fid in frames]) for i,(a,t) in enumerate(zip([5,26,18,6,37],tracks))])
    (NEW/'reviewed_selections'/(SID+'.json')).write_text(json.dumps(sel,indent=2))
    shutil.copytree(TASK/'infra',NEW/'infra',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'waymo_numpy_cpu.py',NEW/'waymo_numpy_cpu.py')
    builder=(ROOT/'build_selected_priors.py').read_text()
    assert builder.count('from data import _actual_body_from_selection')==1
    builder=builder.replace('from data import _actual_body_from_selection','''def _actual_body_from_selection(mask_scene, frame_id, labels, row):
    small=np.asarray(Image.open(mask_scene/'matched_training_labels'/(frame_id+'.png')))
    raw=(small==row['training_slot']+1).astype(np.uint8)*255
    return np.asarray(Image.fromarray(raw).resize((labels.shape[1],labels.shape[0]),Image.Resampling.NEAREST))>0''')
    builder=builder.replace("sel['status']=='visually_reviewed'","sel['status']=='assembled_from_previously_reviewed_identity_masks'")
    (NEW/'build_selected_priors.py').write_text(builder)
    shutil.copy2(ROOT/'prepare_gn11_window.py',NEW/'prepare_gn11_window.py')
    subprocess.run([sys.executable,str(NEW/'build_selected_priors.py'),SID,'--reject-unreliable-frame-priors'],check=True)
    for row in records:
        fid=row['frame_id'];small=np.asarray(Image.open(dest/'matched_training_labels'/(fid+'.png')))
        union=np.asarray(Image.open(dest/'dynamic_masks/dynamic_mask_0'/(fid+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
        assert np.array_equal(union,small>0),'GN11/instance target union differs: '+fid
        for k in range(5):
            a=np.asarray(Image.open(dest/f'dynamic_actor_assets/actor_{k}/body_masks'/(fid+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
            assert np.array_equal(a,small==k+1),'per-ID round-trip differs'
    report=json.loads((dest/'dynamic_multi_actor_preparation.json').read_text())
    receipt=dict(status='MATCHED_FIVE_ID_RAW_INPUTS_PREPARED',frames=49,track_ids=tracks,
        original_3_ID_input_read_only=True,R15_fused_tristate_or_model_parameters_used=False,
        binary_mask_source='same reviewed legacy five-ID labels; added two IDs originate from cached native SAM3 proposals, not R15 fused tri-state masks',
        body_masks_exactly_match_later_480x320_labels=True,
        native_target_expansion='nearest 480x320 to1920x1280 only; no extra measurement detail',
        dynamic_prior_points=[x['total_real_points'] for x in report['actors']],
        geometry_rule=report['geometry'],GT_boxes_used_to_filter_individual_points=False,
        original_builder_sha256=hashlib.sha256((ROOT/'build_selected_priors.py').read_bytes()).hexdigest(),
        adapted_builder_sha256=hashlib.sha256(builder.encode()).hexdigest(),records=records,
        new_GN11_training_updates=0,new_instance_training_updates=0,human_verdict=None)
    (OUT/'matched5_input_receipt.json').write_text(json.dumps(receipt,indent=2))
    subprocess.run([sys.executable,str(NEW/'prepare_gn11_window.py'),SID],check=True)
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
