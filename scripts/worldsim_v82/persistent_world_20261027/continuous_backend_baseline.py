"""Fixed ProPainter across persistent native revisions; ordinary cache control.

No road/repair training or backend change. This measures whether identical
exposed background is regenerated differently when another object is edited.
"""
import argparse,gc,json,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from gpu_slot import exclusive
from resource_control import require_start

def read(p):return json.loads(p.read_text())
def pixels(p):return np.array(Image.open(p).convert('RGB'))
def same_inputs(a,b):
    for channel in ['native_rgb','mask']:
        files=sorted((a/channel).glob('*.png'));other=sorted((b/channel).glob('*.png'))
        if [p.name for p in files]!=[p.name for p in other]:return False
        if not all(np.array_equal(np.array(Image.open(x)),np.array(Image.open(y))) for x,y in zip(files,other)):return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();require_start()
    root=a.suite/'continuous';dest=root/'fixed_backend_audit';dest.mkdir(exist_ok=False)
    summary=read(root/'summary.json');names=['delete_original','insert_reused','move_inserted','turn_inserted','revisit_original_camera'];frames=sorted((root/names[0]/'native_rgb').glob('*.png'));assert len(frames)==31
    state=dict(pid=__import__('os').getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='prepare_disocclusion_masks',completed=[],camera_revisit_backend='pending_unknown_domain_native_only',new_training_updates=0)
    def update(**kw):state.update(kw);(dest/'status.json').write_text(json.dumps(state,indent=2))
    for name in names:
        folder=root/name;(folder/'mask').mkdir(exist_ok=False)
        for path in frames:
            original=np.array(Image.open(a.suite/'delete_target/mask'/path.name))>0
            weights=np.load(folder/'float'/(path.stem+'.npz'))['dynamic_weight'][0]
            mask=original&~(weights>.05);Image.fromarray(mask.astype('uint8')*255).save(folder/'mask'/path.name)
    update(phase='waiting_GPU_fixed_backend_batch')
    # Reuse only pixel-identical inputs to the exact same fixed backend. The
    # cache control is declared explicitly and is not a new inference result.
    assert same_inputs(root/'delete_original',a.suite/'delete_target')
    shutil.copytree(a.suite/'delete_target/propainter',root/'delete_original/propainter')
    (root/'delete_original/ordinary_cache_receipt.json').write_text(json.dumps(dict(source=str(a.suite/'delete_target'),checked_identical_rgb_and_mask_frames=31,fixed_backend_unchanged=True,new_inference=False),indent=2))
    state['completed'].append(dict(revision='delete_original',source='ordinary exact-input cache',new_inference=False));update()
    with exclusive():
        require_start()
        import inpaint_baseline,torch
        for name in ['insert_reused','move_inserted','turn_inserted']:
            require_start();update(phase='fixed_backend',current_revision=name)
            sys.argv=['inpaint_baseline.py','--root',str(root),'--case',name,'--input-kind','native_3d']
            inpaint_baseline.main();gc.collect();torch.cuda.empty_cache()
            state['completed'].append(dict(revision=name,source='same pretrained ProPainter, same seed0 and settings',new_inference=True));update()
    assert same_inputs(root/'revisit_original_camera',root/'turn_inserted')
    shutil.copytree(root/'turn_inserted/propainter',root/'revisit_original_camera/propainter')
    (root/'revisit_original_camera/ordinary_cache_receipt.json').write_text(json.dumps(dict(source=str(root/'turn_inserted'),checked_identical_rgb_and_mask_frames=31,fixed_backend_unchanged=True,new_inference=False),indent=2))
    state['completed'].append(dict(revision='revisit_original_camera',source='ordinary exact-state cache',new_inference=False));update(phase='CPU_cross_revision_consistency')
    rows=[]
    for path in frames:
        common=np.ones((320,480),bool)
        for name in names:
            common&=np.array(Image.open(root/name/'mask'/path.name))>0
            weights=np.load(root/name/'float'/(path.stem+'.npz'))['dynamic_weight'][0]
            common&=~binary_dilation(weights>.001,iterations=3)
        reference_native=pixels(root/'delete_original/native_rgb'/path.name)/255
        reference_video=pixels(root/'delete_original/propainter/model_png'/path.name)/255
        for name in names[1:]:
            native=pixels(root/name/'native_rgb'/path.name)/255;video=pixels(root/name/'propainter/model_png'/path.name)/255
            rows.append(dict(raw=int(path.stem),revision=name,common_exposed_background_pixels=int(common.sum()),native_change_MAE=float(np.abs(native-reference_native)[common].mean()) if common.any() else None,generated_change_MAE=float(np.abs(video-reference_video)[common].mean()) if common.any() else None))
    result=dict(status='fixed_backend_continuous_request_baseline_complete',frames=31,revisions=5,fresh_backend_calls=3,ordinary_cache_reuses=2,reference_revision='delete_original',rows=rows,native_background_change_MAE_macro=float(np.mean([x['native_change_MAE'] for x in rows if x['native_change_MAE'] is not None])),generated_background_change_MAE_macro=float(np.mean([x['generated_change_MAE'] for x in rows if x['generated_change_MAE'] is not None])),counterfactual_RGB_GT=False,not_a_geometry_accuracy_or_success_score=True,learned_memory_gain_established=False,camera_revisit_generated=False,quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(result,indent=2));update(phase='complete');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
