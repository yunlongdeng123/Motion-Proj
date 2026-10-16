"""Correct visibility metadata after RGB routing; certify saved images unchanged."""
import argparse,json,copy
import numpy as np
import torch
from PIL import Image
from train_surface import SurfaceExperiment,NEW
from bc_model import sample_render
from run_experiments import write
import renderer
def run():
    a=argparse.Namespace(output=str(NEW/'visibility_api_audit'),steps=0,warmup=1,restore=str(NEW/'contribution_frozen_v1/final.pt'),probe=True)
    e=SurfaceExperiment(a)
    try:
        evaluation=NEW/'final49_contribution_v1';rows=json.loads((evaluation/'per_frame.json').read_text());summary=json.loads((evaluation/'summary.json').read_text());changed=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                if i%7==0:e.priority()
                pkg=e.render(i,shaded=True);off=e.render(i)
                rendered=np.rint(np.moveaxis(pkg['render'].clamp(0,1).cpu().numpy(),0,-1)*255).astype('uint8')
                old=np.asarray(Image.open(evaluation/'observed'/(row['frame']+'.png')))
                assert np.array_equal(rendered,old),'RGB changed after metadata routing correction'
                target=e.body[i];known=e.mask_known[i,:,None,None]
                before=row['final']['observed_body_visible_ID_L1']
                corrected=float(((pkg['instance_visible_contribution']-target).abs()*known).sum()/(target*known).sum().clamp_min(1))
                row['final']['observed_body_visible_ID_L1']=corrected
                d=e.surface_data;ids=torch.nonzero((d['confidence']>.9)&d['free'][i]&d['shadow'][i]).ravel()
                row['final']['Actor_visible_alpha_on_shadow_pseudo']=float(sample_render(pkg['dynamic_visible_alpha'],d['uv'][i,ids]).mean()) if len(ids) else None
                opaque=pkg['dynamic_visible_alpha'][0]>.999
                row['shadow_on_opaque_actor_max']=float(abs(pkg['render']-off['render'])[:,opaque].max()) if opaque.any() else 0.
                changed.append({'frame':row['frame'],'body_visible_ID_L1_before':before,'body_visible_ID_L1_corrected':corrected,'RGB_PNG_equal':True})
                e.update(state='running',phase='49_frame_visibility_metadata_audit',frame=i+1)
        summary['metrics_mean_frames']['final']['observed_body_visible_ID_L1']=float(np.mean([r['final']['observed_body_visible_ID_L1'] for r in rows]))
        summary['visibility_metadata_corrected_to_RGB_routing']=True
        write(e.out/'corrected_per_frame.json',rows);write(e.out/'corrected_summary.json',summary)
        write(e.out/'audit.json',{'all49_RGB_PNGs_identical':True,'corrected_fields':['Actor visibility metadata','opaque Actor shadow test mask'],
            'optimization_updates':0,'source_checkpoint_unchanged':True,'per_frame':changed})
        e.verify_sources();e.update(state='complete',phase='visibility_metadata_corrected_RGB_unchanged',training_updates=0)
    finally:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':run()
