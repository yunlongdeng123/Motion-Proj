"""No-update ownership counterfactuals; never chooses or trains automatically."""
import sys,json,copy,time,argparse
from pathlib import Path
import numpy as np
import torch
from train_surface import SurfaceExperiment,NEW,OLD
from run_experiments import write
import renderer

def ground_mask(e,kind):
    base=e.surface.ground.clone()
    if kind=='strict':return base
    a=e.surface.anchor_state;h=(a['means3D']@e.surface.plane[:3]+e.surface.plane[3])/e.surface.scale
    result=base|(h<.20)
    if kind in ['below_local','below_union']:
        source='selection_v1' if kind=='below_local' else 'selection_union_v1'
        with np.load(OLD/source/'selection.npz') as f:ids=torch.tensor(f['artifact_ids'],device='cuda')
        result[ids]=True
    return result

def run():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:
        start=time.time()
        while time.time()-start<4*3600:
            path=NEW/'finish_pipeline_status.json'
            if path.exists():
                state=json.loads(path.read_text())['state']
                if state=='complete':break
                if state in ['needs_review','evaluation_failed','deadline_exceeded']:raise SystemExit('Preserve failed pipeline for review')
            time.sleep(30)
        else:raise SystemExit('Waiting deadline exceeded')
    args=argparse.Namespace(output=str(NEW/'ownership_probe'),steps=0,warmup=1,restore=str(NEW/'joint_v1/final.pt'),probe=True)
    e=SurfaceExperiment(args)
    try:
        results={};original=e.surface.ground.clone()
        with torch.no_grad():
            for kind in ['strict','below','below_local','below_union']:
                e.priority();e.surface.ground.copy_(original);mask=ground_mask(e,kind);e.surface.ground.copy_(mask)
                metrics=e.metrics([0,6,12,18,24,36,48]);e.preview(kind)
                fixed=[];ref=None
                for i in [0,6,12,18,24,36,48]:
                    cam=copy.copy(e.cameras[12]);cam.timestamp=e.cameras[i].timestamp
                    pkg=e.render(i,instances=False,camera=cam);conf=pkg['road_confidence'][0]>.99999
                    if ref is None:ref=pkg['render'].clone();refmask=conf.clone()
                    common=conf&refmask;fixed.append(float(abs(pkg['render'][:,common]-ref[:,common]).mean()))
                results[kind]={**metrics,'removed_parent_points':int(mask.sum()),'fixed_supported_road_time_MAE':float(np.mean(fixed))}
                write(e.out/'comparisons.json',results)
                print(json.dumps({'variant':kind,**{k:v for k,v in results[kind].items() if k!='per_frame'}}),flush=True)
        e.verify_sources();e.update(state='complete',phase='no_update_ownership_counterfactuals_ready',training_updates=0,results=results,
            decision='Primary visual inspection and independent refit required; no winner chosen automatically')
    finally:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':run()
