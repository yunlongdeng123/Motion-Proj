"""Frozen row-level ghost suppression controls. Mixed evidence stays unknown."""
import os,json,time
from pathlib import Path
import numpy as np
from train_actor import NativeMotion,ROOT,OLD,torch,cpu
from segmentation import CONTEXTS,write
from diagnose_first_badcases import static_without,save
from gpu_slot import exclusive
from v43_resource_control import require_start

def main():
    ctx=CONTEXTS[0];require_start(ctx);out=ROOT/'first_static_ghost_sweep_v1';out.mkdir(exist_ok=False)
    agg=np.load(ROOT/'attribution_v2'/ctx/'aggregate.npz');ns=int(agg['branch_sizes'][0]);m=agg['mass'][:ns];total=m[:,21];tf=m[:,20]/np.maximum(total,1e-8);vf=m[:,14:20].sum(1)/np.maximum(total,1e-8)
    sweep=[]
    for t in [.02,.05,.1,.2,.5,.85]:
        for v in [.1,.3,.5,.65]:
            mask=(total>1)&(tf>t)&(vf>v);sweep.append(dict(target_fraction=t,vehicle_fraction=v,rows=int(mask.sum()),target_alpha_mass_captured_fraction=float(m[mask,20].sum()/m[:,20].sum())))
    # Two explicitly registered sensitivity settings, not chosen using image scores.
    choices=[('affinity_50',.5,.5),('affinity_20',.2,.3)];records=[]
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',training_updates=0);write(out/'status.json',status)
    with exclusive(),torch.no_grad():
        require_start(ctx);started=time.monotonic();torch.set_num_threads(1);n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);rows=json.loads((OLD/'motion_edit_suite_v1'/ctx/'delete_target.json').read_text())['frames']
        for name,t,v in choices:
            mask=(total>1)&(tf>t)&(vf>v);ids=np.flatnonzero(mask).astype('int32');np.save(out/(name+'_rows.npy'),ids);mt=torch.from_numpy(mask).cuda();metrics=[]
            for row in rows:
                raw=row['raw'];body=torch.from_numpy(np.load(ROOT/'datasets'/ctx/('%03d.npz'%raw))['body']).cuda();base=n.render(row,dict(family='delete'));changed=static_without(n,row,mt);delta=(changed['rgb']-base['rgb']).abs().mean(0);save(out/name/'native_rgb'/('%03d.png'%raw),changed['rgb']);metrics.append(dict(raw=raw,target_region_RGB_change_MAE=float(delta[body].mean()),outside_target_RGB_change_MAE=float(delta[~body].mean()),outside_target_changed_pixels=int((delta[~body]>1/255).sum())))
            records.append(dict(name=name,thresholds=dict(target_fraction=t,vehicle_fraction=v,total_mass_min=1),suppressed_rows=len(ids),metrics=metrics,whole_row_identity_confirmed=False,accepted_as_fix=False))
        assert n.unchanged();write(out/'summary.json',dict(context=ctx,sweep=sweep,probes=records,selection_source='Original49-frame alpha*T to observed target mask and predicted vehicle classes',important='Car appearance affinity can live on background geometry; suppression is causal diagnostic, not ownership acceptance or background completion',source_checkpoint_unchanged=True,training_updates=0));status.update(phase='complete',wall_seconds=time.monotonic()-started);write(out/'status.json',status)
if __name__=='__main__':main()
