"""Read-only shared-renderer diagnostics; each RGB layer shares the final sort.

Requires an available native CUDA environment. Counterfactual row copies are
diagnostic interventions, never writes to model parameters or saved outputs.
"""
from pathlib import Path
import argparse,json,time
import numpy as np

def main():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: no renders or diagnosis results produced')
    from drivingeditor_dynamic_edit.v44_factory import from_manifest
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--saved',required=True,type=Path)
    p.add_argument('--frames',required=True,type=int,nargs='+');p.add_argument('--target',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);start=time.monotonic();scene,m=from_manifest(a.manifest)
    if not scene._include_dynamic:raise ValueError('This diagnostic requires both native row branches')
    ns,nd=len(scene.static.get_xyz),len(scene.dynamic.get_xyz);ops=scene._bindings();names=['target','nonfocus_static','nonfocus_dynamic','road','other_surface'];records=[]
    saved={r['frame_key']:r for r in json.loads((a.saved/'provenance.json').read_text())['frames']}
    with torch.no_grad():
        for key in a.frames:
            rows,owners,cam,coverage=scene._rows(key,(a.target,),(),False)
            group=torch.full_like(owners,4);group[:ns]=1;group[ns:ns+nd]=2
            group[rows['features'][:,5]>0]=3;group[owners==a.target]=0
            features=rows['features'].clone();features[:,8:]=0
            for g in range(5):features[:,8+3*g:11+3*g]=rows['colors_precomp']*(group==g)[:,None]*features[:,1:2]
            deleted=rows['mask']&(owners!=a.target)
            env=scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            original=ops.prepare_rasterization(cam,scene.static,scene.pipe,dynamic=False)['opacities']
            for mode in ['final','gates_off','road_off','road_only']:
                mask=deleted.clone();opacity=rows['opacities']
                if mode=='gates_off':opacity=opacity.clone();opacity[:ns]=original
                if mode=='road_off':mask&=group!=3
                if mode=='road_only':mask&=group==3
                out=ops.raster(cam,scene.pipe,0,dict(rows,features=features,mask=mask,opacities=opacity),scene._background)
                groups=out['extra'][6:21].reshape(5,3,*out['rgb'].shape[1:]);envT=(1-out['alpha'])*env
                rgb=out['rgb']+envT;err=float(abs(groups.sum(0)+envT-rgb).max())
                if err>5e-5:raise RuntimeError('Same-sort colour additivity mismatch: '+str(err))
                rec=dict(frame=key,mode=mode,RGB_additivity_maxabs=err)
                if mode=='final':
                    expected=np.load(a.saved/saved[key]['files']['float_npz'])['rgb']
                    rec['saved_RGB_maxabs']=float(abs(rgb.cpu().numpy()-expected).max())
                    if rec['saved_RGB_maxabs']>3e-6:raise RuntimeError('Saved result parity failed')
                np.savez_compressed(a.output/f'{key}_{mode}.npz',rgb=rgb.cpu().numpy(),RGB_T=groups.cpu().numpy(),RGB_names=np.array(names),environment_RGB_T=envT.cpu().numpy(),alpha=out['alpha'].cpu().numpy())
                records.append(rec)
    scene.assert_persistent_unchanged()
    (a.output/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_JOINT_DIAGNOSTICS',rows=records,training_updates=0,parent_unchanged=True,seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
