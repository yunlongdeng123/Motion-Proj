"""Zero-update gate-loss diagnostics at the saved final state; no optimizer."""
from pathlib import Path
import argparse,json,time
import numpy as np

def main():
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.v44_factory import from_manifest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True);start=time.monotonic();r=a.run
    scene,m=from_manifest(r/'road_manifest.json');ops=scene._bindings();device=scene.static.get_xyz.device
    ids=torch.as_tensor(np.load(r/'gate1000/contributors.npz')['ids'],device=device,dtype=torch.long)
    ck=torch.load(r/'gate1000/checkpoint001000.pt',map_location='cpu',weights_only=False);gate=torch.nn.Parameter(ck['gate'].to(device).clone())
    load=lambda p:json.loads(p.read_text());saved={z['frame_key']:z for z in load(r/'delete_white/provenance.json')['frames']};base={z['frame_key']:z for z in load(r/'semantic_delete/provenance.json')['frames']}
    records=[];vectors={}
    def linear(x):
        x=x.clamp(0,1);return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    def stats(g):
        x=g.detach();return dict(l2=float(x.norm()),l1=float(x.abs().sum()),reduce_opacity_L1_fraction=float(x.clamp_min(0).sum()/x.abs().sum().clamp_min(1e-20)),increase_opacity_L1_fraction=float((-x).clamp_min(0).sum()/x.abs().sum().clamp_min(1e-20)))
    def cosine(x,y):return float((x@y)/(x.norm()*y.norm()).clamp_min(1e-20))
    for key in [60,119,150,198]:
        values,owners,cam,_=scene._rows(key,(10,),(),False)
        opacity=values['opacities'].index_copy(0,ids,values['opacities'][ids]*gate.sigmoid()[:,None]);mask=values['mask']&(owners!=10)
        out=ops.raster(cam,scene.pipe,0,dict(values,opacities=opacity,mask=mask),opacity.new_zeros(3));rgb=out['rgb']+(1-out['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        expected=np.load(r/'delete_white'/saved[key]['files']['float_npz'])['rgb'];parity=float(abs(rgb.detach().cpu().numpy()-expected).max());assert parity<3e-6,parity
        d=np.load(r/'world_road/views'/f'{key:03d}.npz');frame=next(z for z in m.data['frames'] if z['key']==key)
        target=torch.as_tensor(np.asarray(Image.open(frame['instance_mask']).resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST)).copy()==10,device=device)
        weight=torch.as_tensor(d['fit'],device=device);keep=torch.as_tensor(d['keep'],device=device);mark=torch.as_tensor(d['marking'],device=device);other=torch.as_tensor(d['other'],device=device);left=torch.as_tensor(~d['inside']&~d['other'],device=device)
        ref=torch.as_tensor(d['reference_linear'],device=device);before=torch.as_tensor(np.load(r/'semantic_delete'/base[key]['files']['float_npz'])['rgb'],device=device);delta=abs(rgb-before)
        def avg(mask):return delta[:,mask].mean() if mask.any() else delta.sum()*0
        def weighted(w):return (delta*w).sum()/(3*w.sum().clamp_min(1))
        fixedkeep=keep*~target;fixedleft=left&~target
        components=dict(road=(abs(linear(rgb)-ref)*weight).sum()/(3*weight.sum().clamp_min(1)),
            original_preserve=2*weighted(keep),original_left=2*avg(left),
            corrected_preserve=2*weighted(fixedkeep),corrected_left=2*avg(fixedleft),
            target_preserve=2*(delta*keep*target).sum()/(3*keep.sum().clamp_min(1)),
            target_left=2*(delta[:,left&target].sum()/(3*left.sum().clamp_min(1))),
            unchanged_guard=4*avg(mark)+3*avg(other)+.0005*(1-gate.sigmoid()).mean())
        grads={n:torch.autograd.grad(v,gate,retain_graph=True)[0].detach() for n,v in components.items()}
        old=grads['road']+grads['original_preserve']+grads['original_left']+grads['unchanged_guard'];new=grads['road']+grads['corrected_preserve']+grads['corrected_left']+grads['unchanged_guard'];wrong=grads['target_preserve']+grads['target_left']
        records.append(dict(frame=key,saved_RGB_maxabs=parity,target_pixels=int(target.sum()),keep_target_pixels=int(((keep>0)&target).sum()),left_target_pixels=int((left&target).sum()),
            component_losses={n:float(v) for n,v in components.items()},gradients={n:stats(g) for n,g in grads.items()},
            cosine_road_vs_target_preserve=cosine(grads['road'],wrong),cosine_original_vs_corrected_total=cosine(old,new),
            target_preserve_dot_road_descent=float(wrong@(-grads['road'])),original_total=stats(old),corrected_total=stats(new)))
        vectors[str(key)]={n:g.cpu().numpy() for n,g in grads.items()};print(json.dumps(records[-1]),flush=True)
        del out,rgb,values,opacity,components,grads,old,new,wrong;torch.cuda.empty_cache()
    scene.assert_persistent_unchanged()
    np.savez_compressed(a.output/'gradient_vectors.npz',ids=ids.cpu().numpy(),**{k+'_'+n:g for k,v in vectors.items() for n,g in v.items()})
    (a.output/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_ZERO_UPDATE_GATE_GRADIENTS',records=records,training_updates=0,optimizer_created=False,parent_unchanged=True,seconds=time.monotonic()-start,
       role='four-frame local gradients at frozen gate state, not full-training causal effect or edited-quality result',automatic_shutdown=False),indent=2)+'\n')

if __name__=='__main__':main()
