"""Frozen upstream inference, saved Gaussian state, independent-process render restore."""
import os,sys,time,json,dataclasses,random
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
sys.path.insert(0,str(ROOT/'freeinpaint_code'))
import torch,numpy as np
import test as upstream

def move(v,device):
    if torch.is_tensor(v):return v.detach().to(device)
    if dataclasses.is_dataclass(v):return type(v)(**{f.name:move(getattr(v,f.name),device) for f in dataclasses.fields(v)})
    if isinstance(v,dict):return {k:move(x,device) for k,x in v.items()}
    if isinstance(v,list):return [move(x,device) for x in v]
    if isinstance(v,tuple):return tuple(move(x,device) for x in v)
    return v

def float_render(pred,args):
    from depth_anything_3.model.utils.gs_renderer import render_3dgs
    from depth_anything_3.utils.geometry import as_homogeneous
    K=pred['intrinsics'][0,0].clone();K[0]/=args.width;K[1]/=args.height
    rgb,depth,alpha=render_3dgs(extrinsics=as_homogeneous(pred['extrinsics'][0,0][None].float()),intrinsics=K[None],image_shape=(args.height,args.width),gaussian=pred['gaussians'],use_sh=True,color_mode='RGB+ED')
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
    return dict(rgb=rgb.cpu().numpy(),depth=depth.cpu().numpy(),alpha=alpha.cpu().numpy())

if '--restore' in sys.argv or '--recover-saved' in sys.argv:
    recovery='--recover-saved' in sys.argv;flag='--recover-saved' if recovery else '--restore'
    ckpath=Path(sys.argv[sys.argv.index(flag)+1]);ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    destination=ckpath.parent/('recovered_render' if recovery else 'restored');destination.mkdir(exist_ok=True)
    pred=move(ck['prediction'],'cuda');args=ck['args'];args.save_gs=False
    torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    with torch.no_grad():
        actual=float_render(pred,args)
        if recovery:np.savez_compressed(ckpath.parent/'reference_float_render.npz',**actual)
        expected=dict(np.load(ckpath.parent/'reference_float_render.npz'))
        comparison={k:dict(exact=bool(np.array_equal(actual[k],expected[k])),max_abs=float(np.abs(actual[k]-expected[k]).max())) for k in actual}
        upstream.render_outputs(pred,ck['item'],ck['reference'],args,destination)
    (destination/'restore_receipt.json').write_text(json.dumps(dict(completed=True,recovered_after_renderer_build_failure=recovery,comparisons=None if recovery else comparison,training_updates=0,optimizer='none, frozen inference',quality_accepted=False),indent=2))
    print(json.dumps(comparison),flush=True)
else:
    original=upstream.render_outputs;start=time.time()
    def save_then_render(prediction,item,reference,args,folder):
        torch.save(dict(prediction=move(prediction,'cpu'),item=move(item,'cpu'),reference=reference.cpu(),args=args,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),folder/'inference_state.pth')
        np.savez_compressed(folder/'reference_float_render.npz',**float_render(prediction,args))
        result=original(prediction,item,reference,args,folder)
        (folder/'smoke_receipt.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),training_updates=0,purpose='unposed external-prior interface probe; not inserted in baseline or treated as true road',quality_accepted=False,human_verdict=None),indent=2))
        return result
    upstream.render_outputs=save_then_render
    upstream.main()
