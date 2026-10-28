"""Exact pre-rule inputs and post-peel cubemap contributors; no edits."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_detection import ROOT,CTX,load_native,kwargs,attribution,authorized,write,torch,cpu,exclusive
from render_detection import maps_for
from classify_detection import CHANNELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    authorized();a.output.mkdir(parents=True,exist_ok=False);base=ROOT/'unknown_detection_v3';start=time.monotonic();receipts=[]
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz)
        counts=np.zeros(ns,np.uint16)
        for path in sorted((base/'body_probe_r1').glob('[0-9][0-9][0-9].npz')):
            e=np.load(path);v=e['without_actor'];hit=(v[:,0]>.02)&(v[:,0]/np.maximum(v[:,-1],1e-8)>.05);counts[e['ids'][hit]]+=1
        np.save(a.output/'body_counts.npy',counts)
        import nvdiffrast.torch as dr
        for i,f in enumerate(frames):
            raw=f['raw'];r=a.run/f'{raw:03d}'
            if not r.exists():continue
            authorized();cam,st,dy=kwargs(n,actor,f,i)
            maps,_,_,_=maps_for(raw,ds,base/'shadow_evidence_r3',base/'body_probe_r1',n,cam,base/'contact_effect_evidence_r1')
            out,w,closure=attribution(n,cam,st,maps);w=cpu(w);ids=np.flatnonzero(w[:,-1]>0)
            with torch.no_grad():z=(torch.cat((st['means3D'],torch.ones_like(st['means3D'][:,:1])),1)@cam.world_view_transform[:,2:3])[:,0]/n.scale
            np.savez_compressed(a.output/f'{raw:03d}_gate_inputs.npz',ids=ids.astype('int32'),weights=w[ids],center_z=cpu(z)[ids],body_counts=counts[ids])
            del out,w
            final=json.loads((r/'final.json').read_text());state=np.load(r/final['final_stage']/'maps.npz');domain=np.load(r/'domains.npz')['maps']
            transmission=1-state['alpha'];masks=np.stack([domain[2],domain[3],domain[1]])
            # Independent cubemap channel adjoint gives exact bilinear texel support.
            dirs=cam.get_world_directions(False).permute(1,2,0)
            coords=(dirs.reshape(-1,3)@n.env.to_opengl.T).reshape(1,*dirs.shape).contiguous()
            aux=torch.zeros((*n.env.base.shape[:3],3),device='cuda',requires_grad=True)
            sampled=dr.texture(aux[None,...],coords,filter_mode='linear',boundary_mode='cube')[0]
            loss=(sampled*torch.from_numpy((masks*transmission).transpose(1,2,0)).cuda()).sum()
            texel=cpu(torch.autograd.grad(loss,aux)[0]);flat=texel.reshape(-1,3);sel=np.flatnonzero(flat.sum(1)>0);rgb=cpu(n.env.base).reshape(-1,3)[sel]
            actual=np.stack([(state['environment_rgb']*mask).sum((1,2)) for mask in masks])
            predicted=flat[sel].T@rgb;err=float(np.max(np.abs(predicted-actual)/np.maximum(1,np.abs(actual))))
            assert err<.003,err
            np.savez_compressed(a.output/f'{raw:03d}_environment_texels.npz',flat_ids=sel.astype('int32'),weights=flat[sel],rgb=rgb,shape=np.array(n.env.base.shape),predicted_rgb_integral=predicted,actual_rgb_integral=actual)
            receipts.append(dict(raw=raw,original_static_rows_with_any_support=len(ids),adjoint_closure=closure,environment_texel_rows=len(sel),environment_integral_relative_error=err,
                environment_negative_texels=int((rgb<0).any(1).sum()),environment_channels=['body','effect','broad_domain'],final_stage=final['final_stage']))
        assert n.unchanged()
    write(a.output/'summary.json',dict(schema='recall-gate-and-environment-source-v1',channels=CHANNELS,static_rows=ns,omitted_rows_are_zero=True,receipts=receipts,source_checkpoint_unchanged=True,training_updates=0,environment_modified=False,wall_seconds=time.monotonic()-start))
    print(json.dumps(receipts),flush=True)

if __name__=='__main__':main()
