"""Single contribution-driven batch; preserve bounded frozen 3D fragments."""
from pathlib import Path
import hashlib, itertools, json, sys, time
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parent.parent
R4=ROOT.parent/'road_native30k_red_rebuild_20261125_r4'
R5=ROOT.parent/'road_teacher_gaussian_rebuild_20261125_r5'
OBS=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/observations')
sys.path.insert(0,str(R4/'code'))
from native30k_adapter import Native30KAdapter
from prepare_region_retirement_v5 import subset,load,in_box,sample

FRAMES=[106,107,108,109,110,120,137,165]
A9='NTjq-OhZ_97OMJ4xwJAxXg'

def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def basis(packet,ids):
    if packet['scales'] is not None:
        q=packet['rotations'][ids].detach().cpu().numpy()
        r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        return r*packet['scales'][ids].detach().cpu().numpy()[:,None,:]
    c=packet['cov3D_precomp'][ids].detach().cpu().numpy();m=np.zeros((len(c),3,3))
    m[:,0,0]=c[:,0];m[:,0,1]=m[:,1,0]=c[:,1];m[:,0,2]=m[:,2,0]=c[:,2]
    m[:,1,1]=c[:,3];m[:,1,2]=m[:,2,1]=c[:,4];m[:,2,2]=c[:,5]
    return np.linalg.cholesky(m+np.eye(3)*1e-14)
def project(rt,raw,points):
    h=rt.camera_headers[raw];q=points@h['W'][:3,:3].T+h['W'][:3,3]
    with np.errstate(divide='ignore',invalid='ignore'):uv=q[:,:2]/q[:,2:3]*[h['fx'],h['fy']]+[h['cx'],h['cy']]
    return uv,q[:,2]
def main():
    started=time.perf_counter();out=ROOT/'bounded_context_v2';out.mkdir(exist_ok=False)
    rt=Native30KAdapter(R4/'config/native30k.json',out/'native');t=rt.torch
    src=rt.owners(110).cpu().numpy();N=rt.native_row_count
    prior=np.zeros(N,bool);prior[load(ROOT/'red_takeover_v5/retirement.npz')['remove_global_row_ids']]=True
    rows=[];mass=np.zeros(N,np.float64);road_masks={};boxes={}
    for raw in FRAMES:
        road_masks[raw]=load(R4/'masks'/f'{raw:03}'/'native.npz')['red_rebuild']
        boxes[raw]=[b for b in json.loads((OBS/f'{raw:03}.json').read_text())['boxes'] if b['track']!=A9]
        p=rt.packet(raw);ids=t.nonzero(rt.native_visible_mask(raw,p)&~t.as_tensor(prior,device='cuda')).reshape(-1)
        p=subset(p,ids);f=p['features'].detach().clone();f[:,9]=1;f.requires_grad_(True)
        o=rt.raster_packet(raw,dict(p,features=f));mask=t.as_tensor(road_masks[raw],device='cuda')
        loss=(o['extra'][7]*mask).sum();g=t.autograd.grad(loss,f)[0][:,9].detach().cpu().numpy()
        mass[ids.cpu().numpy()]+=np.maximum(g,0)
        rows.append(dict(raw=raw,target_road_pixels=int(mask.sum()),old_road_alpha_mass=float(g.sum())))
        print(json.dumps(rows[-1]),flush=True);del p,f,o,loss,g
    parent=np.flatnonzero((mass>.01)&~prior&(src!=10))
    p=rt.packet(110);model=p['means3D'][t.as_tensor(parent,device='cuda')].detach().cpu().numpy().astype(np.float64)
    B=basis(p,t.as_tensor(parent,device='cuda'));sigma=np.linalg.norm(B,axis=1).max(-1)/rt.meter_to_model
    inv=np.linalg.inv(rt.transform);world=model@inv[:3,:3].T+inv[:3,3]
    body=np.zeros(len(parent),bool)
    for box in boxes[110]:body|=in_box(world,box,.10)
    preserve=body&(sigma<.12)
    parent=parent[~preserve];B=B[~preserve];sigma=sigma[~preserve]
    grid=np.asarray(list(itertools.product([-2.,-1.,0.,1.,2.],repeat=3)))
    grid=grid[np.argsort((grid*grid).sum(1),kind='stable')]
    beta=np.minimum(.35,.08/np.maximum(sigma,1e-8))
    offsets=grid[None]*np.sqrt(1-beta[:,None,None]**2)
    eligible=np.ones((len(parent),len(grid)),bool);body_any=np.zeros_like(eligible)
    for raw in FRAMES:
        p=rt.packet(raw);ids=t.as_tensor(parent,device='cuda');means=p['means3D'][ids].detach().cpu().numpy().astype(np.float64)
        L=basis(p,ids);child=means[:,None,:]+np.einsum('nij,nkj->nki',L,offsets)
        q=child.reshape(-1,3);uv,z=project(rt,raw,q)
        # This is an offline 3D support selection, not a render-time image gate.
        hit=sample(binary_dilation(road_masks[raw],iterations=3),uv,z).reshape(eligible.shape)
        eligible&=~hit
        physical=q@inv[:3,:3].T+inv[:3,3];inside=np.zeros(len(q),bool)
        for box in boxes[raw]:inside|=in_box(physical,box,.03)
        body_any|=inside.reshape(eligible.shape)
    valid=eligible|body_any
    pi=[];oi=[]
    for i in range(len(parent)):
        keep=np.flatnonzero(valid[i])[:27]
        pi.extend([i]*len(keep));oi.extend(keep.tolist())
    pi=np.asarray(pi,np.int64);oi=np.asarray(oi,np.int64)
    parent_ids=parent[pi];off=offsets[pi,oi].astype(np.float32);sc=beta[pi].astype(np.float32)
    alpha=p['opacities'][t.as_tensor(parent,device='cuda')].detach().cpu().numpy().reshape(-1)[pi]
    peak=alpha*np.exp(-.5*(off*off).sum(-1))
    factors=np.log1p(-np.clip(peak,1e-8,1-1e-6))/np.log1p(-np.clip(alpha,1e-8,1-1e-6))
    np.savez_compressed(out/'retained_fragments.npz',parent_global_row_ids=parent_ids,offsets=off,scale_factors=sc,opacity_factors=factors.astype(np.float32))
    remove=prior.copy();remove[parent]=True;remove[src==10]=True
    protected=np.flatnonzero(~remove&(src!=10))
    np.savez_compressed(out/'retirement.npz',remove_global_row_ids=np.flatnonzero(remove),protected_green_core_global_row_ids=protected)
    receipt=dict(complete=True,retirement=pin(out/'retirement.npz'),retained_fragments=pin(out/'retained_fragments.npz'),
        native_contract=pin(R4/'config/native30k.json'),source_retirement=pin(ROOT/'red_takeover_v5/retirement.npz'),
        parent_count=len(parent),frozen_fragment_count=len(pi),total_retired=int(remove.sum()),rows=rows,
        old_Gaussian_parameter_updates=0,offline_fragment_support='fixed 3D Gaussian offsets inside retained vehicle boxes or outside all eight road support frusta',
        render_time_image_gates=False,child_max_sigma_m=.08,max_children_per_parent=27,source_A9_remaining=int(((src==10)&~remove).sum()),
        body_small_parent_preserved=int(preserve.sum()),seconds=time.perf_counter()-started)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);rt.assert_fixed()
if __name__=='__main__':main()
