"""One regional native-row retirement batch. Original Gaussian parameters stay frozen."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.spatial import cKDTree

R4=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_native30k_red_rebuild_20261125_r4')
OBS=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/observations')
A9='NTjq-OhZ_97OMJ4xwJAxXg';A20='bzFbWRrSGOgEaXxnuGV0tw'

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def subset(p,ids):
    n=len(p['means3D']);return{k:v[ids]if v is not None and hasattr(v,'ndim')and v.ndim and len(v)==n else v for k,v in p.items()}
def in_box(world,box,margin):
    pose=np.asarray(box['world_from_actor']).reshape(4,4);inverse=np.linalg.inv(pose)
    local=world@inverse[:3,:3].T+inverse[:3,3]
    return(np.abs(local)<=np.asarray(box['dimensions'])[None]/2+margin).all(1)
def sample(mask,uv,depth):
    valid=np.isfinite(uv).all(1)&(depth>0);xy=np.rint(np.where(np.isfinite(uv),uv,-1)).astype(np.int64)
    valid&=(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0]);out=np.zeros(len(uv),bool);i=np.flatnonzero(valid);out[i]=mask[xy[i,1],xy[i,0]];return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--native-contract',type=Path,default=R4/'config/native30k.json');p.add_argument('--adapter',type=Path,default=R4/'code/native30k_adapter.py')
    p.add_argument('--previous-retirement',type=Path,default=R4/'deletion_green_preserved_v2/retirement.npz');p.add_argument('--mask-root',type=Path,default=R4/'masks')
    p.add_argument('--observations',type=Path,default=OBS);p.add_argument('--ground-field',type=Path,default=R4/'initialization/new_field.npz')
    p.add_argument('--reference-raw',type=int,default=110);p.add_argument('--box-margin-m',type=float,default=.35);p.add_argument('--red-contribution-eps',type=float,default=1e-6)
    p.add_argument('--height-min-m',type=float,default=-.5);p.add_argument('--height-max-m',type=float,default=3.5);p.add_argument('--large-sigma-m',type=float,default=.10);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True);start=time.perf_counter();sys.path.insert(0,str(a.adapter.parent))
    spec=importlib.util.spec_from_file_location('_region_native30k',a.adapter);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=m.Native30KAdapter(a.native_contract,a.output/'native');t=base.torch;frames=read(a.training)['train_frames'];N=base.native_row_count
    semantic=base.owners(frames[0],semantic=True).cpu().numpy();source=base.owners(frames[0]).cpu().numpy();source10=source==10;semantic10=semantic==10
    previous=np.zeros(N,bool);previous[load(a.previous_retirement)['remove_global_row_ids']]=True
    ground=load(a.ground_field)['world_xyz'];origin=ground.mean(0);_,_,V=np.linalg.svd(ground-origin,full_matrices=False);normal=V[-1];normal*=np.sign(normal[2]);tree=cKDTree(ground[:,:2]);inv=np.linalg.inv(base.transform)
    protected=(source>=0)&(source!=10);a20_body=(source==21);center_red=np.zeros(N,bool);a9_volume=np.zeros(N,bool);red_volume=np.zeros(N,bool);large_cross=np.zeros(N,bool)
    packets={};indices={};masks_by_raw={};geometry_rows=[]
    with t.no_grad():
        for raw in frames:
            packet=base.packet(raw);ids=t.nonzero(base.native_visible_mask(raw,packet)).reshape(-1);packet=subset(packet,ids);gid=ids.cpu().numpy()
            world=packet['means3D'].cpu().numpy().astype(np.float64)@inv[:3,:3].T+inv[:3,3]
            h=base.camera_headers[raw];model=packet['means3D'].cpu().numpy().astype(np.float64);cam=model@h['W'][:3,:3].T+h['W'][:3,3]
            with np.errstate(divide='ignore',invalid='ignore'):uv=cam[:,:2]/cam[:,2:3]*np.array([h['fx'],h['fy']])+[h['cx'],h['cy']]
            masks=load(a.mask_root/f'{raw:03d}'/'native.npz');red=sample(masks['red_rebuild'],uv,cam[:,2]);go=sample(masks['keep_green']|masks['keep_orange'],uv,cam[:,2])
            body20=np.zeros(len(gid),bool);deleted_box=np.zeros(len(gid),bool);other_entities=np.zeros(len(gid),bool)
            for box in read(a.observations/f'{raw:03d}.json')['boxes']:
                if box['track']==A9:deleted_box|=in_box(world,box,a.box_margin_m)
            own20=(source[gid]==21)
            core20=own20&(body20|sample(masks.get('A20',masks['keep_green']),uv,cam[:,2]))
            protect=own20
            if raw==a.reference_raw:protect|=go&~deleted_box
            protect|=own20
            protected[gid]|=protect;a20_body[gid]|=core20&~source10[gid]
            height=(world-origin)@normal;slab=(height>=a.height_min_m)&(height<=a.height_max_m)
            distance,_=tree.query(world[:,:2],workers=2)
            if packet['scales'] is not None:
                sigma=packet['scales'].max(-1).values.cpu().numpy()/base.meter_to_model
            else:
                covariance=packet['cov3D_precomp']
                sigma=covariance[:,[0,3,5]].sum(-1).clamp_min(0).sqrt().cpu().numpy()/base.meter_to_model
            center_red[gid]|=red&slab;a9_volume[gid]|=deleted_box;red_volume[gid]|=(distance<=.0145)&slab
            large_cross[gid]|=slab&(sigma>=a.large_sigma_m)&(distance<=3*sigma+.0145)
            packets[raw]=packet;indices[raw]=ids;masks_by_raw[raw]=masks
            geometry_rows.append(dict(raw=raw,visible_native_rows=len(gid),red_centers=int(red.sum()),A9_volume_centers=int(deleted_box.sum()),A20_core_centers=int(core20.sum())))
    # A label alone cannot exempt a red-domain non-entity tail. True protected
    # centers, including original A20 bodies, are the immutable exclusion set.
    # Original dynamic lineage is authoritative: source10 matches the A9 box,
    # source21 matches A20. The transferred semantic sidecar is not row identity.
    protected &= ~source10
    mandatory=source10
    static=np.arange(N)<base.native_static_rows
    pre_remove=(previous&static)|mandatory
    mass=np.zeros((N,3),np.float64);measure_rows=[]
    # Minimal safe v3: all original A20 identities are protected. Additional
    # retirement is confined to static non-entity centers in the red volume;
    # current GPU decomposition already established the unwanted front layer.
    regional=static&(center_red|a9_volume|red_volume)
    crossing=np.zeros(N,bool)
    remove=mandatory|(((previous&static)|regional|crossing)&~protected)
    assert not(remove&protected).any();removed=np.flatnonzero(remove);core=np.flatnonzero(protected);tails=np.flatnonzero(remove&(semantic==21))
    np.savez_compressed(a.output/'retirement.npz',remove_global_row_ids=removed,protected_green_core_global_row_ids=core,
        removed_semantic21_tail_global_row_ids=tails,protected_A20_core_global_row_ids=np.flatnonzero(source==21),native_static_rows=np.asarray(base.native_static_rows),native_dynamic_rows=np.asarray(base.native_dynamic_rows))
    np.savez_compressed(a.output/'region_selection.npz',center_red=center_red,A9_volume=a9_volume,red_volume=red_volume,protected_core=protected)
    # A single actual retained-context preview, with Gaussian and environment
    # terms separated so a hole is not mistaken for successful reconstruction.
    raw=a.reference_raw;packet=base.packet(raw);ids=t.nonzero(base.native_visible_mask(raw,packet)&~t.as_tensor(remove,device='cuda')).reshape(-1);packet=subset(packet,ids)
    f=packet['features'].new_zeros((len(ids),24));f[:,:2]=packet['features'][:,:2];f[:,2]=1;f[:,3]=t.as_tensor(source[ids.cpu().numpy()]==21,device='cuda');packet=dict(packet,features=f)
    with t.no_grad():
        out=base.raster_packet(raw,packet);rgb=out['rgb'];raw_rgb=out['raw_rgb'];arrays={k:v.detach().cpu().numpy()for k,v in dict(rgb=rgb,gaussian_rgb=raw_rgb,environment_rgb=rgb-raw_rgb,alpha=out['alpha'],actor_alpha=out['extra'][1],depth_sum_m=out['depth_sum_m']).items()}
    np.savez_compressed(a.output/f'context_only_{raw:03d}.npz',**arrays)
    from PIL import Image
    Image.fromarray(np.rint(np.clip(arrays['rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(a.output/f'context_only_{raw:03d}.png')
    base.assert_fixed();receipt=dict(schema='scene065.region_red_takeover_retirement.v3',complete=True,retirement=pin(a.output/'retirement.npz'),native_contract=pin(a.native_contract),previous_retirement=pin(a.previous_retirement),
        source_masks=pin(a.mask_root/'manifest.json'),frames=frames,geometry_rules=geometry_rows,measurements=measure_rows,remove_total=len(removed),additional_removed=int((remove&~previous).sum()),
        protected_green_core_rows=len(core),protected_A20_body_rows=int((a20_body&protected).sum()),protected_core_removed=0,source_A9_remaining=int((source10&~remove).sum()),source_A20_removed=int(((source==21)&remove).sum()),semantic_A9_remaining_not_identity=int((semantic10&~remove).sum()),
        removed_semantic21_tail_rows=len(tails),cross_boundary_retired_rows=int((remove&crossing).sum()),outside_support_loss_mass=dict(green=float(mass[remove,1].sum()),orange=float(mass[remove,2].sum())),
        rules=dict(red_mass_dominance_veto=False,height_slab_m=[a.height_min_m,a.height_max_m],box_margin_m=a.box_margin_m,semantic21_global_veto=True,
                   core='original checkpoint dynamic source10 A9 removed completely; source21 A20 and all other dynamic vehicles preserved; static reference green/orange centers preserved',cross_boundary='no kernel edits or covariance clipping; necessary outside reconstruction remains separate'),
        original_parameter_optimizer_updates=0,old_Gaussian_parameter_edits=0,native_forward_calls=base.calls,native_backward_calls=0,wall_seconds=time.perf_counter()-start)
    write(a.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
