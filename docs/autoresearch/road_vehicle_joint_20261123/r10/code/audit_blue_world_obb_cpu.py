"""Native checkpoint CPU geometry versus original same-time GT actor OBBs.

Blue rows are read only for attribution. Non-colour proposal IDs are separate.
Projection-envelope overlap is conservative, not exact ellipsoid intersection.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.spatial import cKDTree


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def qmatrix(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
def quantile(x,w):
    if not (w>0).any():return None
    order=np.argsort(x);v=x[order];s=np.cumsum(w[order],dtype=np.float64);return np.interp(np.array([0,.1,.5,.9,1])*s[-1],s,v).tolist()
def box_test(world,cov,box,margin=0.):
    pose=np.asarray(box['world_from_actor'],np.float64).reshape(4,4);R=pose[:3,:3];half=np.asarray(box['dimensions'],np.float64)/2+margin
    local=(world-pose[:3,3])@R
    std=np.sqrt(np.maximum(np.einsum('ki,nkl,li->ni',R,cov,R),0))
    center=(abs(local)<=half).all(1);envelope=(abs(local)<=half+3*std).all(1)
    return center,envelope


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    start=time.time();r=a.root;assert not a.output.exists();import torch
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    cfg=read(r/'local_measure_contract_v2.json');base=read(cfg['baseline_contract']);scene=read(base['scene_manifest']);proof=read(r/'measure_v1/native_gate0.proof.json');frames=cfg['frames']
    blue=bank(r/'audit/blue_row_eligibility_cpu_v1/blue_row_evidence.npz');proposal=bank(r/'ground_unowned_policy_v1/diagnostic_native_candidates.npz')
    ids=np.union1d(blue['global_row_ids'],proposal['global_row_ids']);ns=proof['native_topology']['static_rows'];nd=proof['native_topology']['dynamic_rows']
    assert ((ids>=0)&(ids<ns+nd)).all()
    source={s['role']:s for s in proof['native_source_pins']}
    payload={}
    for role in ['static','dynamic']:
        payload[role],step=torch.load(source[role]['path'],map_location='cpu',mmap=True,weights_only=False)
        assert len(payload[role])==18 and step==30000
    matrix_path=Path(scene['world_to_model']);T=np.loadtxt(matrix_path);inv=np.linalg.inv(T);meter_to_model=float(np.linalg.norm(T[:3,:3],axis=0).mean())
    binding=read(Path(scene['model'])/'actual_scene_binding.json');times={str(k).split('/')[-1]:float(t) for k,t in zip(binding['frame_ids'],binding['timestamps'])}
    registry_path=Path(base['observations']).parent/'track_registry.json';registry=read(registry_path)
    tracks={v['alias']:k for k,v in registry['actors'].items() if v.get('alias') in ['a9','a20']};assert set(tracks)=={'a9','a20'}
    obs_paths=[Path(base['observations'])/f'{raw:03d}.json' for raw in frames];observations=[read(p) for p in obs_paths]
    geometry_path=r/'audit/local_surface_candidate_v2/candidate_geometry.npz';geometry=bank(geometry_path)
    cells=geometry['cell_world_xyz'][geometry['cell_asset_mask']];normals=geometry['cell_normal_world'][geometry['cell_asset_mask']];tree=cKDTree(cells[:,:2])
    bindex=np.searchsorted(ids,blue['global_row_ids']);pindex=np.searchsorted(ids,proposal['global_row_ids'])
    a.output.mkdir(parents=True);stats=[];locator_pins=[];gt=[]
    for j,(raw,obs) in enumerate(zip(frames,observations)):
        model=np.zeros((len(ids),3),np.float32);sigma=np.zeros_like(model);q=np.zeros((len(ids),4),np.float32)
        t=times[f'0065{raw:03d}']
        for role,take,indices in [('static',ids<ns,ids[ids<ns]),('dynamic',ids>=ns,ids[ids>=ns]-ns)]:
            z=payload[role];idx=torch.as_tensor(indices,dtype=torch.long);omega=2*np.pi/float(z[16])
            m=z[1][idx]+z[9][idx]*torch.sin((t-z[7][idx])*omega)/omega
            model[take]=m.numpy();sigma[take]=z[4][idx].exp().numpy();q[take]=torch.nn.functional.normalize(z[5][idx],dim=1).numpy()
        world=model.astype(np.float64)@inv[:3,:3].T+inv[:3,3]
        rotation_world=np.einsum('ij,njk->nik',inv[:3,:3],qmatrix(q))
        cov=np.einsum('nik,nk,njk->nij',rotation_world,np.square(sigma.astype(np.float64)),rotation_world)
        boxes=obs['boxes'];target=next(b for b in boxes if b['track']==tracks['a9']);a20=next(b for b in boxes if b['track']==tracks['a20'])
        a9_center,a9_envelope=box_test(world,cov,target);a20_center,a20_envelope=box_test(world,cov,a20)
        retained_center=np.zeros(len(ids),bool);retained_envelope=np.zeros(len(ids),bool)
        for box in boxes:
            if box['track']==tracks['a9']:continue
            c,e=box_test(world,cov,box,margin=.2);retained_center|=c;retained_envelope|=e
        distance,near=tree.query(world[:,:2]);normal=normals[near].astype(np.float64);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        height=np.einsum('ni,ni->n',world-cells[near],normal);height_std=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',normal,cov,normal),0))
        sigma_xy=np.sqrt(np.maximum(np.linalg.eigvalsh(cov[:,:2,:2])[:,-1],0))
        center_in_patch=distance<=np.sqrt(2)*.02/2+.002
        envelope_near_patch=distance<=3*sigma_xy+np.sqrt(2)*.02/2+.002
        blueweight=np.zeros(len(ids),np.float64);blueweight[bindex]=blue['ROI_alphaT_mass'][:,j,5]
        flags=dict(A9_center_inside_same_time_GT=a9_center,A9_3sigma_projection_envelope_overlaps_GT=a9_envelope,A20_center_inside_same_time_GT=a20_center,A20_3sigma_projection_envelope_overlaps_GT=a20_envelope,
            retained_GT_center_inside_20cm_margin=retained_center,retained_GT_3sigma_projection_envelope_overlap_20cm_margin=retained_envelope,
            center_XY_in_actual_new_patch_cells=center_in_patch,three_sigma_XY_envelope_near_new_patch=envelope_near_patch,
            native_center_within_5cm_of_candidate_road=abs(height)<.05,native_3sigma_normal_interval_touches_candidate_road=abs(height)<=3*height_std+.05)
        per={k:dict(rows=int(v.sum()),blue_alphaT_mass=float(blueweight[v].sum()),noncolour_proposal_rows=int(v[pindex].sum())) for k,v in flags.items()}
        combos=dict(A9_center_far_retained_center=a9_center&~retained_center,A9_center_far_retained_3sigma=a9_center&~retained_envelope,A9_envelope_far_retained_3sigma=a9_envelope&~retained_envelope,
            outside_new_patch_center=~center_in_patch,outside_new_patch_3sigma=~envelope_near_patch)
        for k,v in combos.items():per[k]=dict(rows=int(v.sum()),blue_alphaT_mass=float(blueweight[v].sum()),noncolour_proposal_rows=int(v[pindex].sum()))
        parity=None;old=r/f'body_overlap_diagnostic_v1/{raw:03d}/retired_prepared_geometry.npz'
        if old.exists():
            actual=bank(old);common,ia,ib=np.intersect1d(ids,actual['global_row_ids'],return_indices=True)
            parity=dict(rows=len(common),means_model_max_abs=float(abs(model[ia]-actual['means_model'][ib]).max()),scales_model_max_abs=float(abs(sigma[ia]-actual['scales_model'][ib]).max()),rotations_model_max_abs=float(abs(q[ia]-actual['rotations_model'][ib]).max()),world_locator_max_abs_m=float(abs(world[ia]-actual['means_world_locator_only'][ib]).max()))
            assert parity['means_model_max_abs']<2e-6 and parity['scales_model_max_abs']<2e-6
        path=a.output/f'{raw:03d}_world_locator.npz'
        np.savez_compressed(path,global_row_ids=ids,prepared_means_model=model,prepared_means_world=world,prepared_scales_model=sigma,prepared_rotations_model=q,covariance_world=cov,
            nearest_new_patch_XY_distance_m=distance,relative_candidate_road_height_m=height,normal_sigma_m=height_std,blue_alphaT_mass=blueweight,**flags,**combos)
        locator_pins.append(pin(path))
        stats.append(dict(raw=raw,blue_total_alphaT=float(blueweight.sum()),partitions=per,blue_weighted_world_XYZ_quantiles=[quantile(world[:,k],blueweight) for k in range(3)],
            blue_weighted_nearest_patch_XY_distance_quantiles_m=quantile(distance,blueweight),blue_weighted_road_height_quantiles_m=quantile(height,blueweight),CPU_prepared_vs_actual_GPU_packet=parity))
        gt.append(dict(raw=raw,A9=dict(track=target['track'],dimensions=target['dimensions'],world_from_actor=target['world_from_actor']),A20=dict(track=a20['track'],dimensions=a20['dimensions'],world_from_actor=a20['world_from_actor'])))
    out=dict(schema='R10.native_world_GT_OBB_CPU_diagnostic.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,frames=frames,rows=len(ids),static_rows_in_scope=int((ids<ns).sum()),dynamic_rows_in_scope=int((ids>=ns).sum()),
        blue_evidence=pin(r/'audit/blue_row_eligibility_cpu_v1/blue_row_evidence.npz'),noncolour_policy=pin(r/'ground_unowned_policy_v1/scope.json'),native_source_pins=source,source_hashes_inherited_from_immutable_parent_proof=True,
        transform=pin(matrix_path),registry=pin(registry_path),observations=[pin(x) for x in obs_paths],candidate_geometry=pin(geometry_path),
        gt_boxes=gt,statistics=stats,locators=locator_pins,
        interpretation='Same-time GT OBB semantics and native prepared means/covariance. Blue is attribution weight only, never proposal eligibility. Three-sigma projected-interval overlap is a conservative envelope, not an exact intersection test. Inferred-road height is diagnostic, not dense ground truth. Other retained GT boxes use a fixed 20cm safety margin.',
        geometric_rule='Any same-time A9 OBB center/support membership, with retained actor box exclusion; combine with immutable semantic/oldguard and measured row ledger before a separately reviewed zero-step counterfactual. No whole-owner deletion and no production gate generated here.',
        current_native_identity_gate_not_modified=True,actual_zero_step_or_fit_authorized=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),rows=len(ids),seconds=out['seconds'],first2=stats[:2]),ensure_ascii=False))


if __name__=='__main__':main()
