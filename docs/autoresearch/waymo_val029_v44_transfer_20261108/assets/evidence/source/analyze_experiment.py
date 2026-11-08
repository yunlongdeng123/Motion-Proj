"""Measured-region metrics and explicit evidence limits; never decides visual acceptance."""
from pathlib import Path
import json,csv
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs')
def read(p):return json.loads(p.read_text())
def aggregate(rows,key):
    values=[(r['raw'],r['regions'][key]) for r in rows if r['regions'][key]['pixels']>0]
    result=dict(frames=len(values),pixels=sum(v['pixels'] for _,v in values))
    for field in ['native_keep_gt_mae','keep_gt_mae','delete_vs_keep_mae','linear_proxy_mae']:
        vals=[(r,v[field],v['pixels']) for r,v in values if v.get(field) is not None];den=sum(n for r,x,n in vals)
        if vals:result[field]=dict(pixel_weighted=sum(x*n for r,x,n in vals)/den,frame_mean=float(np.mean([x for r,x,n in vals])),worst_raw=max(vals,key=lambda t:t[1])[0],worst=max(x for r,x,n in vals))
    compare=[(r,v) for r,v in values if v.get('native_keep_gt_mae') is not None]
    if compare:result.update(frames_improved_vs_native=sum(v['keep_gt_mae']<v['native_keep_gt_mae']-1e-8 for r,v in compare),frames_worse_vs_native=sum(v['keep_gt_mae']>v['native_keep_gt_mae']+1e-8 for r,v in compare))
    return result
def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=False);stages={'semantic':'semantic_all_eval','asset':'semantic_asset_all_r2','final':'final_all_eval'};summaries={k:read(ROOT/v/'summary.json') for k,v in stages.items()};final=summaries['final']['frames'];stats={}
    for name,s in summaries.items():
        rows=s['frames'];stats[name]=dict(frames=len(rows),keep_gt_frame_mean=float(np.mean([r['keep_gt_mae'] for r in rows])),regions={key:aggregate(rows,key) for key in rows[0]['regions']},group_conservation_maxabs=max(max(r['keep_group_conservation_maxabs'],r['delete_group_conservation_maxabs']) for r in rows),keep_again_exact=all(r['keep_again_exact'] for r in rows),delete_direct_target_max=max(r['target_delete_contribution_max'] for r in rows))
    # Same-domain road proxy comparisons; inferred area is never independent GT.
    road=[]
    for row in final:
        raw=row['raw'];d=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');item=dict(raw=raw)
        for stage,folder in [('native','formal10k_eval'),*stages.items()]:
            actual=stages.get(stage,folder);rgb=np.load(ROOT/actual/f'{raw:03d}/delete.npz')['rgb'];linear=np.where(rgb<=.04045,rgb/12.92,((np.clip(rgb,0,1)+.055)/1.055)**2.4)
            item[stage]={k:dict(pixels=int(d[k].sum()),mae=float(abs(linear-d['reference_linear'])[:,d[k]].mean()) if d[k].any() else None) for k in ['real_domain','inferred_domain']}
        road.append(item)
    road_aggregate={stage:{domain:dict(pixels=sum(r[stage][domain]['pixels'] for r in road),mae=sum((r[stage][domain]['mae'] or 0)*r[stage][domain]['pixels'] for r in road)/max(1,sum(r[stage][domain]['pixels'] for r in road))) for domain in ['real_domain','inferred_domain']} for stage in ['native','semantic','asset','final']}
    # Sparse same-physical-surface anchors observed at raw190, queried at raw155.
    correspondence=read(BANK.parent/'verified_correspondences.json');points=np.array([r['actor_local_xyz'] for r in correspondence['pairs']]);d=read(BANK/'observations/155.json');body=next(b for b in d['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');pose=np.array(body['world_from_actor']).reshape(4,4);world=points@pose[:3,:3].T+pose[:3,3];cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (BANK/'calib/0029155.txt').read_text().splitlines()};C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);C=C@np.linalg.inv(np.array(d['ego_world']).reshape(4,4));cam=world@C[:3,:3].T+C[:3,3];K=cal['P0'].reshape(3,4)[:,:3]/4;K[2,2]=1;uv=(cam@K.T)[:,:2]/cam[:,2:]- .5;pix=np.floor(uv+.5).astype(int);assert (pix>=[0,0]).all() and (pix<[480,320]).all();T=np.loadtxt(ROOT/'inputs/native/transform_matrix.txt');scale=np.linalg.det(T[:3,:3])**(1/3);anchors={}
    for stage,folder in [('native','formal10k_eval'),*stages.items()]:
        actual=stages.get(stage,folder);pack=np.load(ROOT/actual/'155/delete.npz');alpha=pack['alpha'][0,pix[:,1],pix[:,0]];depth=pack['depth_sum'][0,pix[:,1],pix[:,0]]/np.maximum(alpha,1e-5)/scale;err=abs(depth-cam[:,2]);groups=pack['group_alpha'] if 'group_alpha' in pack else None;anchors[stage]=dict(count=len(points),unique_render_pixels=len(np.unique(pix,axis=0)),render_depth_mae_m=float(err.mean()),render_depth_median_m=float(np.median(err)),render_alpha_mean=float(alpha.mean()),white_suv_contribution_mean=float(groups[1,pix[:,1],pix[:,0]].mean()) if groups is not None else None)
    held=read(ROOT/'final_heldout_eval/summary.json');nativeheld=read(ROOT/'native_heldout_eval/summary.json');heldroles={role:sum(any(b['track']==track and b.get('official_FRONT_visible',False) for b in read(ROOT/'heldout/observations'/f'{raw:03d}.json')['boxes']) for raw in range(79,119)) for role,track in [('red','lvxXvuSB3kueTyP2ignh7w'),('white','DwjZr_bYUSZOr-XDGregnA'),('gray','AyRpKRmelCT4A709Lo8pvA')]}
    native_spec=read(ROOT/'formal10k/effective_config.json')['followup_scene_spec'];native_raws={int(s[-3:]) for s in native_spec['frame_ids']};assert len(native_raws)==158 and not native_raws.intersection(range(79,119));assert not set(read(ROOT/'road_gates/receipt.json')['fit_frames']).intersection(range(79,119));prior=np.load(ROOT/'revealed_prior/observations.npz');assert not np.isin(prior['sample_raw_frame'],np.arange(79,119)).any()
    absent=[r for r in final if r['regions']['target_red']['pixels']==0];result=dict(status='COMPLETE_MEASURED_EVALUATION_NOT_VISUAL_ACCEPTANCE',stages=stats,road_proxy=road_aggregate,sparse_reveal_anchors=anchors,sparse_anchor_limit='BUILD source190 points; expected alpha-weighted mixed rendering depth, 480x320 pinhole camera; not dense RGB GT or independent holdout',heldout=dict(frames=40,raws=[79,118],native_keep_gt_mae=float(np.mean([r['keep_gt_mae'] for r in nativeheld['frames']])),final_keep_gt_mae=float(np.mean([r['keep_gt_mae'] for r in held['frames']])),official_front_visible_frames=heldroles),target_mask_absent=dict(frames=len(absent),frames_with_any_edit_change=sum(r['keep_delete_changed_pixels']>0 for r in absent),max_changed_pixels=max(r['keep_delete_changed_pixels'] for r in absent)),leakage_audit='158 BUILD fit frames; 40 held-out RGB decoded only for evaluation; no held-out LiDAR decoded',quality_accepted=False,human_verdict=None,method_transfer='new scene-native 10K + shared ownership + new PVD/C2/material/SurfaceLOD + automatic road chart + inherited static gate objective; automatic chart adapters differ from old manual scene008 preprocessing')
    (out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'road_per_frame.json').write_text(json.dumps(road,indent=2));flat=[]
    for stage,s in summaries.items():
        for r in s['frames']:
            for region,v in r['regions'].items():flat.append(dict(stage=stage,raw=r['raw'],region=region,**v))
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'per_frame_regions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(flat)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
