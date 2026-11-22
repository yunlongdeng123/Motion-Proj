"""Actual owner-row evidence and a zero-change same-instance grouping proposal.

No GPU, repeated measurement, deletion, opacity gate, optimizer or asset write.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8');M=R/'owner_measurement_A0_v1';OUT=R/'audit/owner_unification_CPU_v1'
FRAMES=[86,110];ROI='strictV_iso95_jointbelow95'
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stats(x):
    x=np.asarray(x);return dict(n=int(x.size),min=float(x.min()) if x.size else None,median=float(np.median(x)) if x.size else None,p95=float(np.percentile(x,95)) if x.size else None,max=float(x.max()) if x.size else None)
def bbox(x):return dict(min=x.min(0).tolist(),max=x.max(0).tolist()) if len(x) else None
def alpha_png(path):return np.asarray(Image.open(path).convert('L'),np.float64)/255.
def group_mass(receipt,name,key=ROI):return next(x['ROI_mass'][key] for x in receipt['feature_groups'] if x['group']==name)


def main():
    OUT.mkdir(parents=True,exist_ok=False);cfg=read(R/'baseline_contract.json');proof=read(cfg['scene_proof']);ns=proof['topology']['static_rows']
    semantic=load(proof['ownership']['path']);plan=read(cfg['context_plan']);preserved=np.asarray(plan['preserved_native_context_global_row_ids'])
    packets={};receipts={};byframe={};selected_owner=cfg['actors'][0]['owner_id']
    for raw in FRAMES:
        receipt=read(M/f'{raw:03d}/receipt.json');path=M/f'{raw:03d}/actual_contributing_rows.npz';assert sha(path)==receipt['row_packet_SHA']
        a=load(path);packets[raw]=a;receipts[raw]=receipt
        sem=np.full(len(a['branch']),-99,np.int32);static=a['branch']==0;dynamic=a['branch']==1
        sem[static]=semantic['static_actor_ids'][a['native_global_id'][static]]
        sem[dynamic]=semantic['dynamic_actor_ids'][a['native_global_id'][dynamic]-ns]
        groups={
            'old_native_dynamic_owner21':dynamic&(a['semantic_owner']==selected_owner),
            'dynamic21_explicit_semantic21':dynamic&(a['semantic_owner']==selected_owner)&(sem==selected_owner),
            'dynamic21_semantic0_source_fallback':dynamic&(a['semantic_owner']==selected_owner)&(sem==0),
            'old_native_static_owner21':static&(a['semantic_owner']==selected_owner),
            'static_unowned_context':static&(a['semantic_owner']<=0),
            'selected_canonical_field':a['branch']==3}
        a['groups']=groups;a['raw_semantic_id']=sem
        records={}
        for name,mask in groups.items():
            g=a['ROI_mass'].astype(np.float64);v=mask&(g[:,0]>0);touch=v&(g[:,2]>0);clear=v&(g[:,2]==0)
            records[name]=dict(actual_trueV_contributing_rows=int(v.sum()),trueV_mass=float(g[v,0].sum()),proxy_mass=float(g[mask,1].sum()),
                Ktouch_rows_same_view=int(touch.sum()),Ktouch_trueV_mass=float(g[touch,0].sum()),Kzero_rows_same_view=int(clear.sum()),Kzero_trueV_mass=float(g[clear,0].sum()),
                Kmass_on_saved_V_contributing_rows=float(g[v,2].sum()),Kmass_distribution_on_trueV_rows=stats(g[v,2]),
                old_K_preserved_trueV_rows=int((v&np.isin(a['native_global_id'],preserved)).sum()),
                world_bbox_of_trueV_contributors=bbox(a['means_world'][v]))
        byframe[raw]=dict(groups=records)
    # Only IDs PRESENT in both saved V/proxy contributor packets have two actual K measurements.
    old_ids={raw:a['native_global_id'][(a['branch']<2)&(a['semantic_owner']==selected_owner)] for raw,a in packets.items()}
    present_both=np.intersect1d(old_ids[86],old_ids[110]);known_touch=[];perframe_maps={}
    for raw,a in packets.items():
        take=(a['branch']<2)&(a['semantic_owner']==selected_owner)
        perframe_maps[raw]={int(g):float(k) for g,k in zip(a['native_global_id'][take],a['ROI_mass'][take,2])}
        known_touch.extend(a['native_global_id'][take&(a['ROI_mass'][:,2]>0)].tolist())
    known_touch=np.unique(known_touch).astype(np.int64)
    two_zero=np.asarray([g for g in present_both if all(perframe_maps[raw][int(g)]==0 for raw in FRAMES)],np.int64)
    conservative=np.setdiff1d(two_zero,preserved)
    union_ids=np.union1d(old_ids[86],old_ids[110]);unknown_other=np.setdiff1d(union_ids,present_both)
    np.savez_compressed(OUT/'same_instance_native_candidates.npz',both_measured_Kzero_gids=two_zero,
        both_measured_Kzero_and_not_old_preserved_gids=conservative,known_Ktouch_gids=known_touch,
        K_unknown_other_view_gids=unknown_other,old_K_preserved_gids=preserved)
    panels=[]
    for raw,a in packets.items():
        folder=OUT/f'{raw:03d}';folder.mkdir();receipt=receipts[raw]
        low_pixels=next(x['pixels'] for x in receipt['regions'] if x['ROI']==ROI)
        canonical=group_mass(receipt,'selected_actor_field');dynamic=group_mass(receipt,f'native_dynamic_owner{selected_owner}');static=group_mass(receipt,'native_static_selected_owner')
        unified=canonical+dynamic+static;total=next(x['joint_alpha_mass'] for x in receipt['regions'] if x['ROI']==ROI)
        j=load(R/f'baseline_A0_five_views/{raw:03d}/joint_render.npz');c=load(R/f'baseline_A0_five_views/{raw:03d}/component_render.npz')
        masks=load(R/f'audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz');cache=load(R.parent/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz')
        d=alpha_png(M/f'{raw:03d}/native_dynamic_owner{selected_owner}_alphaT.png');s=alpha_png(M/f'{raw:03d}/native_static_selected_owner_alphaT.png')
        estimate=j['actor_alphaT']+d+s;eps=.5/255
        lower=j['actor_alphaT']+np.maximum(0,d-eps)+np.maximum(0,s-eps)
        upper=np.minimum(j['alpha']+2e-6,j['actor_alphaT']+np.minimum(1+2e-6,d+eps+2e-6)+np.minimum(1+2e-6,s+eps+2e-6))
        V=masks['actor_visible'].astype(bool);low=V&(c['actor_alpha'].squeeze()>=.95)&(j['actor_alphaT']<.95)
        byframe[raw].update(exact_unified_owner_integral=dict(low_owner_trueV_pixels=low_pixels,canonical_mass=canonical,old_dynamic_same_owner_mass=dynamic,
            old_static_same_owner_mass=static,unified_actor_mass=unified,unified_mean_alphaT=unified/low_pixels,
            unified_fraction_of_total_scene_alpha_mass=unified/total,RGBA_changed=False),
            unified_alpha95_quantization_bounds=dict(scope_pixels=int(V.sum()),definitely_below95=int((V&(upper<.95)).sum()),definitely_above95=int((V&(lower>=.95)).sum()),
                uncertain_at95=int((V&(lower<.95)&(upper>=.95)).sum()),source='canonical float + two PNG8 measured feature layers; threshold bounds, not exact per-pixel float'),
            two_view_K_candidates_trueV_mass=float(a['ROI_mass'][np.isin(a['native_global_id'],conservative),0].astype(float).sum()),
            trueV_mass_of_K_unknown_other_view=float(a['ROI_mass'][np.isin(a['native_global_id'],unknown_other),0].astype(float).sum()))
        # Actual per-row native center markers; colors show measured K status, never opacity edits.
        photo=np.rint(np.clip(cache['photo_pinhole_srgb'],0,1)*255).astype(np.uint8);marker=Image.fromarray(photo);draw=ImageDraw.Draw(marker)
        old=(a['branch']<2)&(a['semantic_owner']==selected_owner)&(a['ROI_mass'][:,0]>0)
        xyz=a['means_world'][old].astype(float);cam=xyz@cache['world_to_camera'][:3,:3].T+cache['world_to_camera'][:3,3]
        q=cam@cache['camera_K'].T;uv=q[:,:2]/q[:,2,None]-.5;gids=a['native_global_id'][old];mass=a['ROI_mass'][old,0];Kmass=a['ROI_mass'][old,2]
        order=np.argsort(mass)
        for index in order:
            x,y=uv[index];g=gids[index]
            if not np.isfinite([x,y]).all() or cam[index,2]<=0 or not(0<=x<photo.shape[1] and 0<=y<photo.shape[0]):continue
            color=(55,210,135) if g in conservative else ((240,70,90) if Kmass[index]>0 else (245,180,60))
            radius=1 if mass[index]<.05 else 2;draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
        marker.save(folder/'native_same_instance_center_markers.png')
        heat=np.stack([d,np.zeros_like(d),1-d],axis=-1);heat[~V]*=.25;Image.fromarray(np.rint(np.clip(heat,0,1)*255).astype(np.uint8)).save(folder/'old_dynamic_owner21_contribution.png')
        show=np.stack([np.clip(estimate,0,1),np.clip(estimate,0,1),np.clip(estimate,0,1)],-1);show[~V]*=.15
        Image.fromarray(np.rint(show*255).astype(np.uint8)).save(folder/'unified_owner_alphaT_PNG_estimate.png')
        ys,xs=np.nonzero(V);box=(max(int(xs.min())-18,0),max(int(ys.min())-18,0),min(int(xs.max())+19,photo.shape[1]),min(int(ys.max())+19,photo.shape[0]))
        panels.append((raw,box,[Image.fromarray(photo),Image.fromarray(np.rint(np.clip(j['rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)),marker,Image.open(folder/'old_dynamic_owner21_contribution.png'),Image.open(folder/'unified_owner_alphaT_PNG_estimate.png')]))
    sheet=Image.new('RGB',(5*360,2*300),(15,23,34));draw=ImageDraw.Draw(sheet)
    labels=['correct pinhole photo','actual shared RGB','native same-owner centers: green 2viewK0/red touch/amber unknown','old dynamic owner21 alphaT','unified alphaT estimate (PNG bound)']
    for row,(raw,box,images) in enumerate(panels):
        for col,(title,im) in enumerate(zip(labels,images)):
            crop=im.crop(box);crop.thumbnail((360,265));sheet.paste(crop,(col*360,row*300+30));draw.text((col*360+4,row*300+5),f'{raw:03d} {title}',fill=(230,238,247))
    sheet.save(OUT/'same_instance_owner_evidence.png')
    result=dict(schema='R8.actual_same_instance_owner_unification_CPU.v1',complete=True,CPU_only=True,new_nativeF=0,new_nativeB=0,parameter_updates=0,
        actual_measurement_source=str(M),actual_measurement_F=2,actual_measurement_B=2,frame_results=byframe,
        K_scope=dict(native_same_owner_union_rows=len(union_ids),present_in_both_saved_packets=len(present_both),known_Ktouch_union_rows=len(known_touch),
            two_measured_view_Kzero_rows=len(two_zero),two_view_Kzero_and_not_old_preserved_rows=len(conservative),unknown_other_view_K_rows=len(unknown_other),
            missing_K_rows_are_not_zero=True,all_five_views_K_protection_pending=True),
        CPU_v1_inventory_correction='v44 factory semantic zero policy is source_fallback. Previous raw semantic owner21 topology count6517 excludes semantic0 dynamic rows restored to source actor21. Actual measurement owners/gids are authoritative; no change to raster/VJP results.',
        minimal_proposal='Readout/ownership first: aggregate selected canonical field plus retained native semantic/source owner21 as one actor identity, keeping each current provider center/time/trajectory unchanged. Keep static unowned separate. Any later local color/opacity redistribution needs R9 acceptance, all-view K measurement and semantic/visibility review; no whole-owner deletion.',
        optimization_not_executed=True,R9_source_acceptance_pending=True)
    write(OUT/'owner_unification_audit.json',result)
    print(json.dumps(dict(frames={raw:byframe[raw]['exact_unified_owner_integral'] for raw in FRAMES},K_scope=result['K_scope'],groups={raw:byframe[raw]['groups'] for raw in FRAMES}),ensure_ascii=False))


if __name__=='__main__':main()
