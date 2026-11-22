"""Readback-only source qualification, immutable-parent and saved-surface gates.

No renderer, source search, fuser, diffusion, Gaussian or GPU is executed.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats
from audit_local_shared_source_cpu import nf,sample,all4
from atlas_canvas_cpu import embed,minimal_vertex_scope

FIELD='45b4b9d752f1584019e6687008de489de071a81907ab2c3c1b61b20964f64d32'


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10';R7=C/'road_shared_appearance_20261122_r7';R9=C/'old_raw_teacher_joint_20261123_r9'
    fp=R/'timewindow17_shared_requalified_v2/final_atlas.npz';recpath=fp.parent/'receipt.json';rec=read(recpath);assert pin(fp)['sha256']==FIELD
    for key in ['parent','seed','geometry','radiometry_state','solve_state','source_photo_seed_lineage','source_photo_seed_trace']:
        assert pin(Path(rec[key]['path']))['sha256']==rec[key]['sha256'],key
    f=load(fp);seed=load(Path(rec['seed']['path']));g=load(Path(rec['geometry']['path']));state=load(fp.parent/'radiometry_state.npz');solve=load(fp.parent/'solve_state.npz')
    shape=f['permission_mask'].shape;parentraw=load(Path(rec['parent']['path']));parent=embed(parentraw,shape,parentraw['permission_mask'].shape)
    mutable=f['parent_conditioned_extension'];repair=f['qualification_repair_mask'];extent=f['new_material_extent'];known=f['real_locks'];fresh=f['fresh_recertified_real_mask']
    contract=dict(schema='R10.timewindow17_independent_shared_input_contract.v1',complete=True,final_atlas_sha256=FIELD,
        scope='Only qualified shared colour input for limited zero-step joint comparison',
        immutable='Parent RGB/metadata outside extent union75 repair and6332 unaffected real locks byteexact',
        qualification='75 ancestor real certificates withdrawn; whole-asset17-source conservative proxy search retained, no fresh certification is not physical-invisibility GT',
        fixed_overlap='All50251 common qualified texels /150753 RGB channels before influence or robust selection',
        frequency='same sigma6texel=.12m LF; separate complete-photo seed HF correlation/amplitude/delta at2cm/4cm',
        seam='parent/mutable boundary p95 descriptive and compared with seed; no new .005 hard gate or solver tuning',
        saved_surface='All5 original1152 PNGs + saved per-pixelNPZ; no rerender/search/fuser replay',GPU_calls=0)
    write(out/'acceptance_contract.json',contract)
    assert shape==(287,2157) and int(mutable.sum())==46911 and int(repair.sum())==75 and int(extent.sum())==46836
    assert np.array_equal(mutable,extent|repair) and not (extent&repair).any()
    assert np.array_equal(known,(parent['real_locks']&~repair)|fresh) and int(known.sum())==6332 and not fresh.any()
    assert np.array_equal(f['inherited_unaffected_real_mask'],known) and parent['real_locks'][repair].all()
    assert not known[repair].any() and np.isin(f['source_kind'][repair],[2,3]).all() and (seed['source_kind'][repair]==2).all()
    assert np.array_equal(f['permission_mask'],parent['permission_mask']|extent)
    assert np.array_equal(extent,f['local_geometry_scope']&~parent['permission_mask'])
    for bank in [f,seed]:
        assert bank['atlas_linear'][~mutable].tobytes()==parent['atlas_linear'][~mutable].tobytes()
        assert bank['atlas_linear'][known].tobytes()==parent['atlas_linear'][known].tobytes()
    for key in ['source_kind','seed_source_kind','teacher_adopted','effective_teacher_blend','teacher_support_count','confidence']:
        assert f[key][~mutable].tobytes()==parent[key][~mutable].tobytes(),key
    true=load(R7/'quilt_seed/atlas_seed.npz');true=embed(true,shape,true['same_world_lock'].shape)
    assert np.array_equal(known,true['same_world_lock']&~repair)
    assert f['atlas_linear'][known].tobytes()==true['atlas_seed_linear'][known].tobytes()
    assert np.isfinite(f['atlas_linear']).all() and f['atlas_linear'].min()>=0 and f['atlas_linear'].max()<=1
    scope=minimal_vertex_scope(g['cell_asset_mask']);yy,xx=g['uv_global'][...,1],g['uv_global'][...,0];global_scope=np.zeros(shape,bool);global_scope[yy[scope],xx[scope]]=True
    assert np.array_equal(global_scope,f['local_geometry_scope']) and np.array_equal(global_scope,state['overwritten_geometry_scope'])
    assert not (mutable&~global_scope).any() and not (known&global_scope).any()
    lineage=load(Path(rec['source_photo_seed_lineage']['path']));trace=read(Path(rec['source_photo_seed_trace']['path']))
    assert np.array_equal(lineage['extension_mask'],mutable) and np.array_equal(lineage['qualification_repair_mask'],repair)
    weights=lineage['mixture_tile_weight'][mutable];ids=lineage['mixture_tile_id'][mutable];active=weights>0
    assert np.isfinite(weights).all() and (weights>=0).all() and abs(weights.sum(-1)-1).max()<2e-6
    assert (ids[active]>=0).all() and (ids[active]<len(trace['operations'])).all()
    assert np.isin(lineage['source_raw_primary'][mutable],[110,137]).all() and np.isfinite(lineage['source_uv_Brown_primary'][mutable]).all()
    assert all(x['provenance_kind']=='similar_reference_full_texture_inferred_not_same_world' for x in trace['operations'])
    fresh75=load(Path(rec['fresh_qualification_source']['path']));assert not fresh75['fresh_known'].any()
    quv=fresh75['global_uv'];assert repair[quv[:,1],quv[:,0]].all() and len(quv)==75
    search=R/'audit/timewindow17_asset_real_search_v1';searchrec=read(search/'receipt.json');partition=read(search/'fixed_rejection_partition.json')
    freshall=load(search/'fresh_qualification_asset.npz');assert not freshall['fresh_known'].any() and not freshall['all_candidate_qualified_proxy'].any()
    assert np.array_equal(freshall['global_uv'],g['uv_global'][scope]) and np.array_equal(freshall['new_world_xyz'],g['world_xyz'][scope])
    assert len(quv)==75 and len(freshall['global_uv'])==176465 and searchrec['candidate_records']==2999905
    names=searchrec['reason_bit_names'];priority=partition['exclusive_priority'];totals={x:0 for x in priority};nonexclusive={x:0 for x in names};qualified=0
    for raw in searchrec['examined_frames']:
        z=load(search/f'{raw:03d}_all_candidates.npz');bits=z['rejection_bits'];assert len(bits)==176465
        assert np.array_equal(z['qualified_proxy_certificate'],bits==0)
        remaining=np.ones(len(bits),bool)
        for name in names:nonexclusive[name]+=int(((bits&(1<<names.index(name)))!=0).sum())
        for name in priority:
            reject=remaining&((bits&(1<<names.index(name)))!=0);totals[name]+=int(reject.sum());remaining&=~reject
        qualified+=int(remaining.sum())
    assert totals==partition['exclusive_totals'] and sum(totals.values())+qualified==2999905 and qualified==0
    assert not freshall['strict_shadow_GT_verified'].any()
    observations=Path(rec['new_observation_receipt']['path']).parent;before=[];valids=[];replays=[]
    import cv2
    for i,raw in enumerate(state['frames']):
        raw=int(raw);obpath=observations/f'{raw:03d}/world_observation.npz';ob=load(obpath);m=ob['valid_mask']&scope;before.append(ob['canonical_linear_RGB'].astype(float));valids.append(m)
        assert state['raw_RGB'][i,yy[m],xx[m]].tobytes()==ob['canonical_linear_RGB'][m].tobytes()
        assert np.array_equal(state['valid'][i,yy[scope],xx[scope]],m[scope])
        vr=load(R7/f'projection_cache_v1/{raw:03d}/view_cache.npz');E=vr['world_to_camera'];intr=vr['Brown_intrinsics'];K=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
        rv,_=cv2.Rodrigues(E[:3,:3]);uv,_=cv2.projectPoints(g['world_xyz'].reshape(-1,3),rv,E[:3,3],K,intr[4:]);brown=uv.reshape(*scope.shape,2)*[.6,.6]-.5
        obrec=read(observations/f'{raw:03d}/receipt.json');rp=Path(obrec['raw_model']['path']);assert rp.name=='raw_model.png' and pin(rp)['sha256']==obrec['raw_model']['sha256']
        encoded=np.asarray(Image.open(rp).convert('RGB'),float)/255;linear=np.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
        sampled,bounds=sample(linear,brown);sampled*=obrec['exposure_gain'];error=float(abs(sampled[m]-ob['canonical_linear_RGB'][m]).max());assert error<2e-6
        replays.append(dict(raw=raw,source=pin(rp),RGB_replay_max_abs_linear=error,valid_texels=int(m.sum()),original_raw_not_hard_final=True))
    before=np.asarray(before);valids=np.asarray(valids);after=np.zeros((len(before),*scope.shape,3),np.float64)
    after[:,scope]=state['corrected_RGB'][:,yy[scope],xx[scope]]
    pair=valids[0]&valids[1];assert pair.sum()==50251
    bands={}
    for name,sigma in [('full_RGB',None),('LF_sigma0p12m',6.)]:
        a0=before if sigma is None else np.array([nf(x,m,sigma) for x,m in zip(before,valids)])
        a1=after if sigma is None else np.array([nf(x,m,sigma) for x,m in zip(after,valids)])
        v0=float(abs(a0[0]-a0[1])[pair].mean());v1=float(abs(a1[0]-a1[1])[pair].mean());bands[name]=dict(before_MAE=v0,after_MAE=v1,reduction=1-v1/v0,pair_texels=int(pair.sum()),channel_samples=int(pair.sum()*3))
        assert 1-v1/v0>=.2
    grain={}
    for metres in [.02,.04]:
        h0=(seed['atlas_linear']-nf(seed['atlas_linear'],mutable,metres/.02))[mutable].ravel();h1=(f['atlas_linear']-nf(f['atlas_linear'],mutable,metres/.02))[mutable].ravel();rms=np.sqrt(np.mean(h0*h0))
        grain[str(metres)]=dict(HF_correlation=float(np.corrcoef(h0,h1)[0,1]),RMS_ratio=float(np.sqrt(np.mean(h1*h1))/rms),HF_delta_RMS_over_seed=float(np.sqrt(np.mean((h1-h0)**2))/rms))
        assert grain[str(metres)]['HF_correlation']>=.99 and .95<=grain[str(metres)]['RMS_ratio']<=1.05
    parent_lock=parent['permission_mask']&~repair
    def boundary(image):
        a0=nf(parent['atlas_linear'],parent_lock,6);a1=nf(image,mutable,6);values=[]
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
            m=mutable&np.roll(parent_lock,(-dy,-dx),(0,1))
            if dy>0:m[-1]=False
            if dy<0:m[0]=False
            if dx>0:m[:,-1]=False
            if dx<0:m[:,0]=False
            values.append(abs(a1-np.roll(a0,(-dy,-dx),(0,1)))[m])
        return stats(np.concatenate(values).ravel())
    seam=dict(seed_LF=boundary(seed['atlas_linear']),final_LF=boundary(f['atlas_linear']))
    reconstructed=(seed['atlas_linear'].astype(float)+solve['RGB_residual'].astype(float)).astype(np.float32);reconstructed[~mutable]=seed['atlas_linear'][~mutable]
    replayerror=float(abs(reconstructed-f['atlas_linear']).max());assert replayerror<5e-7
    correctionerror=float(abs(state['affine_RGB'].astype(float)+state['world_bias'].astype(float)-state['corrected_RGB']).max());assert correctionerror<5e-7
    review=R/'timewindow17_requalified_surface_review_v2_replay1';vr=read(review/'receipt.json');assert vr['final_atlas_sha256']==FIELD and vr['geometry_sha256']==rec['geometry']['sha256']
    gains={int(x['raw']):float(x['gain']) for x in read(R7/'exposure_gain.json')['records']};views=[]
    for raw in [86,110,134,179,198]:
        z=load(review/f'{raw:03d}/surface_review.npz');shown=z['shown'];assert not z['unavailable_material'].any()
        sampled,valid=sample(f['atlas_linear'],z['atlas_uv_global']);expected=sampled/gains[raw];expected=np.where(expected<=.0031308,expected*12.92,1.055*np.maximum(expected,0)**(1/2.4)-.055)
        image=np.asarray(Image.open(review/f'{raw:03d}/direct_surface_material.png'));pngerror=int(abs(np.rint(np.clip(expected[shown],0,1)*255).astype(int)-image[shown].astype(int)).max());assert pngerror<=1
        assert not (shown&(image.max(-1)<40)).any()
        N=g['cell_asset_mask'].size;triangle=z['triangle_index'];exact=z['projection_valid']&g['cell_asset_mask'].ravel()[np.maximum(triangle,0)%N]
        assert np.array_equal(z['projection_valid'],exact)
        views.append(dict(raw=raw,material_pixels=int(shown.sum()),unavailable_material=0,near_black_pixels=0,PNG_replay_max_8bit=pngerror,exact_cell_asset_scope=True,
            material=pin(review/f'{raw:03d}/direct_surface_material.png')))
    numeric=dict(schema='R10.timewindow17_independent_source_numeric_gate.v1',complete=True,decision='PASS',final_atlas_sha256=FIELD,
        geometry_sha256=rec['geometry']['sha256'],scope='Qualified shared RGB input for limited zero-step comparison only',
        parent_outside46911_mutable_RGB_and_metadata_byteexact=True,unaffected_real6332_mask_and_RGB_byteexact=True,
        repair75_current_real_certification_withdrawn=True,fresh_recertified_real_count=0,mutable_texels=46911,new_extent_texels=46836,
        full_asset_fresh_search=dict(candidate_records=2999905,asset_vertices=176465,asset_cells=175302,world_area_m2=searchrec['asset_world_surface_area_m2'],fresh_strict_certified=0,
            ordered_rejection_partition=totals,nonexclusive_rejections=nonexclusive,partition_recomputed_from_saved_bits=True,
            OBB_and_shadow_are_proxies_not_visibility_GT=True,no_fresh_certificate_does_not_prove_physical_invisibility=True,all199_photos_not_claimed_invisible=True,
            receipt=pin(search/'receipt.json'),fixed_partition=pin(search/'fixed_rejection_partition.json')),
        inferred_photo_lineage_complete=True,similar_photo_sources=[110,137],confidence_not_calibrated_probability=True,
        original_raw_source_sampling_replay=replays,fixed_pair_metrics=bands,photo_HF_retention=grain,seam_LF=seam,
        seam_p95_descriptive_not_new_0p005_hard_gate=True,joint_calibration_both_views_no_first_writer_lock=True,
        affine_bias_replay_max_abs=correctionerror,seed_plus_continuous_residual_replay_max_abs=replayerror,
        no_pointwise_visibility_alpha_RGB_multiplication=True,finite_in_range_linear_RGB=True,saved_five_view_material_checks=views,
        source_receipt=pin(recpath),surface_receipt=pin(review/'receipt.json'),whole_joint_scene_quality_accepted=False,
        renderer_search_fuser_not_reexecuted=True,GPU_forward=0,GPU_backward=0,Gaussian_updates=0,seconds=time.time()-start)
    write(out/'numeric_gate.json',numeric)
    visual=dict(schema='R10.timewindow17_independent_source_visual_gate.v1',complete=True,decision='PASS',final_atlas_sha256=FIELD,
        geometry_sha256=rec['geometry']['sha256'],scope='175302cell material input, not final native joint quality',reviewer='native_clean_photo_regions',root_concurred=False,
        frames=[86,110,134,179,198],original_resolution=[1152,768],
        observations='All five original-size direct material images inspected. Interior grain remains visible with a somewhat flat tone. No obvious black rim, macro colour seam or lattice stripe within the actual material domain. Dark outside-domain background is not road supervision.75 repaired addresses are inferred sources2/3, not certified new real observations.',
        frame_images=views,contactsheet=pin(review/'five_view_direct_surface_context.png'),numeric_gate=pin(out/'numeric_gate.json'),
        source_input_accepted_for_limited_zero_step=True,whole_restoration_or_native_occlusion_accepted=False,GPU_forward=0,GPU_backward=0,Gaussian_updates=0)
    write(out/'visual_gate.json',visual)
    print(json.dumps(dict(complete=True,numeric=pin(out/'numeric_gate.json'),visual=pin(out/'visual_gate.json'),bands=bands,grain=grain,seam=seam)),flush=True)


if __name__=='__main__':main()
