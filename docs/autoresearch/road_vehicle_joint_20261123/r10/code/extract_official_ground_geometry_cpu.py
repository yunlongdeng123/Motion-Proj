"""Original Waymo point semantics, exact TOP index alignment, diagnostic only."""
import argparse, importlib.util, json, sys, time, zlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull, cKDTree
from diagnose_nonplanar_inference_cpu import load, read, write, pin, stats, design, neighbourhood


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R10=C/'road_underbody_local_20261123_r10';strict=R10/'audit/local_surface_geometry_v1'
    oldcontract=read(strict/'geometry_qualification_contract.json');gpath=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(gpath)
    s=load(strict/'local_surface_geometry.npz');u0,v0,u1,v1=s['bounds_uvxy'].astype(int);shape=s['geometry_valid'].shape
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');receipt=read(bank/'receipt.json')
    droot=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');decoder=module('official_ground_raw_decoder',droot/'waymo_numpy_cpu.py')
    decoder.matrix=module('official_ground_fast_matrix',droot/'fast_range_matrix.py').matrix;pb=decoder.protobuf()
    from waymo_open_dataset.protos import segmentation_pb2
    segsource=Path(segmentation_pb2.__file__)
    enum={str(k):int(v) for k,v in segmentation_pb2.Segmentation.Type.items()}
    roadID=enum['TYPE_ROAD'];laneID=enum['TYPE_LANE_MARKER']
    code_contract=dict(schema='R10.official_ground_point_diagnostic_contract.v1',complete=True,
        written_before_official_points_examined=True,decision='READ_ONLY_POINT_EVIDENCE_NOT_GEOMETRY_ACCEPTANCE',
        raw_pool_train=oldcontract['train_raw'],raw_pool_heldout=oldcontract['heldout_raw'],
        selected_by_original_pool_not_point_quality=True,official_label_enum=enum,official_enum_source=pin(segsource),
        official_semantic_channels=['instance_id','semantic_class'],
        documentation_source=pin(segsource.parents[1]/'v2/perception/segmentation.py'),
        ground_classes={'road':roadID,'lane_marker':laneID},other_ground_not_silently_added=True,
        alignment='exported column6 TOP encoded=((return_index*H+row)*W+column); nonTOP=-1 remains unknown; original valid/NLZ mask and row order independently verified',
        OBB_policy='preserve every official-ground point; original OBB and expanded OBB are independent diagnostic flags, never default rejection',
        photo_policy='original Brown photo road0 and erode2 are independent diagnostic flags; camera occlusion cannot delete physical official-ground evidence',
        fit_and_activation='none in this stage; no strict/old model edit; next official geometry protocol must be separately registered',
        GPU_forward=0,updates=0)
    write(out/'point_evidence_contract.json',code_contract)
    assert pin(droot/'waymo_numpy_cpu.py')['sha256']==receipt['decoder_sha256']
    assert pin(droot/'fast_range_matrix.py')['sha256']==receipt['matrix_decoder_sha256']
    collector=module('official_ground_Brown_helper',R10/'code/qualify_local_surface_geometry_cpu.py')
    frames=set(oldcontract['train_raw']+oldcontract['heldout_raw'])
    M=g['world_from_uv_xy'][:2,:2];origin=g['world_from_uv_xy'][:2,3];iM=np.linalg.inv(M)
    query=s['world_xyz'][...,:2].reshape(-1,2);xybounds=np.asarray(oldcontract['XY_fit_context_bbox'])
    zref=float(np.median([np.asarray(read(bank/f'observations/{r:03d}.json')['ego_world']).reshape(4,4)[2,3] for r in oldcontract['train_raw']]))
    all_points=[];records=[];inputs=[];range_payload={}
    for r,payload,crc in decoder.records(receipt['raw'],max(frames)):
        if r not in frames:continue
        frame=pb.Frame.FromString(payload);obs_path=bank/f'observations/{r:03d}.json';obs=read(obs_path)
        original_path=bank/obs['lidar_path'];original=np.fromfile(original_path,dtype='<f4').reshape(-1,6)
        labels=np.full(len(original),-1,np.int16);instances=np.full(len(original),-1,np.int32)
        top=next(x for x in frame.lasers if x.name==1);returns=[];haslabels=False;aligned=True
        for ri in [0,1]:
            source=getattr(top,f'ri_return{ri+1}');image=decoder.matrix(source.range_image_compressed,pb);H,W=image.shape[:2]
            compressed=getattr(source,'segmentation_label_compressed',b'')
            rec=dict(return_index=ri,range_image_HW=[H,W],compressed_semantic_bytes=len(compressed))
            if compressed:
                mat=pb.MatrixInt32();mat.ParseFromString(zlib.decompress(compressed));seg=np.asarray(mat.data,np.int32).reshape(tuple(mat.shape.dims))
                assert seg.shape==(H,W,2);haslabels=True
                hit=(original[:,5]>=ri*H*W)&(original[:,5]<(ri+1)*H*W)
                encoded=original[hit,5].astype(np.int64);assert np.array_equal(encoded.astype(np.float32),original[hit,5])
                local=encoded-ri*H*W;labels[hit]=seg.reshape(-1,2)[local,1];instances[hit]=seg.reshape(-1,2)[local,0]
                valid=(image[...,0]>0)&(image[...,3]!=1);yy,xx=np.where(valid)
                expected=((ri*H+yy)*W+xx).astype(np.float32)
                assert np.array_equal(original[hit,5],expected)
                assert np.array_equal(labels[hit],seg[...,1][valid])
                rec['actual_valid_NLZ_row_order_and_semantic_labels_exact']=True
                rec['TOP_retained_points']=int(hit.sum());rec['official_class_counts']={str(int(k)):int(n) for k,n in zip(*np.unique(labels[hit],return_counts=True))}
                range_payload[f'{r:03d}_return{ri}_labels_instance_semantic']=seg
            returns.append(rec)
        if haslabels:
            regenerated,details=decoder.lidar(frame,pb)
            assert regenerated.shape==original.shape and regenerated.astype('<f4').tobytes()==original.tobytes()
        pose=np.asarray(obs['ego_world']).reshape(4,4);assert np.array_equal(pose,np.asarray(frame.pose.transform).reshape(4,4))
        assert frame.timestamp_micros==obs['timestamp_micros']
        world=original[:,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
        region=np.isfinite(world).all(-1)&(world[:,:2]>=xybounds[0]).all(-1)&(world[:,:2]<=xybounds[1]).all(-1)&(abs(world[:,2]-zref)<=1.5)
        ids=np.flatnonzero(region);X=world[region];lab=labels[region];inst=instances[region]
        box0=np.zeros(len(X),bool);boxwide=box0.copy()
        for box in obs['boxes']:
            if 'world_from_actor' not in box:continue
            T=np.asarray(box['world_from_actor']).reshape(4,4);delta=abs((X-T[:3,3])@T[:3,:3]);half=.5*np.asarray(box['dimensions'])
            box0|=(delta<=half).all(-1);boxwide|=(delta<=half+[.15,.15,.08]).all(-1)
        calpath=C/f'road_multisource_20261122_r1/sources/qualified_observations/{r:03d}.npz';E=load(calpath)['world_to_camera']
        uv,depth=collector.brown(X,E,np.asarray(obs['camera_intrinsic_Brown']));sem_path=bank.parent/f'semantics/{r:03d}.png';photo=np.asarray(Image.open(sem_path));road2=binary_erosion(photo==0,iterations=2)
        ix=np.rint(np.nan_to_num(uv[:,0])).astype(int);iy=np.rint(np.nan_to_num(uv[:,1])).astype(int)
        bounded=np.isfinite(uv).all(-1)&(depth>.1)&(ix>=0)&(iy>=0)&(ix<photo.shape[1])&(iy<photo.shape[0])
        plabel=np.full(len(X),255,np.uint8);proad2=np.zeros(len(X),bool);q=np.flatnonzero(bounded);plabel[q]=photo[iy[q],ix[q]];proad2[q]=road2[iy[q],ix[q]]
        uvglobal=(X[:,:2]-origin)@iM.T;uq=np.floor(uvglobal[:,0]).astype(int);vq=np.floor(uvglobal[:,1]).astype(int)
        gh,gw=g['geometry_valid'].shape;inbounds=(uq>=0)&(vq>=0)&(uq+1<gw)&(vq+1<gh);uq=np.clip(uq,0,gw-2);vq=np.clip(vq,0,gh-2)
        parent4=inbounds&g['geometry_valid'][vq,uq]&g['geometry_valid'][vq,uq+1]&g['geometry_valid'][vq+1,uq]&g['geometry_valid'][vq+1,uq+1]
        ground=np.isin(lab,[roadID,laneID]);road=lab==roadID;lane=lab==laneID
        flags=dict(inside_original_OBB=box0,inside_expanded_OBB=boxwide,photo_bounded=bounded,photo_road0=plabel==0,photo_road0_erode2=proad2,parent_geometry_all4=parent4)
        row=dict(raw=r,train_or_heldout='train' if r in oldcontract['train_raw'] else 'heldout',official_labels_present=haslabels,
            timestamp_micros=int(frame.timestamp_micros),TOP_returns=returns,input_rows=len(original),local_broad_region_points=len(X),
            local_TOP_points_with_semantic_entry=int((lab>=0).sum()),local_defined_official_semantic_points=int((lab>0).sum()),
            local_semantic_undefined_or_nonTOP=int((lab<=0).sum()),
            local_official_class_counts={str(int(k)):int(n) for k,n in zip(*np.unique(lab,return_counts=True))},
            official_road_points=int(road.sum()),official_lane_marker_points=int(lane.sum()),official_ground_union_points=int(ground.sum()),
            official_ground_independent_flag_counts={k:int((ground&v).sum()) for k,v in flags.items()},
            ground_photo_nonroad_inbounds=int((ground&bounded&(plabel!=0)).sum()),ground_photo_outside=int((ground&~bounded).sum()),
            official_ground_outside_expanded_OBB=int((ground&~boxwide).sum()),
            official_ground_in_parent_missing_quad=int((ground&inbounds&~parent4).sum()),
            ground_height_minus_old_profile_m=stats(X[ground,2]-design(X[ground,:2],s['fit_XY_centre'])@s['fit_coefficients']),
            source_lidar=pin(original_path),source_observation=pin(obs_path),source_real_photo_semantic_proxy=pin(sem_path),
            official_point_alignment_verified=bool(haslabels),no_ground_point_rejected_by_OBB_or_photo=True)
        records.append(row);inputs.extend([pin(original_path),pin(obs_path),pin(sem_path),pin(calpath)])
        all_points.append(dict(world_xyz=X,raw=np.full(len(X),r,np.int16),lidar_row_id=ids.astype(np.int32),semantic=lab,instance=inst,
            TOP_encoded_index=original[ids,5],photo_label=plabel,atlas_uv_global=uvglobal,ground=ground,
            used_for_fit=np.zeros(len(X),bool),**flags))
        print(json.dumps(dict(stage='official_point_evidence',**row)),flush=True)
    payload={k:np.concatenate([x[k] for x in all_points]) for k in all_points[0]}
    np.savez_compressed(out/'local_original_point_evidence.npz',**payload)
    np.savez_compressed(out/'official_TOP_range_labels.npz',**range_payload)
    train=np.isin(payload['raw'],oldcontract['train_raw']);ground=payload['ground'];use=train&ground
    X=payload['world_xyz'][use];raw=payload['raw'][use]
    assert len(X)>3
    factors=neighbourhood(X,raw,X,raw,query)
    hull=ConvexHull(X[:,:2]);margin=-(query@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
    factors['official_ground_hull_margin_m']=margin
    original=s['original_geometry_supported'];missing=~original
    maps={k:v.reshape(shape) for k,v in factors.items()}
    named=dict(missing_parent_inside_official_ground_hull=(margin.reshape(shape)>=0)&missing,
        missing_parent_nearest_official_ground_le0p8m=(maps['nearest_training_point_m']<=.8)&missing,
        missing_parent_three_frames_le0p8m=(maps['distinct_frames_within_0.8m']>=3)&missing,
        missing_parent_three_quadrants_le1m=(maps['support_quadrants_within_1m']>=3)&missing)
    targetrows=[]
    for r in [86,110]:
        tpath=R10/f'audit/local_surface_inference_v1_replay1/{r:03d}_target_coverage_DIAGNOSTIC.npz';t=load(tpath)
        uv=t['quadratic_diagnostic_atlas_uv_global'];pvalid=t['quadratic_full_hull_DIAGNOSTIC_projected']
        regions={}
        for name in ['red_diagnostic','focused_residual_candidate']:
            mask=t[name];eligible=mask&pvalid;xy=uv[eligible]@M.T+origin
            qf=neighbourhood(X,raw,X,raw,xy)
            qm=-(xy@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
            regions[name]=dict(fixed_pixels=int(mask.sum()),proposed_profile_locus_pixels=int(eligible.sum()),
                point_evidence_hull_inside=int((qm>=0).sum()),nearest_official_ground_le0p8m=int((qf['nearest_training_point_m']<=.8).sum()),
                at_least3_official_ground_frames_within0p8m=int((qf['distinct_frames_within_0.8m']>=3).sum()),
                at_least3_quadrants_within1m=int((qf['support_quadrants_within_1m']>=3).sum()),
                factors={k:stats(v) for k,v in qf.items()},
                conditional_XY_locus='Only uses prior unapproved profile ray intersection to locate diagnostic pixel; this is not a depth/geometry coverage PASS')
        targetrows.append(dict(raw=r,regions=regions,original_target=pin(tpath)))
    np.savez_compressed(out/'official_ground_support_DIAGNOSTIC.npz',uv_global=s['uv_global'],parent_missing=missing,
        renderable_mask=np.zeros(shape,bool),**maps,**named)
    result=dict(schema='R10.official_ground_physical_point_evidence.v1',complete=True,decision='POINT_EVIDENCE_ONLY_NO_GEOMETRY_APPROVAL',
        point_contract=pin(out/'point_evidence_contract.json'),original_TFRecord_reference=receipt['raw'],original_TFRecord_sha256=receipt['raw_sha256'],
        train_frames=oldcontract['train_raw'],heldout_frames=oldcontract['heldout_raw'],frame_records=records,
        official_ground_training_points=int(use.sum()),official_ground_heldout_points=int((ground&~train).sum()),
        original_parent_missing_texels=int(missing.sum()),missing_parent_independent_support_counts={k:int(v.sum()) for k,v in named.items()},
        target_rows=targetrows,point_evidence=pin(out/'local_original_point_evidence.npz'),
        exact_TOP_semantic_arrays=pin(out/'official_TOP_range_labels.npz'),
        support_diagnostic=pin(out/'official_ground_support_DIAGNOSTIC.npz'),
        strict837_or_old_profile_modified=False,new_fitted_geometry=False,new_renderable_geometry=False,
        geometry_confidence_is_calibrated_probability=False,heldout_fitted=False,
        actual_readback_inputs=inputs,CPU_only=True,GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'official_ground_point_receipt.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'official_ground_point_receipt.json'))),flush=True)


if __name__=='__main__':main()
