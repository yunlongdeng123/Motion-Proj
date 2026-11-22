"""Register every available official-road frame in the fixed86..198 window.

No height-based pool selection, camera/vehicle edit, heldout input or GPU work.
"""
import argparse,hashlib,importlib.util,json,struct,time,zlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from diagnose_nonplanar_inference_cpu import load,read,write,pin
from diagnose_expanded_actual_scope_cpu import labels

FRAMES=list(range(90,171,5))
HELDOUT=[134,179,198]


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def imatrix(compressed,pb):
    value=pb.MatrixInt32();value.ParseFromString(zlib.decompress(compressed));return np.asarray(value.data,np.int32).reshape(tuple(value.shape.dims))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10'
    avpath=R/'audit/official_TOP_availability_199_v1/TOP_segmentation_availability.json';av=read(avpath)
    assert av['available_in_time_window']==FRAMES and not set(FRAMES)&set(HELDOUT)
    index={x['raw']:x['record'] for x in av['rows']}
    gp=R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz';g=load(gp)
    xy=g['cell_world_xyz'][g['requested_cell_asset_mask'],:2];tree=cKDTree(xy);centre=(xy.min(0)+xy.max(0))/2;origin=xy.min(0)-.4
    grid=load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz');M,O=grid['world_from_uv_xy'][:2,:2],grid['world_from_uv_xy'][:2,3]
    contract=dict(schema='R10.timewindow17_reference_pool_contract.v1',complete=True,written_before_pool_point_heights_read=True,
        fixed_evaluation_window_inclusive=[86,198],selection_rule='Every available official TOP semantic frame inside time window, all17 included, no height/colour selection',
        frames=FRAMES,availability=pin(avpath),geometry_heldout_do_not_fit=HELDOUT,
        comparison_scope='Time-window change plus additional valid annotated observations; not a count-only ablation, not a declaration that raw60/70 or their poses are invalid',
        XY_scope='Same1844 pre-screen requested_cell_asset_mask footprint plus .4m Euclidean XY context',scope_geometry=pin(gp),
        point_filter='All original official ROAD18/LANE19 points finite and in declared XY context; no Z/residual/OBB/photo filter',
        initial_audit='All17 cross-frame nearest XY consistency with anchor110 and eps .02/.05/.10m, retaining original denominators/mutual matches; no offset synchronization or point rejection',
        later_model_family='Unchanged plane/quadratic curvature ridge family, frame LOO/oneSE and fixed physical gates; no extra models to hide correspondence conflict',
        old8_reference_and_failed_models_unchanged=True,heldout_not_used_for_pool_selection=True,
        A20_camera_oldroad_and_sourceRGB_unchanged=True,model_fitted=False,GPU_forward=0,backward=0,updates=0)
    write(out/'reference_pool_contract.json',contract)
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');receipt=read(bank/'receipt.json')
    dr=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');decoder=module('window17_original_records',dr/'waymo_numpy_cpu.py');pb=decoder.protobuf();crc=decoder.crc_function()
    data=[];rows=[];pins=[]
    with Path(receipt['raw']).open('rb') as stream:
        for rr in FRAMES:
            ir=index[rr];stream.seek(ir['header_offset']);header=stream.read(12);length,ch=struct.unpack('<QI',header)
            assert length==ir['payload_length'] and crc(header[:8])==ch
            payload=stream.read(length);trailer=stream.read(4)
            assert crc(payload)==struct.unpack('<I',trailer)[0] and hashlib.sha256(payload).hexdigest()==ir['payload_sha256']
            frame=pb.Frame.FromString(payload);op=bank/f'observations/{rr:03d}.json';obs=read(op)
            assert frame.timestamp_micros==obs['timestamp_micros'] and np.array_equal(np.asarray(frame.pose.transform),obs['ego_world'])
            lp=bank/obs['lidar_path'];cloud=np.fromfile(lp,dtype='<f4').reshape(-1,6);lab=np.full(len(cloud),-1,np.int16);frontuv=np.full((len(cloud),2),-1,np.int32);front_present=np.zeros(len(cloud),bool)
            top=next(x for x in frame.lasers if x.name==1)
            for ret in [0,1]:
                source=getattr(top,f'ri_return{ret+1}');seg=imatrix(source.segmentation_label_compressed,pb);H,W=seg.shape[:2]
                take=(cloud[:,5]>=ret*H*W)&(cloud[:,5]<(ret+1)*H*W);ids=np.flatnonzero(take);encoded=cloud[ids,5].astype(np.int64)-ret*H*W
                lab[ids]=seg.reshape(-1,2)[encoded,1]
                cp=imatrix(source.camera_projection_compressed,pb).reshape(-1,6)[encoded]
                first=cp[:,0]==1;second=(cp[:,3]==1)&~first;frontuv[ids[first]]=cp[first,1:3];frontuv[ids[second]]=cp[second,4:6];front_present[ids]=first|second
            official=np.isin(lab,[18,19]);assert int(official.sum())==av['rows'][rr]['valid_range_nonNLZ_official_ground_count']
            ids=np.flatnonzero(official);ego=np.asarray(obs['ego_world']).reshape(4,4);X=cloud[ids,:3].astype(float)@ego[:3,:3].T+ego[:3,3]
            finite=np.isfinite(X).all(-1);distance=np.full(len(X),np.inf);distance[finite]=tree.query(X[finite,:2])[0];keep=finite&(distance<=.4)
            X,ids,distance=X[keep],ids[keep],distance[keep];uv=(X[:,:2]-O)@np.linalg.inv(M).T;actual=labels(uv,g,'requested_cell_asset_mask')
            data.append(dict(world_xyz=X,source_raw=np.full(len(X),rr,np.int16),source_lidar_row_id=ids.astype(np.int32),
                official_semantic=lab[ids],TOP_encoded=cloud[ids,5].astype(np.int64),official_front_uv=frontuv[ids],
                official_front_present=front_present[ids],actual_requested_spatial_mask=actual,distance_to_requested_footprint_m=distance))
            row=dict(raw=rr,timestamp_micros=int(frame.timestamp_micros),all_official_ROAD_LANE_points=int(official.sum()),
                XY_context_points=len(X),actual_requested_points=int(actual.sum()),ROAD_points=int((lab[ids]==18).sum()),LANE_points=int((lab[ids]==19).sum()),
                source_record=ir,cloud=pin(lp),obs=pin(op),all_pool_frames_retained=True,no_height_filter=True)
            rows.append(row);pins.extend([pin(lp),pin(op)])
            print(json.dumps(dict(stage='timewindow17_frame',raw=rr,context_points=len(X),actual_requested=int(actual.sum()))),flush=True)
    result={key:np.concatenate([x[key] for x in data]) for key in data[0]}
    np.savez_compressed(out/'training_points.npz',**result,fit_XY_centre=centre,medoid_origin=origin,
        reference_frames=np.asarray(FRAMES,np.int16),geometry_heldout=np.asarray(HELDOUT,np.int16))
    receipt=dict(schema='R10.timewindow17_original_official_ground_cache.v1',complete=True,decision='DATA_READY_NO_GEOMETRY_QUALITY_CLAIM',
        training_points=pin(out/'training_points.npz'),frames=FRAMES,points=len(result['world_xyz']),
        actual_requested_points=int(result['actual_requested_spatial_mask'].sum()),rows=rows,
        contract=pin(out/'reference_pool_contract.json'),source_dataset_receipt=pin(bank/'receipt.json'),
        all_record_payload_SHA_CRC_verified=True,all_frame_pose_timestamp_bindings_exact=True,
        original_bin_export_not_redecoded_in_this_cache_phase=True,
        no_Z_residual_filter=True,no_RGB_read=True,no_heldout_read=True,no_model_fit=True,
        old_assets_cameras_source_and_split_unchanged=True,GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'training_data_receipt.json',receipt)
    print(json.dumps(dict(complete=True,receipt=pin(out/'training_data_receipt.json'),cache=receipt['training_points'],points=receipt['points'],actual=receipt['actual_requested_points'])),flush=True)


if __name__=='__main__':main()
