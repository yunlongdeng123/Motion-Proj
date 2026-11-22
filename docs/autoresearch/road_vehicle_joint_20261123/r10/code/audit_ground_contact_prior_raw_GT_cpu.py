"""GT box ground-contact evidence and frozen A20 mean-height quantiles; read-only."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def offset_to_box_bottom(world,pose,dimensions):
    local=(world-pose[:3,3])@pose[:3,:3]
    outside=np.maximum(abs(local[:,:2])-dimensions[:2]/2,0)
    distance=np.linalg.norm(outside,axis=-1)
    # Positive means GT annotation bottom is above the measured road point.
    offset=-(local[:,2]+dimensions[2]/2)/pose[2,2]
    return distance,offset,local


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R=C/'road_underbody_local_20261123_r10';ev=R/'audit/official_ground_geometry_v1'
    points=load(ev/'local_original_point_evidence.npz');ranges=load(ev/'official_TOP_range_labels.npz')
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');receipt=read(bank/'receipt.json')
    registry=read(bank/'track_registry.json');tracks={x['alias']:k for k,x in registry['actors'].items() if x['alias'] in ['a20','a9']};assert len(tracks)==2
    pool=read(R/'audit/local_surface_geometry_v1/geometry_qualification_contract.json');frames=set(pool['train_raw']+pool['heldout_raw'])
    code_contract=dict(schema='R10.GT_contact_prior_readonly_contract.v1',complete=True,written_before_comparison=True,
        tracks=tracks,frames=sorted(frames),primary_evidence='Original Waymo 3D GT box bottom vs original official ROAD18/LANE19 points',
        box_bottom='actor-local centre origin, XYZ axes length/width/height; z=-height/2, exact check against raw label centre/dimensions/heading and ego pose',
        offsets='positive GT-box bottom minus physical road height, vertically at each actual road XY; no road residual-based point trim',
        neighbour_footprint_radii_m=[.5,1.,1.5],main_radius_m=1.,
        canonical_actor='approved ab4fa A20 active mean positions, opacity>=.5; report lower quantiles, never treat minimum/means as GT contact',
        prior_calibration='training frames only if later adopted; current task only descriptive statistics, no fitted ground or actor changes',
        primary_pose_source='Direct original TFRecord ego pose @ rawGT heading/centre and rawGT length/width/height. Observation exact-equality predicate remains separately false when derived dimensions differ by micrometres.',
        source_evidence=pin(ev/'official_ground_point_receipt.json'),GPU_forward=0,updates=0)
    write(out/'contact_diagnostic_contract.json',code_contract)
    official_by_raw={}
    for r in pool['train_raw']:
        if f'{r:03d}_return0_labels_instance_semantic' not in ranges:continue
        obs=read(bank/f'observations/{r:03d}.json');cloud=np.fromfile(bank/obs['lidar_path'],dtype='<f4').reshape(-1,6)
        lab=np.full(len(cloud),-1,np.int16)
        for ri in [0,1]:
            key=f'{r:03d}_return{ri}_labels_instance_semantic'
            if key not in ranges:continue
            seg=ranges[key];H,W=seg.shape[:2];hit=(cloud[:,5]>=ri*H*W)&(cloud[:,5]<(ri+1)*H*W)
            index=cloud[hit,5].astype(np.int64)-ri*H*W;lab[hit]=seg.reshape(-1,2)[index,1]
        take=np.isin(lab,[18,19]);pose=np.asarray(obs['ego_world']).reshape(4,4)
        world=cloud[take,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
        official_by_raw[r]=dict(world=world,raw=np.full(len(world),r,np.int16),lidar_ids=np.flatnonzero(take).astype(np.int32),semantic=lab[take])
    combined={k:np.concatenate([x[k] for x in official_by_raw.values()]) for k in ['world','raw','lidar_ids','semantic']}
    droot=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');decoder=module('contact_raw_decoder',droot/'waymo_numpy_cpu.py');pb=decoder.protobuf()
    gtrows=[];world_bottom={};observations={};binding=True
    for r,payload,crc in decoder.records(receipt['raw'],max(frames)):
        if r not in frames:continue
        frame=pb.Frame.FromString(payload);obs=read(bank/f'observations/{r:03d}.json');ego=np.asarray(frame.pose.transform).reshape(4,4)
        observations[r]=obs
        for alias,track in tracks.items():
            label=next((x for x in frame.laser_labels if x.id==track),None);ob=next((x for x in obs['boxes'] if x['track']==track),None)
            if label is None or ob is None:
                gtrows.append(dict(raw=r,alias=alias,track=track,present=False));continue
            box=label.box;h=box.heading;local=np.eye(4);local[:3,:3]=[[np.cos(h),-np.sin(h),0],[np.sin(h),np.cos(h),0],[0,0,1]]
            local[:3,3]=[box.center_x,box.center_y,box.center_z];expected=ego@local
            pose=np.asarray(ob['world_from_actor']).reshape(4,4);dims=np.array([box.length,box.width,box.height])
            pose_error=float(abs(expected-pose).max());dim_error=float(abs(dims-np.asarray(ob['dimensions'])).max());good=pose_error<1e-10 and dim_error==0
            binding&=good
            # Use direct raw GT, independently of the observation roundoff comparison.
            pose=expected
            local_corners=np.array([[x,y,-dims[2]/2] for x in [-dims[0]/2,dims[0]/2] for y in [-dims[1]/2,dims[1]/2]]+[ [0,0,-dims[2]/2] ])
            bottom=local_corners@pose[:3,:3].T+pose[:3,3];world_bottom[(r,alias)]=bottom
            groups={}
            for group,world in [('same_frame',official_by_raw.get(r,{}).get('world',np.empty((0,3)))),('all_training_frames_static_road_pool',combined['world'])]:
                distance,offset,actorxyz=offset_to_box_bottom(world,pose,dims)
                groups[group]={}
                for radius in [.5,1.,1.5]:
                    hit=distance<=radius
                    groups[group][str(radius)]=dict(points=int(hit.sum()),GT_bottom_minus_measured_road_Z_m=stats(offset[hit]),
                        inside_box_XY_points=int((hit&(distance<=1e-12)).sum()),world_XY_bbox=np.stack([world[hit,:2].min(0),world[hit,:2].max(0)]).tolist() if hit.any() else None,
                        source_raw_count=int(len(np.unique(combined['raw'][hit]))) if group.startswith('all_') else (1 if hit.any() else 0))
            gtrows.append(dict(raw=r,alias=alias,track=track,present=True,train_or_heldout='train' if r in pool['train_raw'] else 'heldout',
                raw_GT_pose_formula_max_abs=pose_error,raw_GT_dimensions_max_abs=dim_error,GT_origin_and_dimensions_verified=good,
                dimensions_lwh_m=dims.tolist(),world_from_actor=pose.tolist(),bottom_world_corners_and_centre=bottom.tolist(),
                point_offset_statistics=groups,original_observation=pin(bank/f'observations/{r:03d}.json')))
    actorpath=C/'a20_joint_reconstruction_20261122_r1/execution_r3/post_geometry_ablation/assets/geometry_ablation/actor.npz'
    actorpin=pin(actorpath);assert actorpin['sha256']=='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
    actor=load(actorpath);take=actor['active']&(actor['opacity']>=.5);XYZ=actor['xyz'][take].astype(np.float64)
    bodyrows=[]
    for row in gtrows:
        if row['alias']!='a20' or not row['present'] or not len(XYZ):continue
        pose=np.asarray(row['world_from_actor']);dims=np.asarray(row['dimensions_lwh_m']);world=XYZ@pose[:3,:3].T+pose[:3,3]
        _,offset,_=offset_to_box_bottom(world,pose,dims)
        bodyrows.append(dict(raw=row['raw'],active_high_opacity_means=len(XYZ),
            world_Z_quantiles={str(q):float(np.quantile(world[:,2],q)) for q in [.001,.01,.05,.10,.50]},
            mean_Z_above_GT_bottom_at_same_XY_m_quantiles={str(q):float(np.quantile(-offset,q)) for q in [.001,.01,.05,.10,.50]},
            minimum_Z_outlier_diagnostic_only=float(world[:,2].min()),minimum_not_used_as_ground_truth=True))
    summaries={}
    for alias in tracks:
        trainrows=[x for x in gtrows if x['alias']==alias and x.get('present') and x.get('train_or_heldout')=='train']
        medians=[];per=[]
        for row in trainrows:
            same=row['point_offset_statistics']['same_frame']['1.0']['GT_bottom_minus_measured_road_Z_m']
            if same.get('count',0)>0:
                medians.append(same['median']);per.append(dict(raw=row['raw'],points=same['count'],median_m=same['median']))
        summaries[alias]=dict(training_same_frame_radius1m_per_frame_medians=stats(medians),per_frame=per,
            offset_is_annotation_bottom_proxy_not_measured_hidden_ground=True)
    np.savez_compressed(out/'GT_bottom_and_road_evidence.npz',**{f'{r:03d}_{alias}_bottom_world':v for (r,alias),v in world_bottom.items()},
        official_training_road_worldXYZ=combined['world'],official_training_road_source_raw=combined['raw'],official_training_road_lidar_row_id=combined['lidar_ids'])
    result=dict(schema='R10.GT_ground_contact_prior_audit.v1',complete=True,decision='DESCRIPTIVE_PRIOR_EVIDENCE_ONLY_NO_GEOMETRY_APPROVAL',
        observation_exact_dimension_binding_pass=bool(binding),raw_GT_used_directly_for_bottom_and_dimensions=True,
        observation_roundoff_does_not_define_rawGT_geometry=True,registry=pin(bank/'track_registry.json'),tracks=tracks,
        GT_bottom_rows=gtrows,training_offset_stability=summaries,
        approved_A20_original_asset=actorpin,actor_mean_quantile_rule='active and opacity>=.5; no scale/opacity/geometry modified; centre quantiles are diagnostic not tyre surface GT',
        canonical_A20_high_opacity_Z_quantiles={str(q):float(np.quantile(XYZ[:,2],q)) for q in [.001,.01,.05,.1,.5]} if len(XYZ) else {},
        canonical_A20_high_opacity_count=len(XYZ),A20_world_quantile_rows=bodyrows,
        no_prior_interval_activated=True,no_vehicle_adjustment=True,new_GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'ground_contact_prior_audit.json',result);print(json.dumps(dict(complete=True,receipt=pin(out/'ground_contact_prior_audit.json'),summary=summaries)),flush=True)


if __name__=='__main__':main()
