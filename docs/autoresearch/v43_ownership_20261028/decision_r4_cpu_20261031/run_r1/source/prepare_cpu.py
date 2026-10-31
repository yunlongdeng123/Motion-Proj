"""真实M0/自然重建/真实街景的R4 CPU入口；不渲染、不训练。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from r4_core import (OPERATIONS, canonical_hash, enumerate_routes, freeze_family,
                     register_tasks, validate_candidate)

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
SIM = ROOT / 'sim_mechanism_20261030'
CTX = '14705303724557273004_3105_000_3125_000'
NATIVE = ROOT.parent/'persistent_world_20261027'
SOURCE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
HASH_CACHE = {}
INPUTS = {}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def receipt(path, role):
    path = Path(path).resolve()
    key = str(path)
    stat = path.stat()
    if key not in HASH_CACHE:
        digest=sha256()
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(4*1024*1024),b''):
                digest.update(data)
        HASH_CACHE[key]=dict(path=key,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest.hexdigest())
    row=dict(HASH_CACHE[key],role=role)
    INPUTS[key]=row
    return row


def read(path, role):
    receipt(path,role)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def matrix(path, role):
    receipt(path,role)
    data=np.loadtxt(path)
    if data.shape!=(4,4) or not np.isfinite(data).all():
        raise ValueError('转换矩阵形状/数值错误')
    scale=float(np.linalg.norm(data[:3,0]))
    if scale<=0 or not np.allclose(data[:3,:3].T@data[:3,:3],np.eye(3)*scale**2,atol=1e-8):
        raise ValueError('必须恢复原训练有限各向同性similarity')
    return data


def tensor_summary(value, prefix=''):
    result=[]
    if torch.is_tensor(value):
        if value.device.type!='cpu':
            raise ValueError('map_location=cpu后发现非CPU tensor')
        sample=value.detach().reshape(-1)
        if sample.numel():
            probe=torch.cat((sample[:8],sample[-8:]))
            finite=bool(torch.isfinite(probe).all()) if probe.is_floating_point() else True
            if not finite:
                raise ValueError('checkpoint边界采样出现非有限值')
        result.append(dict(name=prefix,shape=list(value.shape),dtype=str(value.dtype),device='cpu',
                           finite_boundary_sample=True))
    elif isinstance(value,dict):
        for key,child in value.items():result.extend(tensor_summary(child,prefix+'.'+str(key)))
    elif isinstance(value,(list,tuple)):
        for i,child in enumerate(value):result.extend(tensor_summary(child,prefix+'.'+str(i)))
    return result


def load_checkpoint(event):
    if event['status']!='complete' or set(event['checkpoints'])!={'static','dynamic','environment','training_state'}:
        raise ValueError('完整checkpoint bundle缺失')
    records, static = [], None
    for name,entry in event['checkpoints'].items():
        path=Path(entry['path'])
        if not path.is_file() or path.stat().st_size!=entry['bytes']:
            raise ValueError('checkpoint文件缺失或字节数不符')
        frozen=receipt(path,'frozen_native_checkpoint:'+name)
        started=time.monotonic()
        loaded=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if name=='training_state':
            payload=loaded
            step=payload.get('iteration',event['iteration'])
        else:
            payload,step=loaded
        if step!=event['iteration']:
            raise ValueError('checkpoint步数不匹配')
        tensors=tensor_summary(payload,name)
        records.append(dict(name=name,path=str(path),sha256=frozen['sha256'],bytes=frozen['bytes'],
                            iteration=step,map_location='cpu',mmap=True,tensors=tensors,
                            load_and_boundary_check_wall_seconds=time.monotonic()-started,
                            restored_native_model_class=False,optimizer_created=False))
        if name=='static':
            if len(payload)!=18 or payload[1].ndim!=2 or payload[1].shape[1]!=3:
                raise ValueError('未知原生static capture布局')
            static=payload
        else:
            del loaded,payload
        gc.collect()
    if static is None:raise ValueError('缺少static')
    return static,records


def camera_world(row, transform):
    scale=np.linalg.norm(transform[:3,0])
    inverse=np.linalg.inv(transform)
    c2w=np.eye(4)
    c2w[:3,:3]=inverse[:3,:3]@np.asarray(row['rotation'])*scale
    c2w[:3,3]=inverse[:3,:3]@np.asarray(row['position'])+inverse[:3,3]
    return c2w


def sim_inputs(event, source):
    prep=read(source/'preparation_manifest.json','original_native_preparation_contract')
    obs=Path(prep['observations'])
    manifest=read(obs/'manifest.json','legal_sensor_oracle_observation_manifest')
    if manifest.get('declared_input_mode')!='ideal_sensor_and_tracks_interface_validation':
        raise ValueError('未登记的sim观测合同')
    tracks=read(obs/manifest['tracks_path'],'legal_sensor_oracle_tracks')['tracks']
    byid={str(x['frame_id']).zfill(6):x for x in tracks}
    prepared={x['id']:x for x in prep['frames']}
    frames=[]
    for row in manifest['frames']:
        fid=str(row['id']).zfill(6)
        if fid not in byid or fid not in prepared:raise ValueError('sim相机/轨迹映射不完整')
        mask_path=obs/row['body_mask']; lidar_path=obs/row['lidar']
        receipt(mask_path,'legal_sensor_oracle_original_body_mask')
        receipt(lidar_path,'legal_noise_free_sparse_lidar_no_owner_labels')
        mask=np.asarray(Image.open(mask_path))>0
        if mask.ndim==3:mask=mask.any(axis=2)
        with np.load(lidar_path,allow_pickle=False) as data:
            # 严格只读许可数组；没有访问semantic/surface/dense depth。
            xyz=data['xyz_world'].copy();uv=data['uv'].copy()
        if xyz.shape!=(len(uv),3) or uv.shape!=(len(xyz),2):raise ValueError('sim LiDAR布局错误')
        pix=np.floor(uv).astype(np.int64)
        valid=np.isfinite(xyz).all(1)&np.isfinite(uv).all(1)&(pix[:,0]>=0)&(pix[:,0]<mask.shape[1])&(pix[:,1]>=0)&(pix[:,1]<mask.shape[0])
        support=np.flatnonzero(valid)
        support=support[mask[pix[support,1],pix[support,0]]]
        track=byid[fid]
        center=np.asarray(track['box_center_world'],dtype=float)
        rotation=np.asarray(track['object_to_world'],dtype=float)[:3,:3]
        frames.append(dict(id=fid,model_time=float(prepared[fid]['model_time']),
            timestamp_s=float(row['timestamp_s']),camera_to_world=np.asarray(row['camera_to_world']),
            center=center,rotation=rotation,body_pixels=int(mask.sum()),
            measured_depth_pixels=len(np.unique(pix[support],axis=0)),
            support_world=xyz[support].astype(np.float64),support_source_rows=support,
            support_ref=str(lidar_path),mask_ref=str(mask_path)))
    return frames,dict(role='simulator_sensor_oracle',
        role_detail='body/sky masks and ideal tracks explicitly authorized by inherited mechanism observation contract; not deployed RGB predictions',
        original_input_contract=str(obs/'manifest.json'),
        limitations=['one procedural world; sensor-oracle masks/tracks and noise-free sparse LiDAR',
                     'no independent edited RGB, semantic/depth reference or full mesh used'])


def real_inputs(event, source):
    dataset=ROOT/'datasets'/CTX
    manifest=read(dataset/'manifest.json','original_real_actor_dataset_manifest')
    review=read(source/'native_preparation/selection_reviewed.json','original_reviewed_target_association_not_pixel_GT')
    indices_path=source/'dynamic_actor_assets/actor_0/source_lidar_indices.npz'
    receipt(indices_path,'reviewed_DEVA_annotation_assisted_target_to_measured_lidar_rows')
    prep=read(source/'preparation_manifest.json','original_native_preparation_contract')
    frames=[]
    with np.load(indices_path,allow_pickle=False) as indices:
        for row in manifest['frames']:
            fid=row['frame_id'];data_path=dataset/('%03d.npz'%row['raw'])
            receipt(data_path,'original_real_RGB_body_and_measured_sparse_depth')
            with np.load(data_path,allow_pickle=False) as data:
                body=data['body'].copy();depth=data['depth'].copy()
            point_path=source/'velodyne'/(fid+'.bin');pose_path=source/'pose'/(fid+'.txt')
            receipt(point_path,'original_measured_lidar');receipt(pose_path,'original_ego_pose')
            selected=indices[fid].astype(np.int64)
            raw=np.fromfile(point_path,np.float32).reshape(-1,6)
            if len(np.unique(selected))!=len(selected) or (selected<0).any() or (selected>=len(raw)).any():
                raise ValueError('真实LiDAR原始行索引错误')
            pose=np.loadtxt(pose_path)
            points=raw[selected,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            frames.append(dict(id=fid,raw=int(row['raw']),model_time=float(row['model_time']),
                timestamp_s=None,camera_to_world=None,
                center=np.asarray(row['actor_center_world']),rotation=np.asarray(row['actor_rotation_world']),
                body_pixels=int(body.sum()),measured_depth_pixels=int((body&np.isfinite(depth)&(depth>0)).sum()),
                support_world=points,support_source_rows=selected,
                support_ref=str(point_path),mask_ref=str(data_path)))
    return frames,dict(role='reviewed_measured_real_observations',
        role_detail='reviewed_DEVA_association_and_measured_lidar_with_annotation_assisted_target_track',
        original_input_contract=str(dataset/'manifest.json'),source_identity_review=str(source/'native_preparation/selection_reviewed.json'),
        original_preparation_training_ready=prep.get('training_ready'),
        limitations=['target association uses original annotations and reviewed DEVA masks; not an annotation-free method',
                     'review association is not per-Gaussian owner/capability GT; human_verdict remains null',
                     'original preparation training_ready flag is false; complete checkpoint event exists, native restore still pending'])


def observations(event, source, cameras, transform, kind):
    frames,role=sim_inputs(event,source) if kind=='sim' else real_inputs(event,source)
    byid={str(row['img_name']):row for row in cameras}
    restored=[]
    for frame in frames:
        if frame['id'] not in byid:raise ValueError('观测不属于冻结训练相机')
        c2w=camera_world(byid[frame['id']],transform)
        error=None
        if frame['camera_to_world'] is not None:
            error=float(np.max(np.abs(frame['camera_to_world']-c2w)))
            if error>1e-4:raise ValueError('CPU camera/gauge重建不一致')
        else:frame['camera_to_world']=c2w
        restored.append(dict(frame_id=frame['id'],model_time=frame['model_time'],
            camera_to_world=frame['camera_to_world'].tolist(),original_camera_consistency_abs_error=error,
            actor_center_world=frame['center'].tolist(),actor_rotation_world=frame['rotation'].tolist(),
            measured_target_returns=len(frame['support_world']),original_body_pixels=frame['body_pixels']))
    return frames,role,restored


def view_metric(frames, operation):
    # 12个方位扇区只是合法视向代理；不是完整表面可见性/颜色支持。
    observed=set();required=set()
    for f in frames:
        if not f['body_pixels']:continue
        origin=f['camera_to_world'][:3,3]
        current=(origin-f['center'])@f['rotation']
        angle=np.arctan2(current[1],current[0])
        observed.add(int(np.floor((angle%(2*np.pi))/(2*np.pi)*12))%12)
        shifted=f['center']+np.array([0,1.5,0]) if operation=='shift' else f['center']
        future=(origin-shifted)@f['rotation']
        needed=np.arctan2(future[1],future[0])-(np.pi/6 if operation=='rotate' else 0)
        required.add(int(np.floor((needed%(2*np.pi))/(2*np.pi)*12))%12)
    return dict(numerator=len(observed&required),denominator=len(required),
        status='legal_azimuth_sector_proxy_not_surface_coverage',
        observed_sectors=sorted(observed),required_sectors=sorted(required),
        total_sector_count=12,frame_denominator=len(frames))


def legal_evidence(frames,role):
    valid=[]
    for f in frames:
        values=np.concatenate((f['center'].reshape(-1),f['rotation'].reshape(-1),f['camera_to_world'].reshape(-1),[f['model_time']]))
        valid.append(bool(np.isfinite(values).all() and np.allclose(f['rotation'].T@f['rotation'],np.eye(3),atol=1e-4)))
    geom=dict(numerator=None,denominator=sum(f['body_pixels'] for f in frames),
        status='unknown_native_full_static_plus_dynamic_alphaT_not_rendered',
        measured_body_depth_pixel_denominator=sum(f['measured_depth_pixels'] for f in frames),
        view_denominator=len(frames),definition='full original scene contribution near measured depth within legal body mask; cannot be replaced by candidate nearest-neighbour count')
    bg=dict(numerator=None,denominator=None,status='unknown_historical_world_ray_reveal_matching_not_implemented',
        definition='pre-registered predicted reveal domain and historical depth-valid visibility; independent delete GT forbidden')
    motion=dict(numerator=sum(valid),denominator=len(frames),status='CPU_camera_track_field_consistency_only',
        native_scene_and_model_runtime_restore='E0_GPU_gate_pending',
        model_time_min=min(f['model_time'] for f in frames),model_time_max=max(f['model_time'] for f in frames))
    return dict(schema='r4_legal_preedit_evidence_v1',role=role['role'],role_detail=role['role_detail'],
        legal_observation_ids=[f['support_ref'] for f in frames]+[role['original_input_contract']],
        metrics=dict(f_geom=geom,f_view=view_metric(frames,'none'),f_bg=bg,f_motion=motion),
        operation_view={q:view_metric(frames,q) for q in OPERATIONS},
        operation_bg={q:dict(bg) for q in OPERATIONS},
        input_role_limitations=role['limitations'])


def static_at(payload, timestamp, start, stop):
    """精确继承当前static get_xyz_SHM/get_marginal_t的轻量CPU公式。"""
    xyz=payload[1][start:stop].detach().numpy()
    center=payload[7][start:stop].detach().numpy()
    velocity=payload[9][start:stop].detach().numpy()
    period=float(payload[16]);frequency=2*np.pi/period
    means=xyz+velocity*np.sin((timestamp-center)*frequency)/frequency
    with np.errstate(over='ignore',under='ignore',invalid='raise',divide='ignore'):
        scale_t=np.exp(payload[8][start:stop].detach().numpy())
        marginal=np.exp(-.5*((center-timestamp)/scale_t)**2)
        logit=payload[6][start:stop].detach().numpy()
        opacity=1/(1+np.exp(-np.clip(logit,-80,80)))*marginal
    return means,opacity.reshape(-1)


def candidates(payload,frames,transform,scene_id,ancestor,role,checkpoint_sha,radius):
    n=len(payload[1]);assigned=np.zeros(n,bool);inverse=np.linalg.inv(transform)
    anchors=[frames[i] for i in sorted({0,len(frames)//2,len(frames)-1})]
    groups=[];records=[];anchor_receipts=[]
    for f in anchors:
        supported=f['support_world'];group_rows=[];group_nearest=[];group_distance=[];group_world=[];group_opacity=[]
        if len(supported):
            tree=cKDTree(supported)
            for start in range(0,n,65536):
                stop=min(n,start+65536)
                means,opacity=static_at(payload,f['model_time'],start,stop)
                world=means.astype(np.float64)@inverse[:3,:3].T+inverse[:3,3]
                if not np.isfinite(world).all():raise ValueError('真实checkpoint位置/时间公式非有限')
                distance,nearest=tree.query(world,k=1,workers=1)
                selected=(distance<=radius)&~assigned[start:stop]
                ids=np.flatnonzero(selected)+start
                assigned[ids]=True
                group_rows.extend(ids.tolist());group_nearest.extend(nearest[selected].tolist())
                group_distance.extend(distance[selected].tolist());group_world.extend(world[selected].tolist())
                group_opacity.extend(opacity[selected].tolist())
        anchor_receipts.append(dict(frame_id=f['id'],measured_target_return_denominator=len(supported),
            first_hit_group_rows=len(group_rows),original_static_row_denominator=n,scope_radius_world_m=radius,
            no_hit_interpretation='outside_this_finite_measured_neighbourhood_not_no_error_or_no_owner'))
        if group_rows:
            gid='g%02d_%s'%(len(groups),f['id'])
            groups.append(dict(group_id=gid,original_binding='static',rebind_target='actor_0',
                anchor_frame_id=f['id'],anchor_model_time=f['model_time'],
                original_static_rows=group_rows,
                evidence_ref=f['support_ref'],scope_rule='first_anchor_within_fixed_measured_target_lidar_radius',
                anchor_actor_center_world=f['center'].tolist(),anchor_actor_rotation_world=f['rotation'].tolist()))
            nearest=np.asarray(group_nearest,dtype=np.int64)
            records.append(dict(group_id=gid,static_rows=np.asarray(group_rows,dtype=np.int64),
                nearest_source_lidar_row=f['support_source_rows'][nearest],
                distance_world_m=np.asarray(group_distance,dtype=np.float32),
                world_at_anchor=np.asarray(group_world,dtype=np.float64),
                effective_opacity_at_anchor=np.asarray(group_opacity,dtype=np.float32)))
    candidate=dict(schema='r4_frozen_group_candidates_v1',scene_id=scene_id,scene_ancestor=ancestor,
        target_id='actor_0',checkpoint_static_sha256=checkpoint_sha,
        row_namespace=checkpoint_sha+':static_original_row',static_row_count=n,
        legal_input_role=role['role'],legal_input_role_detail=role['role_detail'],
        scope_rule=dict(radius_world_m=radius,anchors=[f['id'] for f in anchors],
            selected_before_edit=True,first_anchor_disjoint_grouping=True,max_groups=3,
            radius_role='predeclared_local_candidate_scope_not_calibrated_eligibility_threshold',
            positions='native_static_SHM_at_anchor_using_original_cycle_and_velocity',
            lifetime='record_effective_opacity_no_identity_or_coverage_veto',
            untouched_fields='all stored geometry covariance SH opacity lifetime velocity counts environment'),
        groups=groups,anchor_denominators=anchor_receipts,
        limitations=['neighbourhood candidates are not verified ownership or physical Gaussian GT',
                     'candidate support is not native composited contribution/representation coverage',
                     'all other rows fixed; 2^groups domain not full row-level or global domain'])
    validate_candidate(candidate)
    return candidate,records


def ledger_registry(kind,scene_id,candidate,output,enabled=True):
    if kind=='sim':
        name='allrow_observed16_r1' if scene_id=='sim_m0_original10k' else 'fresh_allrow_observed16_r1'
        path=SIM/'source_control_r1'/name/'allrow_history/aggregate.npz'
        if not enabled or not path.is_file():
            return dict(path=str(path),status='optional_diagnostic_unavailable',used_for_candidate_filter=False,
                used_for_eligibility=False,used_for_student_features=False,association_only=True)
        receipt(path,'isolated_historical_GT_diagnostic_mapping_only')
        with np.load(path,allow_pickle=False) as data:
            # 只取原行映射；不取类别贡献、阈值、selected。
            ids=data['static_index'].copy();columns=list(data.files)
        if len(ids)!=candidate['static_row_count'] or not np.array_equal(ids,np.arange(len(ids))):
            raise ValueError('历史来源账本与当前checkpoint原行namespace不符')
        role='privileged_semantic_category_diagnostic_only'
        info=dict(mapping_column='static_index',original_rows=len(ids),columns_available=columns,
                  mapping_covers_original_static_rows=True)
    else:
        path=ROOT/'unknown_detection_v3/source_rows_r1/ids.npy'
        if not enabled or not path.is_file():
            return dict(path=str(path),status='optional_diagnostic_unavailable',used_for_candidate_filter=False,
                used_for_eligibility=False,used_for_student_features=False,association_only=True)
        receipt(path,'isolated_historical_source_subset_mapping_only')
        ids=np.load(path,allow_pickle=False)
        if (ids<0).any() or (ids>=candidate['static_row_count']).any():raise ValueError('真实历史原行越界')
        role='historically_probe_selected_subset_diagnostic_only_not_candidate_filter'
        info=dict(mapping_column='ids.npy',original_rows=len(ids),mapping_covers_original_static_rows=False)
    allids=np.concatenate([np.asarray(g['original_static_rows'],dtype=np.int64) for g in candidate['groups']]) if candidate['groups'] else np.empty(0,np.int64)
    overlap=np.intersect1d(allids,ids)
    np.savez_compressed(output/'historical_ledger_row_association.npz',candidate_rows=allids,
        in_historical_mapping=np.isin(allids,ids),historical_mapping_rows=ids)
    return dict(path=str(path),role=role,used_for_candidate_filter=False,used_for_eligibility=False,
        used_for_student_features=False,association_only=True,candidate_overlap_rows=len(overlap),
        candidate_rows=len(allids),**info)


def candidate_preview(payload,frames,transform,candidate,kind,output,diagnostics=True):
    reference=frames[len(frames)//2]
    ids=np.array(sorted(i for g in candidate['groups'] for i in g['original_static_rows']),dtype=np.int64)
    selection=[]
    # 每组确定性均匀采样；总数不超过100，避免大组遮掉其他来源组。
    quota=100//max(1,len(candidate['groups']))
    for group in candidate['groups']:
        rows=group['original_static_rows']
        selection.extend(rows[i] for i in np.linspace(0,len(rows)-1,min(quota,len(rows)),dtype=int))
    selected=np.asarray(selection,dtype=np.int64)
    # 预览所有组统一到同一个reference时刻，不能把不同anchor坐标混入同一3D场景。
    means,_=static_at(payload,reference['model_time'],0,len(payload[1]))
    inverse=np.linalg.inv(transform)
    world=means[selected].astype(np.float64)@inverse[:3,:3].T+inverse[:3,3]
    verification=dict(status='not_applicable_to_sim_platform',maximum_abs_world_error_m=None)
    if kind=='real' and diagnostics:
        mapping=ROOT/'unknown_detection_v3/source_rows_r1/ids.npy'
        actual=ROOT/'unknown_detection_v3/source_rows_r1'/('%03d.npz'%reference['raw'])
        if mapping.is_file() and actual.is_file():
            receipt(mapping,'historical_mapping_for_coordinate_validation_only')
            receipt(actual,'historical_actual_native_world_positions_for_coordinate_validation_only')
            reference_ids=np.load(mapping,allow_pickle=False)
            common=np.intersect1d(ids,reference_ids)[:100]
            with np.load(actual,allow_pickle=False) as data:
                original_world=data['world'].copy()
        else:
            common=[]
        if len(common):
            predicted=means[common].astype(np.float64)@inverse[:3,:3].T+inverse[:3,3]
            lookup={int(r):i for i,r in enumerate(reference_ids)}
            expected=original_world[[lookup[int(r)] for r in common]]
            error=float(np.max(np.abs(predicted-expected)))
            verification=dict(status='coordinates_match_existing_native_export' if error<1e-4 else 'coordinate_mismatch_do_not_overlay',
                reference=str(actual),row_ids_compared=common.tolist(),compared_count=len(common),
                maximum_abs_world_error_m=error,used_for_candidate_selection=False)
        else:
            verification=dict(status='no_common_rows_do_not_claim_overlay_verified',reference=str(actual),
                compared_count=0,maximum_abs_world_error_m=None,used_for_candidate_selection=False)
    elif kind=='real':
        verification=dict(status='optional_coordinate_diagnostic_disabled',maximum_abs_world_error_m=None)
    group_for={i:g['group_id'] for g in candidate['groups'] for i in g['original_static_rows']}
    preview=dict(schema='r4_readonly_candidate_preview_v1',scene_id=candidate['scene_id'],
        scene_ancestor=candidate['scene_ancestor'],checkpoint_namespace=candidate['row_namespace'],
        checkpoint_static_sha256=candidate['checkpoint_static_sha256'],reference_frame_id=reference['id'],
        reference_model_time=reference['model_time'],coordinate_system='original_world_metres',
        world_to_model=transform.tolist(),coordinate_verification=verification,
        origin_binding='static',target_binding='actor_0',candidate_count=len(ids),preview_count=len(selected),
        all_group_row_mapping_ref='candidate_manifest.json',
        groups=[dict(group_id=g['group_id'],original_static_rows=g['original_static_rows']) for g in candidate['groups']],
        points=[dict(original_static_row=int(i),group_id=group_for[int(i)],reference_world_xyz=p)
                for i,p in zip(selected,world.tolist())],
        read_only=True,edited=False,rendered=False,owner_GT=False,human_verdict=None)
    write(output/'candidate_preview.json',preview)


def available_registry(summaries):
    rows=[dict(scene_id=s['scene_id'],scene_ancestor=s['scene_ancestor'],status='CPU_prepared_this_run',
               task_registration='12 tasks total across 3 configurations',checkpoint_cpu_loaded=True)
          for s in summaries]
    for context in ('4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095'):
        dataset=ROOT/'datasets'/context/'manifest.json'
        event_path=NATIVE/'native_scene_training_v1'/context/'formal30k/checkpoint_events/iteration_030000.json'
        metadata=read(dataset,'available_unprepared_real_scene_metadata') if dataset.exists() else None
        event=read(event_path,'available_unprepared_checkpoint_metadata') if event_path.exists() else None
        rows.append(dict(scene_id='real_context_'+context,scene_ancestor='waymo_context_'+context,
            status='available_unprepared' if metadata and event else 'metadata_incomplete_unprepared',
            dataset_manifest=str(dataset),checkpoint_event=str(event_path),
            checkpoint_paths={k:v['path'] for k,v in event['checkpoints'].items()} if event else {},
            frame_denominator=len(metadata['frames']) if metadata else None,
            target_track=metadata.get('track') if metadata else None,
            checkpoint_cpu_loaded=False,task_registration='not_prepared_this_run',eligibility=None,
            reason='CPU scope limited to first real scene; preserve availability denominator'))
    rows.append(dict(scene_id='legacy_scene0008',scene_ancestor='legacy_scene0008_ancestor_unverified',
        status='available_source_unprepared_checkpoint_identity_unknown',
        source_path=str(SOURCE),source_directory_exists=SOURCE.is_dir(),
        checkpoint_event=None,checkpoint_cpu_loaded=False,task_registration='not_prepared_this_run',
        reason='current native engine source exists; shared-platform checkpoint identity needs explicit mapping; not inferred from directory name'))
    return dict(schema='r4_available_scene_registry_v1',rows=rows,
        cpu_prepared_configurations=len(summaries),cpu_prepared_independent_ancestors=len({s['scene_ancestor'] for s in summaries}),
        additional_available_unprepared_real_contexts=2,legacy_checkpoint_mapping_unknown=1,
        actual_registered_task_denominator=sum(s['task_count'] for s in summaries),
        note='available/unprepared scenes are not silently counted as qualified, trained or CPU-accepted')


def process_scene(spec, output, radius,diagnostics=True):
    output.mkdir()
    scene_start=time.monotonic()
    event=read(spec['event'],'complete_native_checkpoint_event_metadata')
    source=Path(event['source_path']);transform=matrix(source/'transform_matrix.txt','frozen_training_world_to_model_gauge')
    cameras=read(Path(event['model_path'])/'cameras.json','frozen_original_training_cameras')
    payload,loaded=load_checkpoint(event)
    static_sha=next(x['sha256'] for x in loaded if x['name']=='static')
    frames,role,context=observations(event,source,cameras,transform,spec['kind'])
    candidate,mapping=candidates(payload,frames,transform,spec['id'],spec['ancestor'],role,static_sha,radius)
    evidence=legal_evidence(frames,role)
    family=freeze_family(candidate);routes=enumerate_routes(candidate)
    component_hashes={r['name']:r['sha256'] for r in loaded}
    scene_context=dict(component_sha256=component_hashes,
        checkpoint_bundle_sha256=canonical_hash([dict(name=name,sha256=component_hashes[name]) for name in sorted(component_hashes)]),
        event_sha256=HASH_CACHE[str(Path(spec['event']).resolve())]['sha256'],
        gauge_sha256=HASH_CACHE[str((source/'transform_matrix.txt').resolve())]['sha256'],
        camera_sha256=HASH_CACHE[str((Path(event['model_path'])/'cameras.json').resolve())]['sha256'],
        preparation_sha256=HASH_CACHE[str((source/'preparation_manifest.json').resolve())]['sha256'])
    tasks=register_tasks(spec['id'],candidate,evidence,routes,static_sha,scene_context)
    ledger=ledger_registry(spec['kind'],spec['id'],candidate,output,enabled=diagnostics)
    for record in mapping:
        gid=record.pop('group_id')
        np.savez_compressed(output/('row_mapping_'+gid+'.npz'),**record)
    write(output/'checkpoint_cpu_receipt.json',dict(components=loaded,iteration=event['iteration'],
        cpu_loading_complete=True,native_model_restore_complete=False,
        bundle_sha256=scene_context['checkpoint_bundle_sha256'],
        scope='read-only CPU state loading plus layout/boundary samples; CUDA native Scene/render/runtime restoration pending'))
    write(output/'candidate_manifest.json',candidate)
    write(output/'legal_preedit_evidence.json',evidence)
    write(output/'camera_track_context.json',dict(role=role,frames=context,
        world_to_model=transform.tolist(),native_scene_restoration='E0_GPU_gate_pending'))
    write(output/'frozen_H0.json',family)
    write(output/'shared_routes.json',routes)
    write(output/'tasks.json',tasks)
    write(output/'historical_ledger_role.json',ledger)
    candidate_preview(payload,frames,transform,candidate,spec['kind'],output,diagnostics=diagnostics)
    summary=dict(scene_id=spec['id'],scene_ancestor=spec['ancestor'],kind=spec['kind'],
        split='development_existing_seen_ancestor',independent_test=False,
        frame_denominator=len(frames),registered_operations=list(OPERATIONS),task_count=len(tasks),
        original_static_rows=len(payload[1]),candidate_rows=family['candidate_row_count'],groups=len(candidate['groups']),
        possible_shared_routes=family['possible_count'],enumerated_shared_routes=len(routes),
        shared_route_hashes_identical_across_operations=all(
            [a['route_assignment_hash'] for a in tasks[0]['action_evaluations']]==[a['route_assignment_hash'] for a in t['action_evaluations']]
            for t in tasks),
        eligibility_passed_count=0,eligibility_unknown_count=len(tasks),eligibility_denominator=len(tasks),
        eligible_coverage=None,decision_gap=None,representation_residual=None,
        rendering_calls=0,reward_queries=0,training_updates=0,native_execution_status='E0_GPU_gate_pending',
        human_verdict=None,wall_seconds=time.monotonic()-scene_start)
    write(output/'scene_summary.json',summary)
    del payload,frames
    gc.collect()
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radius-world-m',type=float,default=.25)
    parser.add_argument('--skip-historical-diagnostics',action='store_true')
    args=parser.parse_args()
    if args.radius_world_m!=.25:raise ValueError('本次已冻结候选半径0.25m；更改需新预注册与新run')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU入口必须隐藏CUDA')
    started=time.monotonic();usage=resource.getrusage(resource.RUSAGE_SELF)
    specs=[dict(id='sim_m0_original10k',kind='sim',ancestor='procedural_world_dataset_r2_vehicle0',
          event=SIM/'formal_r3_10k/checkpoint_events/iteration_010000.json'),
        dict(id='sim_natural_hybrid8_10k',kind='sim',ancestor='procedural_world_dataset_r2_vehicle0',
          event=SIM/'evidence_reconstruction_r1/formal_hybrid8_10k/checkpoint_events/iteration_010000.json'),
        dict(id='real_first_scene_native30k',kind='real',ancestor='waymo_context_'+CTX,
          event=NATIVE/'native_scene_training_v1'/CTX/'formal30k/checkpoint_events/iteration_030000.json')]
    write(args.output/'run_contract.json',dict(schema='v43_R4_CPU_preparation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_inputs=[dict(s,event=str(s['event'])) for s in specs],
        cpu_threads=1,cuda_visible_devices='',training=False,rendering=False,
        candidate_scope_radius_world_m=.25,anchors='first/middle/last original registered frames',
        maximum_groups=3,choices=['keep','rebind'],defer='query_layer_only',
        frozen_operations=list(OPERATIONS),eligibility_thresholds={m:None for m in ('f_geom','f_view','f_bg','f_motion')},
        scientific_conclusion='none_CPU_engineering_and_candidate_preparation_only',
        split_policy='existing development only; original10k and hybrid8 same ancestor/split; real first scene no new transferable head training',
        failure_ledger_refs=['V82-F01'],failure_ledger_delta='none',human_verdict=None))
    summaries=[process_scene(s,args.output/s['id'],args.radius_world_m,diagnostics=not args.skip_historical_diagnostics) for s in specs]
    write(args.output/'available_scene_registry.json',available_registry(summaries))
    unchanged=[]
    for path,row in INPUTS.items():
        stat=Path(path).stat()
        unchanged.append(stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'])
    if not all(unchanged):raise RuntimeError('输入现场在CPU运行期间发生变化')
    write(args.output/'frozen_inputs.json',dict(inputs=list(INPUTS.values()),
        unchanged_size_and_mtime_after_read=True,hashes='only consumed frozen inputs and historical mapping receipts'))
    now=resource.getrusage(resource.RUSAGE_SELF)
    summary=dict(schema='v43_R4_CPU_preparation_summary_v1',status='CPU_PREPARATION_COMPLETE_GPU_GATE_PENDING',
        scenes=summaries,independent_ancestors=len({s['scene_ancestor'] for s in summaries}),
        reconstruction_configurations=len(summaries),registered_tasks=sum(s['task_count'] for s in summaries),
        rendered_action_evaluations=0,native_render_calls=0,reward_queries=0,training_updates=0,
        eligibility_passed_count=0,eligibility_unknown_count=sum(s['task_count'] for s in summaries),
        eligibility_denominator=sum(s['task_count'] for s in summaries),eligibility_coverage=None,
        frozen_input_count=len(INPUTS),input_unchanged=True,
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cpu_threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        max_rss_mib=now.ru_maxrss/1024,cpu_user_seconds=now.ru_utime-usage.ru_utime,
        cpu_system_seconds=now.ru_stime-usage.ru_stime,wall_seconds=time.monotonic()-started,
        human_verdict=None,scientific_result='unavailable_no_native_render_or_new_learning')
    write(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
