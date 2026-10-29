"""冻结场景一：原生贡献→实测双表面支持→双侧切分初始化对照。

不训练分类器、不把R2删行当语义GT、不改原检查点。切面仅来自实测深度
与独立掩码的支持质心；两个子高斯都保留。此脚本检验初始化，非最终归属。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

CTX = '14705303724557273004_3105_000_3125_000'
TASK = 'WS-V43-EDITABLE-OWNERSHIP-20261030'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def decode_assignment(p):
    text = Path(p).read_text(encoding='utf-8-sig')
    return json.JSONDecoder().raw_decode(text[text.index('=')+1:])[0]

def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def native_adjoint(m,n,cam,kw,maps):
    """24个原生feature含depth/alpha；每块最多22个aux，保留有符号矩。"""
    torch=m.torch
    assert kw['features'].shape[1]==2
    weights=np.empty((len(kw['means3D']),len(maps)),np.float64)
    reference=None;chunks=[]
    for first in range(0,len(maps),22):
        block=maps[first:first+22]
        aux=torch.zeros((len(kw['means3D']),len(block)),device='cuda',requires_grad=True)
        inp={**kw,'features':torch.cat([kw['features'],aux],1)}
        assert inp['features'].shape[1]<=24
        out=m.helpers.raster(cam,n.args,n.static.active_sh_degree,inp,n.bg)
        grad=torch.autograd.grad((out['extra']*torch.from_numpy(block).cuda()).sum(),aux)[0]
        values=grad.detach().cpu().numpy().astype('float64')
        assert np.isfinite(values).all(), 'Nonfinite native signed-moment adjoint'
        rendered={key:out[key].detach() for key in ['rgb','alpha','depth_sum']}
        if reference is None:
            reference=rendered
        else:
            assert all(torch.equal(reference[key],rendered[key]) for key in reference), 'Aux chunks changed native raster outputs'
        alpha=rendered['alpha'].cpu().numpy()[0].astype('float64')
        expected=(block.astype('float64')*alpha).sum(axis=(1,2))
        denominator=np.maximum((np.abs(block.astype('float64'))*alpha).sum(axis=(1,2)),1.)
        closure=np.abs(values.sum(axis=0)-expected)/denominator
        assert np.isfinite(closure).all() and float(closure.max())<2e-4, 'Native per-channel adjoint closure failed'
        weights[:,first:first+len(block)]=values
        chunks.append(dict(first_map=first,aux_channels=len(block),native_feature_channels=int(inp['features'].shape[1]),max_channel_closure=float(closure.max())))
        del out,grad,aux,inp,values,rendered
    return reference,weights,chunks

def moments_map(mask, xyz):
    # 先减actor世界原点，避免大世界坐标的二阶矩相消。
    x,y,z = xyz.transpose(2,0,1)
    return np.stack([mask,mask*x,mask*y,mask*z,mask*x*x,mask*x*y,
                     mask*x*z,mask*y*y,mask*y*z,mask*z*z]).astype('float32')

def conditional(w):
    mass=w[:,0]; mean=w[:,1:4]/np.maximum(mass[:,None],1e-12)
    second=np.zeros((len(w),3,3),np.float64)
    for j,(a,b) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
        second[:,a,b]=second[:,b,a]=w[:,4+j]/np.maximum(mass,1e-12)
    cov=second-mean[:,:,None]*mean[:,None,:]
    return mass,mean,cov

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-contract',type=Path,required=True)
    p.add_argument('--code-dir',type=Path,required=True)
    p.add_argument('--scope-file',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--raws',type=int,nargs='+',default=[98,113,114])
    p.add_argument('--split-max',type=int,default=8,help='每帧最多切分的父高斯数，范围0..8')
    a=p.parse_args()
    if not 0<=a.split_max<=8:
        p.error('--split-max must be between 0 and 8 per frame')
    auth=read(a.authorization)
    assert auth['task']==TASK and auth['context']==CTX and auth['allow_gpu_probe']
    assert not auth['allow_classifier_training'] and not auth['resume_old_queue']
    assert not a.output.exists(), 'Choose fresh output; preserve prior attempts'
    c=read(a.input_contract); assert c['context']==CTX
    assert c['branch_sizes']==dict(static=918850,actor=6241), 'Input checkpoint population mismatch'
    assert set(a.raws)<=set(range(77,126))
    ownership_path=a.input_contract.parent/c['source_assets']['ownership']['path']
    assert sha(ownership_path)==c['source_assets']['ownership']['sha256'], 'Ownership differs from frozen input contract'
    evidence=decode_assignment(ownership_path)
    assert evidence['context']==CTX and evidence['branch_sizes']==c['branch_sizes'], 'Ownership namespace/population mismatch'
    assert evidence['static_namespace']==c['static_namespace'], 'Ownership static checkpoint namespace mismatch'
    assert evidence['raws']==list(range(77,126)), 'Ownership observation frame binding differs'
    assert c['static_namespace']==c['static_checkpoint_ref']+'@30000:static', 'Static checkpoint reference differs from namespace'
    assert sha(c['actor_checkpoint_ref'])==c['actor_checkpoint_sha256'], 'Actor checkpoint differs from frozen input contract'
    sys.path.insert(0,str(a.code_dir))
    m=importlib.import_module('train_actor')
    source_maps=importlib.import_module('probe_target_unknown').source_maps
    spec=importlib.util.spec_from_file_location('bounded_scope',a.scope_file)
    scope_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope_module)
    from two_sided_split import split_gaussian, covariance_from_quaternion_scale, covariance_to_quaternion_scale
    torch=m.torch
    assert torch.cuda.is_available(), 'Need real native CUDA rasterizer'
    torch.set_num_threads(1)
    a.output.mkdir(parents=True)
    start=time.monotonic();records=[]
    def status(phase,**kw):
        write(a.output/'status.json',dict(task=TASK,phase=phase,raws=a.raws,training_updates=0,**kw))
    def stopped():
        assert not (m.ROOT/'USER_STOP').exists(), 'User stop'
        assert read(a.authorization)['allow_gpu_probe'], 'Probe authorization withdrawn'
    def arr(x):return x.detach().cpu().numpy()
    def save(p,x):
        p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.round(np.clip(arr(x).transpose(1,2,0),0,1)*255).astype('uint8')).save(p)
    status('loading')
    with importlib.import_module('gpu_slot').exclusive():
        stopped()
        ds=m.ROOT/'datasets'/CTX;manifest=read(ds/'manifest.json');frames=manifest['frames']
        assert manifest['context']==CTX and [f['raw'] for f in frames]==list(range(77,126)), 'Dataset frame binding differs'
        actor=m.Actor(np.load(ds/'seeds.npz'))
        ck=torch.load(c['actor_checkpoint_ref'],map_location='cuda')
        assert ck['step']==10000
        actor.load_state_dict(ck['actor']);del ck
        n=m.NativeMotion(Path(c['static_checkpoint_ref']),30000)
        ns=len(n.static._xyz); assert ns==918850 and len(actor.xyz)==6241
        actor_before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        for raw in a.raws:
            stopped();i=raw-77;f=frames[i];assert f['raw']==raw
            expected_frame=c['groups']['v43_actor_10000']['frames'][str(raw)]
            assert expected_frame['raw']==raw and expected_frame['camera_index']==i, 'Input contract camera index differs'
            assert str(f['frame_id'])==str(expected_frame['frame_id'])==str(n.records[i]['img_name']), 'Observed LiDAR/body and native camera frame_id differ'
            assert np.isclose(float(f['model_time']),float(expected_frame['model_time']),rtol=0,atol=1e-12), 'Observed/native model_time differs from frozen input contract'
            cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{})
            with torch.no_grad():
                st=m.prepare_rasterization(cam,n.static,n.args)
                st['means2D']=torch.zeros_like(st['means3D']);st['features']=st['features'][:,:2]
                ak=actor.kwargs(n,cam,f);ak['features']=ak['features'][:,:2]
                full=m.helpers.merge_rows(st,ak)
                full={k:v.detach() if torch.is_tensor(v) else v for k,v in full.items()}
                st={k:v.detach() if torch.is_tensor(v) else v for k,v in st.items()}
                env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            sem=np.load(m.ROOT/'segformer_b0_attempt3'/CTX/('%03d.npz'%raw))
            ob,body,safe,depth,_=source_maps(n.source,f,ds,sem)
            scope=scope_module.build_scope(body,context=CTX,raw=raw)
            assert scope['status']=='ok'
            # 独立输入只使用观测mask/分割/实测LiDAR。R2仅作为比较集合。
            target_valid=binary_erosion(body,iterations=2)&np.isfinite(ob['depth'])
            road_valid=safe&(sem['labels']==1)&np.isfinite(depth)
            K={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in (n.source/'calib'/(f['frame_id']+'.txt')).read_text().splitlines()}['P0'].reshape(3,4)[:,:3]/4
            origin=np.asarray(f['actor_center_world'])
            yy,xx=np.indices(body.shape)
            def backproject(z):
                z=np.where(np.isfinite(z),z,0)
                camera=np.stack([(xx+.5-K[0,2])/K[0,0]*z,(yy+.5-K[1,2])/K[1,1]*z,z],-1)*n.scale
                h=np.concatenate([camera,np.ones((*z.shape,1))],-1)
                model=h@np.linalg.inv(arr(cam.world_view_transform))
                world=(model[...,:3]-n.T[:3,3])@np.linalg.inv(n.T[:3,:3]).T
                return world-origin
            target_xyz=backproject(ob['depth']);road_xyz=backproject(depth)
            maps=np.concatenate([np.stack([np.ones_like(body),body,safe,scope['allowed'],scope['user_contact_query']]),
                                 moments_map(target_valid,target_xyz),moments_map(road_valid,road_xyz)]).astype('float32')
            # 每个状态重新计算排序/透射率；不借用剥离前贡献。
            state_records={};frame_path=a.output/('%03d'%raw);frame_path.mkdir()
            baseline=None;full_w=None
            for state in ['intact','actor_removed','r2_removed']:
                kw=dict(full if state=='intact' else st)
                if state=='r2_removed':
                    kw['opacities']=kw['opacities'].clone()
                    ids=torch.as_tensor(evidence['frames'][str(raw)]['static']['final_removed_ids'],device='cuda',dtype=torch.long)
                    kw['opacities'][ids]=0
                # 每块<=22 aux；XYZ有符号矩不按非负类别质量裁剪。
                r,w,adjoint_chunks=native_adjoint(m,n,cam,kw,maps)
                closure=float(abs(w[:,0].sum()-float(r['alpha'].sum()))/max(float(r['alpha'].sum()),1))
                assert closure<2e-4
                rgb=r['rgb'].detach()+(1-r['alpha'].detach())*env
                if state=='intact':
                    baseline=rgb
                    ref=a.input_contract.parent/c['groups']['v43_actor_10000']['frames'][str(raw)]['baseline_file']
                    pixels=np.round(np.clip(arr(rgb).transpose(1,2,0),0,1)*255).astype('uint8')
                    assert np.array_equal(pixels,np.asarray(Image.open(ref).convert('RGB'))), 'Native baseline mismatch'
                    full_w=w[:ns].copy()
                save(frame_path/(state+'.png'),rgb)
                save(frame_path/(state+'_gaussian.png'),r['rgb'].detach())
                save(frame_path/(state+'_environment.png'),(1-r['alpha'].detach())*env)
                active=(w[:ns,1]>.001)|(w[:ns,4]>.001)
                active_ids=np.flatnonzero(active)
                np.savez_compressed(frame_path/(state+'_evidence.npz'),ids=active_ids.astype('int32'),weights=w[active_ids].astype('float32'),
                    alpha=arr(r['alpha']),gaussian_rgb=arr(r['rgb']),environment_rgb=arr(env),rgb=arr(rgb))
                state_records[state]=dict(adjoint_closure=closure,adjoint_chunks=adjoint_chunks,active_static_rows=len(active_ids),
                    target_static_mass=float(w[:ns,1].sum()),contact_static_mass=float(w[:ns,4].sum()),
                    environment_min=float(env.min()),environment_max=float(env.max()))
                del r,w
            mt,ct,vt=conditional(full_w[:,5:15]);mr,cr,vr=conditional(full_w[:,15:25])
            xyz_model=arr(st['means3D']);h=np.column_stack([xyz_model,np.ones(ns)])
            clip=h@arr(cam.full_proj_transform);screen=(clip[:,:2]/clip[:,3:]+1)*np.array([240,160])-.5
            z=(h@arr(cam.world_view_transform))[:,2]
            eligible=scope_module.eligible_from_screen(screen,z,scope['allowed'])
            # 只是双实测支持候选；空间可分还需跨帧检验，绝不伪称owner真值。
            dual=(mt>1e-4)&(mr>1e-4)&eligible
            ids=np.flatnonzero(dual);rank=np.minimum(mt[ids],mr[ids]);ids=ids[np.argsort(-rank,kind='stable')]
            np.savez_compressed(frame_path/'dual_support.npz',ids=ids.astype('int32'),target_mass=mt[ids],road_mass=mr[ids],
                target_centroid_world=ct[ids]+origin,road_centroid_world=cr[ids]+origin,
                target_covariance_world=vt[ids],road_covariance_world=vr[ids],parent_xyz_model=xyz_model[ids])
            split_rows=[];skips=[]
            for gid in ids[:a.split_max]:
                try:
                    direction=ct[gid]-cr[gid];separation=float(np.linalg.norm(direction))
                    if separation<.05:
                        skips.append(dict(row=int(gid),reason='support_centroids_not_separated',separation_m=separation));continue
                    # Convert world normal to normalized-model plane, world→model affine.
                    normal_world=direction/separation;point_world=(ct[gid]+cr[gid])/2+origin
                    normal_model=np.linalg.inv(n.T[:3,:3]).T@normal_world
                    normal_model/=np.linalg.norm(normal_model)
                    point_model=n.T[:3,:3]@point_world+n.T[:3,3]
                    cov=covariance_from_quaternion_scale(arr(st['rotations'][gid]),arr(st['scales'][gid]))
                    result=split_gaussian(xyz_model[gid],cov,float(st['opacities'][gid,0]),normal_model,float(normal_model@point_model),parent_id=int(gid))
                    # Refuse unsafe tails/opacity; retain parent rather than clip or drop a side.
                    if not result['native_safe']:
                        skips.append(dict(row=int(gid),reason='split_initialization_unsafe',details=result['warnings']));continue
                    split_rows.append(dict(row=int(gid),separation_m=separation,result=result))
                except (ValueError,ArithmeticError) as error:
                    skips.append(dict(row=int(gid),reason='split_initialization_error_parent_retained',error_type=type(error).__name__,details=str(error)))
            write(frame_path/'split_proposals.json',dict(candidates=len(ids),evaluated=min(len(ids),a.split_max),
                retained=len(split_rows),skipped=skips,proposals=split_rows,owner_labels_are_hypotheses=True))
            split_result=dict(applied_rows=0,reason='no_safe_proposal')
            if split_rows:
                # All children remain static in reconstruction. No unsupported car motion is assigned.
                parents=np.array([q['row'] for q in split_rows]);kwargs=dict(full)
                kwargs['opacities']=full['opacities'].clone();kwargs['opacities'][torch.as_tensor(parents,device='cuda')]=0
                children=[]
                for proposal in split_rows:
                    gid=proposal['row']
                    for child in proposal['result']['children']:
                        q,s=covariance_to_quaternion_scale(np.asarray(child['covariance']))
                        children.append((gid,child,q,s))
                source=torch.as_tensor([x[0] for x in children],dtype=torch.long,device='cuda')
                childkw={k:v[source].clone() if torch.is_tensor(v) and v.shape[0]==len(full['means3D']) else v for k,v in full.items()}
                childkw['means3D']=torch.as_tensor(np.array([x[1]['mean'] for x in children]),dtype=torch.float32,device='cuda')
                childkw['means2D']=torch.zeros_like(childkw['means3D'])
                childkw['opacities']=torch.as_tensor([[x[1]['opacity_candidate']] for x in children],dtype=torch.float32,device='cuda')
                childkw['rotations']=torch.as_tensor(np.array([x[2] for x in children]),dtype=torch.float32,device='cuda')
                childkw['scales']=torch.as_tensor(np.array([x[3] for x in children]),dtype=torch.float32,device='cuda')
                cz=torch.cat([childkw['means3D'],torch.ones_like(childkw['means3D'][:,:1])],1)@cam.world_view_transform[:,2:3]
                childkw['features']=torch.cat([cz,torch.ones_like(cz)],1)
                splitkw=m.helpers.merge_rows(kwargs,childkw)
                with torch.no_grad():
                    rr=m.helpers.raster(cam,n.args,n.static.active_sh_degree,splitkw,n.bg)
                    rgb=rr['rgb']+(1-rr['alpha'])*env
                    save(frame_path/'split_both_retained.png',rgb)
                    delta=(rgb-baseline).abs().mean(0)
                    np.savez_compressed(frame_path/'split_float.npz',rgb=arr(rgb),alpha=arr(rr['alpha']),delta=arr(delta),parents=parents)
                split_result=dict(applied_rows=len(parents),children=len(children),
                    mean_abs_rgb_change=float(delta.mean()),outside_allowed_mean_abs_rgb_change=float(delta[torch.from_numpy(~scope['allowed']).cuda()].mean()),
                    max_abs_rgb_change=float((rgb-baseline).abs().max()),no_edit_quality_accepted=False)
            record=dict(raw=raw,states=state_records,target_lidar_pixels=int(target_valid.sum()),road_lidar_pixels=int(road_valid.sum()),
                dual_measured_support_rows=len(ids),split=split_result)
            records.append(record);write(a.output/'summary.json',dict(task=TASK,status='partial',records=records,training_updates=0))
            status('probing',finished=len(records),last_raw=raw)
            print(json.dumps(record),flush=True)
            del full_w,st,ak,full
        assert n.unchanged()
        assert all(torch.equal(actor_before[k],v.detach().cpu()) for k,v in actor.state_dict().items())
        summary=dict(task=TASK,context=CTX,status='complete',records=records,training_updates=0,
            source_checkpoints_unchanged=True,semantic_ground_truth=False,oracle_executed=False,
            split_is_initialization_only=True,edit_benefit_proven=False,quality_accepted=False,human_verdict=None,
            wall_seconds=time.monotonic()-start,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            failure_ledger_refs=['V82-F01'],failure_ledger_delta='none')
        write(a.output/'summary.json',summary);status('complete',wall_seconds=summary['wall_seconds'])

if __name__=='__main__':main()
