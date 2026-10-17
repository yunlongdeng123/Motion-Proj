"""R15-O01：真实reset克隆、完整alpha*T漏斗、静态道路参考覆盖；不改checkpoint。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--performance',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.path.insert(0,args.source)
    if not json.loads(Path(args.performance).read_text()).get('geometry_gate',{}).get('passed'):
        raise RuntimeError('性能门未通过，停止后续GPU实验')
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset,clean_reference
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import cuda_raster
    from shadow_rebuild.losses import pixel_valid
    from shadow_rebuild.math3d import linear_to_srgb
    torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(run_id='R15-O01-ENTRY-AUDIT',status='RUNNING',training_updates=0,
                variants={},BUILD_only=True,QUERY_used=False)
    def save():
        p=out/'owner_entry.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(p)
    try:
        base=Path(args.base);saved=torch.load(base/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
        raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
        data=Dataset(base/'data','cuda',True);assert saved['input_contract']==data.manifest
        model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
        field=model.field;n=len(field.xyz);owner=field.owner.cpu().numpy();tag=field.source_candidate_owner.cpu().numpy()
        original_opacity=field.opacity_logit.detach().clone();arrays={}
        for variant in ['current','cloned_reset','delete37_current']:
            with torch.no_grad():
                field.opacity_logit.copy_(original_opacity)
                if variant=='cloned_reset':field.opacity_logit.copy_(torch.logit(field.opacity_logit.sigmoid().clamp_max(.01)))
            sums=np.zeros((n,5),np.float64);hits=np.zeros((n,4),np.uint16);perframe=[]
            start=time.monotonic();result['active_variant']=variant;save()
            for j,index in enumerate(data.train_indices):
                frame=data.frame(index);assert frame.split=='train'
                with torch.no_grad():
                    g=field.evaluate(frame.poses,frame.valid,frame.camera.center,{37:{'delete':True}} if variant.startswith('delete37') else None)
                    body=frame.masks&pixel_valid(frame);union=body.any(-1)
                    road=frame.road_certified&(frame.depth>0)&~union
                    domains=torch.stack((body[...,6],body[...,37],union,road,torch.ones_like(union)),-1).float()
                feature=torch.ones(len(g['xyz']),5,device='cuda',requires_grad=True)
                image=cuda_raster(frame.camera,g['xyz'],g['cov'],g['opacity'],feature,10)
                contribution=torch.autograd.grad((image*domains).sum(),feature)[0].detach().cpu().numpy()
                rows=g['rows'].cpu().numpy();sums[rows]+=contribution;hits[rows]+=(contribution[:,:4]>.02).astype(np.uint16)
                perframe.append(dict(frame_id=frame.frame_id,body6_pixels=int(body[...,6].sum()),body37_pixels=int(body[...,37].sum()),
                    road_pixels=int(road.sum()),a6_road_mass=float(contribution[owner[rows]==6,3].sum()),
                    a37_road_mass=float(contribution[owner[rows]==37,3].sum())))
                if (j+1)%10==0:result['progress']=dict(variant=variant,frames=j+1,total=len(data.train_indices));save()
                del image,feature,contribution,g
            records=[]
            for actor,col in [(6,0),(37,1)]:
                selected=owner==actor;semantic=np.maximum(sums[:,2]+sums[:,3],1e-9)
                repeated_road=(hits[:,3]>=3)&(sums[:,3]>=.5)
                repeated_body=(hits[:,col]>=3)&(sums[:,col]>=.5)
                road=selected&repeated_road&(sums[:,3]/semantic>.8)&(sums[:,col]/semantic<.05)&~repeated_body
                mixed=selected&repeated_body&(hits[:,3]>=3)&(sums[:,3]>=.2)
                static=owner==-1;body=static&repeated_body&(sums[:,col]/np.maximum(sums[:,2],1e-9)>.8)
                records.append(dict(actor=actor,owner_points=int(selected.sum()),road_mass=float(sums[selected,3].sum()),
                    visible_points=int((selected&(sums[:,4]>.5)).sum()),positive_road_mass_ge05=int((selected&(sums[:,3]>=.5)).sum()),
                    repeated_road_ge3=int((selected&repeated_road).sum()),repeated_body=int((selected&repeated_body).sum()),
                    road_final=int(road.sum()),mixed_final=int(mixed.sum()),
                    static_total=int(static.sum()),static_source_tag=int((static&(tag==actor)).sum()),
                    static_repeated_body_without_tag_gate=int(body.sum()),
                    static_repeated_body_with_tag_gate=int((body&(tag==actor)).sum())))
            arrays[variant]=(sums,hits)
            np.savez_compressed(out/f'{variant}_contributions.npz',sums=sums,hits=hits)
            result['variants'][variant]=dict(actors=records,seconds=time.monotonic()-start,frames=perframe)
            save();print(json.dumps(dict(variant=variant,seconds=time.monotonic()-start,actors=records)),flush=True)
        with torch.no_grad():field.opacity_logit.copy_(original_opacity)
        # tag-less静态候选也必须经过独立真实道路/身体证据，绝不直接改owner。
        sums,hits=arrays['delete37_current']
        candidates=np.flatnonzero((owner==-1)&(sums[:,1]>=.5)&(hits[:,1]>=3))
        xyz=field.xyz.detach().cpu().numpy();roadbank=dict(np.load(base/'actual_world_road_support.npz'))
        distance,near=cKDTree(roadbank['xyz']).query(xyz[candidates],workers=2)
        repeated=roadbank['support_frames'][near]>=3
        road_cert=(distance<.06)&repeated
        proposal=candidates[road_cert]
        result['a37_reference_coverage']=dict(static_delete_body_contribution_candidates=len(candidates),
            measured_world_road_within_006=int((distance<.06).sum()),repeated_world_road=len(proposal),
            existing_road_flag=int(field.road[torch.as_tensor(proposal,device='cuda')].sum()),
            existing_reference_positive=int((field.reference_confidence[torch.as_tensor(proposal,device='cuda')]>0).sum()))
        # 真实跨时刻参考，严格沿用深度一致/可见/车身guard；缺失观测仍为unknown。
        samples=np.zeros((len(proposal),len(data.train_indices),3),np.float32)
        valid=np.zeros(samples.shape[:2],bool);view_direction=np.zeros_like(samples)
        data.device='cpu'
        for j,index in enumerate(data.train_indices):
            frame=data.frame(index);uv,z=frame.camera.project(torch.from_numpy(xyz[proposal]))
            ij=torch.floor(uv+.5).long().numpy();z=z.numpy();h,w=frame.depth.shape
            good=(z>.1)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            take=np.flatnonzero(good);yy,xx=ij[take,1],ij[take,0]
            guard=binary_dilation(frame.masks.any(-1).numpy(),iterations=2)
            okay=(frame.depth.numpy()[yy,xx]>0)&(np.abs(frame.depth.numpy()[yy,xx]-z[take])<.2)&~guard[yy,xx]&frame.road.numpy()[yy,xx]
            take,yy,xx=take[okay],yy[okay],xx[okay]
            samples[take,j]=linear_to_srgb(frame.rgb).numpy()[yy,xx];valid[take,j]=True
            vector=xyz[proposal]-frame.camera.center.numpy();vector/=np.maximum(np.linalg.norm(vector,axis=-1,keepdims=True),1e-8)
            view_direction[:,j]=vector
        target,confidence=clean_reference(samples,valid)
        use=confidence>0
        result['a37_reference_coverage'].update(observed_at_least3=int((valid.sum(-1)>=3).sum()),
            certified_clean_reference_points=int(use.sum()),candidate_denominator=len(candidates),
            reference_definition='existing bright-cluster prior on real BUILD RGB, not clean-road ground truth')
        np.savez_compressed(out/'a37_static_reference.npz',rows=proposal[use],target_srgb=target[use],
            confidence=confidence[use],samples_srgb=samples[use],observed=valid[use],view_direction=view_direction[use],
            source_train_frame_indices=np.asarray(data.train_indices))
        result.update(status='COMPLETE',original_checkpoint_modified=False,
            reset_interpretation='same4k frozen geometry, actualopacity reset clone; not recovered pre-reset3000',
            failure_ledger_refs=['V82-F06'],human_verdict=None)
        save();print(json.dumps(result['a37_reference_coverage']),flush=True)
    except BaseException as e:
        result.update(status='ERROR',error=repr(e));save();raise


if __name__=='__main__':main()
