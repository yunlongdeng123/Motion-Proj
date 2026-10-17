"""R15-N02：冻结4k、多帧/全接收点/粗筛/真实梯度子集验收。"""
import argparse,json,math,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from factor_integral import factor_transmittance,factor_integral
from packed_shadow import Factors,factors_for_geometry,transmittance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--base',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.path.insert(0,args.source)
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.math3d import disk_directions
    import shadow_rebuild.tracing as old
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(run_id='R15-N02-GPU-NUMERIC',status='RUNNING',training_updates=0,frames=[],gradient_cases=[],tf32=False)
    def save():
        p=out/'numeric_full.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(p)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-start
    try:
        base=Path(args.base);saved=torch.load(base/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
        raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
        data=Dataset(base/'data','cuda',True);assert saved['global_step']==4000 and saved['input_contract']==data.manifest
        model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
        primary=None
        for raw_id in [107,93,105,117,121,137]:
            index=next(i for i,r in enumerate(data.rows) if int(r['frame_id'])%1000==raw_id)
            frame=data.frame(index);assert frame.split=='train'
            with torch.no_grad():
                g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
                factors=factors_for_geometry(model,frame,g)
                h,w=p['alpha'].shape;s=cfg.shadow_train_stride
                ys=torch.linspace(0,h-1,math.ceil(h/s)+1,device='cuda').round().long()
                xs=torch.linspace(0,w-1,math.ceil(w/s)+1,device='cuda').round().long()
                yy,xx=torch.meshgrid(ys,xs,indexing='ij');grid=(yy*w+xx).flatten()
                active=p['road_alpha'].flatten()[grid]>1e-5
                all_o=(p['road_position'].reshape(-1,3)[grid]+p['road_normal'].reshape(-1,3)[grid]*cfg.ray_bias_m)[active]
                use=torch.arange(len(all_o),device='cuda') if raw_id==107 else torch.linspace(0,len(all_o)-1,min(16,len(all_o)),device='cuda').round().long()
                origins=all_o[use]
                sun=disk_directions(model.light.direction(),torch.deg2rad(5*model.light.radius_logit.sigmoid()),5)
                o=origins.repeat(5,1);d=sun[:,None,:].expand(5,len(origins),3).reshape(-1,3)
                reference,seconds=timed(lambda:factor_transmittance(o,d,factors.means,factors.rotation,factors.scales,factors.opacity,4096))
                record=dict(raw=raw_id,split=frame.split,receiver_count=len(all_o),selected_receiver_indices=use.tolist(),
                    reference_seconds=seconds,cases=[],reference_range=[float(reference.min()),float(reference.max())])
                result['frames'].append(record);save()
                for dtype in [torch.float64,torch.float32]:
                    for block in [256,512]:
                        stats={}
                        value,seconds=timed(lambda:transmittance(o,d,factors,block,256,dtype,False,10.,stats))
                        error=(value.double()-reference).abs();finite=bool(torch.isfinite(value).all())
                        case=dict(dtype=str(dtype),block_size=block,pair_batch=256,seconds=seconds,
                            finite=finite,max_abs_T_error=float(error.max()),mean_abs_T_error=float(error.mean()),
                            passed=finite and bool((error<=1e-5).all()),stats=stats)
                        record['cases'].append(case);save()
                if raw_id==107:
                    # 与原诊断的物理FP64协方差参照交叉核查；它仍可能受病态求逆影响。
                    cov=(factors.rotation*factors.scales.square()[:,None,:])@factors.rotation.transpose(-1,-2)
                    precision=torch.linalg.inv(cov)
                    amp=-torch.log1p(-factors.opacity.double().clamp(1e-6,1-1e-6))/(math.sqrt(2*math.pi)*factors.scales.amin(-1))
                    def physical_reference():
                        tau=torch.zeros(len(o),device='cuda',dtype=torch.float64)
                        for start in range(0,len(precision),4096):
                            tau+=old._precision_integral(o.double(),d.double(),factors.means[start:start+4096].double(),
                                precision[start:start+4096],amp[start:start+4096]).sum(-1)
                        return torch.exp(-tau.clamp_max(80))
                    physical,secs=timed(physical_reference)
                    err=(physical-reference).abs()
                    record['physical_FP64_covariance_crosscheck']=dict(seconds=secs,max_abs_T_error=float(err.max()),
                        passed=bool((err<=1e-5).all()))
                    del cov,precision,amp
                    candidate=transmittance(o,d,factors,256,256,torch.float64,False)
                    grid_delta=grid.new_zeros(len(grid),dtype=torch.float64)
                    grid_delta[active]=(candidate.double()-reference).reshape(5,-1).mean(0)
                    delta=F.interpolate(grid_delta.reshape(1,1,*yy.shape),size=(h,w),mode='bilinear',align_corners=True)[0,0]
                    cosine=(p['road_normal']*model.light.direction()).sum(-1).clamp_min(0)
                    rgb_delta=p['road_alpha'][...,None]*p['road_albedo']*(1-model.light.ambient_logit.sigmoid())*cosine[...,None]*model.light.strength_logit.sigmoid()*delta[...,None]
                    record['FP64_packed_linear_RGB_max_error']=float(rgb_delta.abs().max())
                    primary=(factors,o,d,reference)
                    save()
                del g,p
        # 真实贡献排序 + 条件数/opacity分层，防止用全零梯度子集敷衍。
        factors,o,d,reference=primary
        with torch.no_grad():
            ray_order=torch.argsort(reference)[:10]
            oo=o[ray_order];dd=d[ray_order]
            score=torch.zeros(len(factors.means),device='cuda',dtype=torch.float64)
            for start in range(0,len(score),4096):
                end=start+4096
                score[start:end]=factor_integral(oo,dd,factors.means[start:end],factors.rotation[start:end],
                    factors.scales[start:end],factors.opacity[start:end]).sum(0)
            condition=(factors.scales.amax(-1)/factors.scales.amin(-1)).square()
            selected=torch.unique(torch.cat((score.topk(3072).indices,condition.topk(512).indices,factors.opacity.topk(512).indices)))
            leaf=[x.detach().double().clone().requires_grad_() for x in
                  [oo,dd,factors.means[selected],factors.rotation[selected],factors.scales[selected].log(),torch.logit(factors.opacity[selected].double())]]
        owners=factors.owners[selected]
        ref=factor_transmittance(leaf[0],leaf[1],leaf[2],leaf[3],leaf[4].exp(),leaf[5].sigmoid(),4096)
        ref_grad=torch.autograd.grad(ref.sum(),leaf)
        for dtype in [torch.float64,torch.float32]:
            bundle=Factors(leaf[2],leaf[3],leaf[4].exp(),leaf[5].sigmoid(),owners)
            pred=transmittance(leaf[0],leaf[1],bundle,256,128,dtype,True)
            grads=torch.autograd.grad(pred.sum(),leaf)
            metrics=[]
            for name,a,b in zip(['origins','directions','means','world_rotation','log_scales','opacity_logit'],grads,ref_grad):
                norm=float(b.norm());diff=float((a-b).norm());maximum=float((a-b).abs().max())
                rel=diff/max(norm,1e-8);finite=bool(torch.isfinite(a).all())
                metrics.append(dict(parameter=name,reference_norm=norm,relative_L2=rel,max_absolute_error=maximum,
                    nonzero_reference_entries=int((b.abs()>1e-10).sum()),
                    passed=finite and (rel<=1e-3 if norm>1e-6 else maximum<=1e-5)))
            result['gradient_cases'].append(dict(dtype=str(dtype),gaussians=len(selected),rays=len(oo),
                max_abs_T_error=float((pred-ref).abs().max()),parameters=metrics,passed=all(x['passed'] for x in metrics)))
            save()
        by_dtype={}
        for dtype in ['torch.float64','torch.float32']:
            numeric=all(c['passed'] for fr in result['frames'] for c in fr['cases'] if c['dtype']==dtype)
            gradient=next(c['passed'] for c in result['gradient_cases'] if c['dtype']==dtype)
            cross=all(fr.get('physical_FP64_covariance_crosscheck',{'passed':True})['passed'] for fr in result['frames'])
            by_dtype[dtype]=dict(numeric_pass=numeric,gradient_pass=gradient,reference_crosscheck_pass=cross,
                                passed=numeric and gradient and cross)
        result.update(status='COMPLETE',dtype_verdicts=by_dtype,
            selected_dtype='torch.float32' if by_dtype['torch.float32']['passed'] else ('torch.float64' if by_dtype['torch.float64']['passed'] else None),
            scope='6 BUILD frames, full raw107 grid and16 rays each other frame; no full-scene training yet',
            failure_ledger_refs=['V82-F05'],human_verdict=None)
        save();print(json.dumps({k:v for k,v in result.items() if k not in ['frames','gradient_cases']}),flush=True)
    except BaseException as e:
        result.update(status='ERROR',error=repr(e));save();raise


if __name__=='__main__':main()
