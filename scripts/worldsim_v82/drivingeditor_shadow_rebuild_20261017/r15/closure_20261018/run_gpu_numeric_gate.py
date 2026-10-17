"""R15 第一轮冻结4k数值实验。默认只打印计划；本轮未执行 GPU 路径。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true',help='用户开卡后才可显式执行')
    ap.add_argument('--source',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--frame-id',default='0008107')
    ap.add_argument('--max-rays',type=int,default=32)
    args=ap.parse_args()
    if args.max_rays<5:
        ap.error('至少包括已知5条失败射线')
    plan={'run_id':'R15-N01-GPU-PILOT','status':'WAITING_FOR_USER_GPU',
          'execute':args.execute,'frame_id':args.frame_id,'max_rays':args.max_rays,
          'checkpoint_step':4000,'training_updates':0,'production_code_modified':False,
          'scope':'unculled factor-reference pilot only; not full numerical or performance acceptance'}
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    # 所有 CUDA 接触都在显式 execute 后；脚本不启动训练/控制器。
    import torch
    from factor_integral import factor_transmittance
    sys.path.insert(0,str(Path(args.source).resolve()))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.math3d import disk_directions,quaternion_matrix
    import shadow_rebuild.tracing as tracing
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=Path(args.output)
    if out.exists():
        raise FileExistsError('使用新的输出目录，禁止覆盖既有实验')
    out.mkdir(parents=True)
    result=dict(plan,status='RUNNING',torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(0),tf32=False,cases=[])
    def save():
        p=out/'numeric_pilot.json';tmp=p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        tmp.replace(p)
    def measure(label,function):
        torch.cuda.synchronize();start=time.perf_counter()
        value=function()
        torch.cuda.synchronize()
        record={'variant':label,'seconds':time.perf_counter()-start,
                'nonfinite':int((~torch.isfinite(value)).sum())}
        if record['nonfinite']==0:
            record['T']=value.cpu().tolist()
        result['cases'].append(record);save();return value
    save()
    try:
        with torch.no_grad():
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if saved['global_step']!=4000:raise ValueError('要求冻结4k checkpoint')
            raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights'])
            cfg=Config(**raw).validate()
            if cfg.sun_samples!=5:raise ValueError('必须保留5个太阳方向')
            data=Dataset(args.data,'cuda',True)
            if saved['input_contract']!=data.manifest:raise ValueError('输入契约不符')
            indices=[i for i,row in enumerate(data.rows) if row['frame_id']==args.frame_id]
            if len(indices)!=1:raise ValueError('frame_id必须唯一')
            model=SceneModel(seed_from_state(saved['state']),cfg).cuda()
            model.load_state_dict(saved['state']);model.requires_grad_(False)
            frame=data.frame(indices[0]);g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
            h,w=p['alpha'].shape;s=cfg.shadow_train_stride
            ys=torch.linspace(0,h-1,math.ceil(h/s)+1,device='cuda').round().long()
            xs=torch.linspace(0,w-1,math.ceil(w/s)+1,device='cuda').round().long()
            yy,xx=torch.meshgrid(ys,xs,indexing='ij');grid=(yy*w+xx).flatten()
            active=p['road_alpha'].flatten()[grid]>1e-5
            pos=p['road_position'].reshape(-1,3)[grid][active]
            normal=p['road_normal'].reshape(-1,3)[grid][active]
            nr=len(pos)
            if args.frame_id=='0008107' and nr!=317:
                raise ValueError('原始raw107接收网格不是317，必须先重新核对失败射线映射')
            fixed=[79,209,50,49,283] if args.frame_id=='0008107' else []
            if any(i>=nr for i in fixed):raise ValueError('原始失败射线索引已失配')
            # 其余样本固定均匀覆盖接收网格，不按候选误差选样。
            wanted=min(args.max_rays,nr)
            for i in torch.linspace(0,nr-1,max(wanted,1)).round().long().tolist()+list(range(nr)):
                if len(fixed)>=wanted:break
                if i not in fixed:fixed.append(i)
            ids=torch.tensor(fixed,device='cuda')
            origins=(pos+normal*cfg.ray_bias_m)[ids]
            radius=torch.deg2rad(5*model.light.radius_logit.sigmoid())
            sun=disk_directions(model.light.direction(),radius,5)
            mask=g['owner']>=0;rows=g['rows'][mask];owner=g['owner'][mask]
            # 不从已舍入的 FP32 covariance 或其 inverse 恢复因子。
            rotation=frame.poses[owner,:3,:3].double()@quaternion_matrix(model.field.quaternion[rows].double())
            scales=model.field.log_scales[rows].double().exp().clamp(cfg.scale_min_m,cfg.scale_max_m)
            means=g['xyz'][mask].double()  # 固定渲染器已经求出的同一世界均值，隔离积分错误。
            alpha=g['opacity'][mask,0].double().clamp(1e-6,1-1e-6)
            result.update(receiver_grid_count=nr,selected_receiver_indices=fixed,
                          valid_actor_gaussians=len(means),split=frame.split,
                          source_checkpoint=str(Path(args.checkpoint)),
                          fp64_reference='raw principal factors; same rendered world means/receiver inputs; no AABB')
            save()
            def factor(dtype,chunk,batch):
                if batch:
                    all_o=origins.repeat(5,1)
                    all_d=sun[:,None,:].expand(5,len(origins),3).reshape(-1,3)
                    return factor_transmittance(all_o,all_d,means,rotation,scales,alpha,
                        chunk,dtype).reshape(5,len(origins))
                return torch.stack([factor_transmittance(origins,d.expand_as(origins),means,
                    rotation,scales,alpha,chunk,dtype) for d in sun])
            ref=measure('FP64_factor_unculled_chunk512_serial5',lambda:factor(torch.float64,512,False))
            for dtype,chunk,batch in [(torch.float64,2048,True),(torch.float32,512,False),
                                      (torch.float32,2048,True)]:
                value=measure(f'{dtype}_factor_unculled_chunk{chunk}_batch{batch}',
                              lambda:factor(dtype,chunk,batch))
                if torch.isfinite(value).all():
                    error=(value-ref).abs()
                    result['cases'][-1].update(max_abs_T_error=float(error.max()),
                        mean_abs_T_error=float(error.mean()),
                        proposed_T_tolerance_pass=bool((error<=1e-5).all()))
                    save()
            for chunk in [512,2048]:
                def old():
                    return torch.stack([tracing.geometry_transmittance(origins,d.expand_as(origins),
                        g,cfg.ray_chunk,chunk) for d in sun])
                value=measure(f'original_FP32_covariance_chunk{chunk}',old)
                if torch.isfinite(value).all():
                    result['cases'][-1]['max_abs_T_error']=float((value-ref).abs().max());save()
            result.update(status='PILOT_COMPLETE_FULL_GATE_PENDING',
                gpu_gradient_gate='NOT_RUN',all_frame_gate='NOT_RUN',
                culling_error_budget_gate='NOT_RUN',performance_gate='NOT_RUN',
                human_verdict=None,problem_solved=None,
                failure_ledger_refs=['V82-F05'],failure_ledger_delta='none')
            save()
    except BaseException as error:
        result.update(status='ERROR_STOPPED',error=repr(error));save();raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
