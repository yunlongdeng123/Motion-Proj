"""真实场景的加速前后遮挡值与梯度一致性探测；不报告重建质量。"""
import argparse
import json
from pathlib import Path
import torch
from .config import Config
from .data import Dataset
from .model import SceneModel
from .tracing import geometry_transmittance, dense_geometry_transmittance
from .train import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--actor-evidence',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print('GPU_TRACER_PROBE_NOT_STARTED');return
    config=Config.load(args.config)
    torch.set_num_threads(config.threads)
    dataset=Dataset(args.data,'cuda',allow_partial_masks=True)
    model=SceneModel(dataset.seed,config).cuda()
    frame=dataset.frame(12)
    admitted=json.loads(Path(args.actor_evidence).read_text())['eligible_track_ids']
    with torch.no_grad():
        geometry,p=model.buffers(frame,len(dataset.actor_ids),'cuda',isolated=False)
        road=p['road_position'][(p['road_alpha']>.1)&(p['road_normal'].norm(dim=-1)>.5)]
        if not len(road):raise ValueError('真实道路接收面为空')
        direction=model.light.direction()
        picks=[]
        for track in admitted:
            a=dataset.actor_ids.index(track)
            points=geometry['xyz'][geometry['owner']==a]
            if not len(points):continue
            center=points.median(0).values
            height=(center[2]-road[:,2].median()).clamp_min(.1)
            projected=center-direction*height/direction[2].clamp_min(.1)
            picks.append(((road[:,:2]-projected[:2]).square().sum(-1)).topk(min(8,len(road)),largest=False).indices)
        origins=road[torch.cat(picks).unique()]+torch.tensor([0.,0.,.003],device='cuda')
        origins=origins[:64].detach()
    del geometry,p
    results=[];values=[];gradients=[]
    for name,trace in [('dense',dense_geometry_transmittance),('actor_broadphase',geometry_transmittance)]:
        model.zero_grad(set_to_none=True)
        geometry=model.field.evaluate(frame.poses,frame.valid,frame.camera.center)
        direction=model.light.direction().expand_as(origins)
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        transmittance=trace(origins,direction,geometry,config.ray_chunk,config.gaussian_chunk)
        # 避免完全遮挡射线主导；同一组非均匀权重比较几何和光照梯度。
        weight=torch.linspace(.5,1.5,len(origins),device='cuda')
        (transmittance*weight).sum().backward()
        end.record();torch.cuda.synchronize()
        grads={k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
        if not all(torch.isfinite(g).all() for g in grads.values()):raise FloatingPointError(name+' nonfinite gradients')
        values.append(transmittance.detach().clone());gradients.append(grads)
        results.append(dict(method=name,milliseconds=start.elapsed_time(end),rays=len(origins),
                            shadowed_rays=int((transmittance<.99).sum()),unsaturated_shadow_rays=int(((transmittance>.001)&(transmittance<.999)).sum())))
        del geometry,transmittance
    error=float((values[0]-values[1]).abs().max())
    gradient_checks={}
    for key in gradients[0]:
        if key not in gradients[1]:raise ValueError('missing optimized gradient '+key)
        x,y=gradients[0][key],gradients[1][key]
        difference=float((x-y).abs().max());scale=float(x.abs().max())
        gradient_checks[key]=dict(max_absolute_error=difference,reference_max=scale)
        if difference>2e-3+scale*.01:raise ValueError('gradient mismatch '+key)
    if error>2e-3 or not results[0]['shadowed_rays']:raise ValueError('tracer equivalence or nontrivial real-scene coverage failed')
    receipt=dict(status='PASSED',value_max_absolute_error=error,gradient_checks=gradient_checks,methods=results,
                 quality_validated=False,scope='real seed geometry; same analytic half-ray integral, detached six-sigma culling')
    write(args.output,receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
