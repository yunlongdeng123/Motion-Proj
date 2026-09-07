"""一次解析几何实验：正确命中、前后遮挡、重复表面、无支持及轮廓梯度。"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.first_event import FirstSurfaceEventLoss


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); torch.set_num_threads(2)
    objective=FirstSurfaceEventLoss()
    origin=torch.zeros(1,3,device='cuda'); direction=torch.tensor([[0.,0.,1.]],device='cuda')
    target=torch.tensor([5.],device='cuda')
    faces=torch.tensor([[0,1,2],[0,2,3]],device='cuda')
    results={}
    for name,depths,shift_x,half in [('correct',[5.],0.,.15),('early_occludes_correct',[4.6,5.],0.,.15),
            ('duplicate_early',[4.6,4.6,5.],0.,.15),('no_support',[5.],.4,.06),('edge',[5.],.025,.06)]:
        offset=torch.tensor([[shift_x,0.,d] for d in depths],device='cuda',requires_grad=True)
        quad=torch.tensor([[-half,-half,0.],[half,-half,0.],[half,half,0.],[-half,half,0.]],device='cuda')
        vertices=(offset[:,None]+quad[None]).reshape(-1,3)
        mesh_faces=torch.cat([faces+4*i for i in range(len(depths))])
        loss,statistics=objective(vertices,mesh_faces,origin,direction,target)
        loss.backward()
        results[name]={'capped_nll':loss.item(),'offset_gradient':offset.grad.tolist(),**statistics}
    # 本次新物理语义的关键解析数值；不扩展成重复回归或验收流程。
    assert abs(results['correct']['capped_nll'])<1e-4
    assert abs(results['early_occludes_correct']['capped_nll']-2.)<1e-3
    assert max(abs(x) for x in results['early_occludes_correct']['offset_gradient'][1])<1e-5
    assert abs(results['duplicate_early']['capped_nll']-2.)<1e-3
    assert results['no_support']['capped_nll']==28. and results['no_support']['no_support_rays']==1
    assert results['edge']['offset_gradient'][0][0]>0
    output={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'wall_s':time.monotonic()-start,'sigma_m':.2,'cap':28.,'width_m':.03,'resolution':32,
        'cases':results,'boundary':'analytic finite-footprint capped NLL proxy only; no-support gradient remains zero and requires direct geometry coverage; not real-data effectiveness'}
    (args.output/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output))


if __name__=='__main__': main()
