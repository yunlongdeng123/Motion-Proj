"""一次几何机制实验：区分硬range位置梯度与有限射线管轮廓梯度。"""
import argparse,json
from pathlib import Path
import subprocess,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.surface_visibility import BeamTubeFreeSpaceLoss
from motion_proj.worldsim_v73.surface_readout import first_triangle_intersection,direct_free_space_loss


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    loss_fn=BeamTubeFreeSpaceLoss()
    origin=torch.zeros(1,3,device='cuda'); direction=torch.tensor([[0.,0.,1.]],device='cuda')
    base=torch.tensor([[-.06,-.06,5.],[.06,-.06,5.],[.06,.06,5.],[-.06,.06,5.]],device='cuda')
    faces=torch.tensor([[0,1,2],[0,2,3]],device='cuda')
    rows=[]
    for name,offset,observed,repeat in [('front',.025,20.,False),('behind_observed_return',.025,4.,False),
                                        ('outside_tube',.5,20.,False),('duplicated_front',.025,20.,True)]:
        shift=torch.tensor([offset,0.,0.],device='cuda',requires_grad=True)
        vertices=base+shift
        triangles=faces
        if repeat:
            vertices=torch.cat([vertices,vertices]); triangles=torch.cat([faces,faces+4])
        target=torch.tensor([observed],device='cuda')
        soft=loss_fn(vertices,triangles,origin,direction,target)
        grad=torch.autograd.grad(soft,shift,retain_graph=True)[0]
        depth,_=first_triangle_intersection(vertices,triangles,origin,direction)
        hard=direct_free_space_loss(depth,target)
        hard_grad=torch.autograd.grad(hard,shift)[0]
        row={'case':name,'tube_coverage':soft.item(),'tube_translation_gradient':grad.tolist(),
             'hard_intrusion_m':hard.item(),'hard_translation_gradient':hard_grad.tolist()}
        rows.append(row); print(json.dumps(row),flush=True)
    result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'width_m':.03,'resolution':32,'tolerance_m':.2,'rows':rows,'wall_s':time.monotonic()-started,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'boundary':'analytic finite-width geometry proxy only, not real-data improvement or exact visibility derivatives everywhere'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
