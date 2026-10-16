"""Final bounded test: close local time freedoms across all49 source states."""
import argparse,json,traceback
import numpy as np
import torch
import torch.nn.functional as F
from run_deferred_refinement import DeferredExperiment
from run_plane_refinement import PlaneRoad
from run_experiments import BASE,write,sha
import renderer

class UnionRoad(PlaneRoad):
    def patch(self,viewpoint_camera,pc,*args,**kwargs):
        result=super().patch(viewpoint_camera,pc,*args,**kwargs)
        if self.enabled:
            # Closing position/opacity is insufficient if SH directions still use
            # the old moving centers. Evaluate remaining SH at fixed centers too.
            ids=self.artifact_ids
            directions=F.normalize(self.artifact_xyz_anchor-viewpoint_camera.camera_center,dim=-1,eps=1e-8)
            colors=renderer.evaluate_sh_rgb(pc.get_features[ids],directions,int(pc.active_sh_degree))
            result['colors_precomp']=result['colors_precomp'].index_copy(0,ids,colors)
        return result

class UnionExperiment(DeferredExperiment):
    def __init__(self,args):
        super().__init__(args)
        with np.load(BASE/'selection_union_plane_v1/selection.npz') as f:self.selection={k:f[k] for k in f.files}
        self.warm=torch.load(BASE/'runs/deferred_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        self.contract.update(geometry_adapter='UnionRoad',selection_sha256=sha(BASE/'selection_union_plane_v1/selection.npz'),
            warm_start_addon=str(BASE/'runs/deferred_1000_v1/R4/addon_001000.pt'),
            source_coverage='all49 potentially foreground kernels in evidenced road window; other points opacity-only, not semantic bad labels',
            all_covered_SH_directions_use_fixed_centers=True,
            remaining_additional_budget_after_this_run=0)
        write(self.out/'runtime_contract.json',self.contract)
    def activate(self,arm,shadow=None):
        renderer.prepare_static=self.original_prepare
        self.active=UnionRoad(self.selection,self.static,self.instances,self.data,arm,shadow).cuda();self.active.original_prepare=self.original_prepare
        old=self.warm['road_addon']
        with torch.no_grad():
            for id_name,names in [('receiver_ids',['color_raw','opacity_raw']),('artifact_ids',['artifact_raw'])]:
                old_ids=old[id_name].numpy();new_ids=getattr(self.active,id_name).cpu().numpy()
                lookup={int(k):j for j,k in enumerate(old_ids)}
                new_rows=np.array([j for j,k in enumerate(new_ids) if int(k) in lookup],dtype=np.int64)
                old_rows=np.array([lookup[int(new_ids[j])] for j in new_rows],dtype=np.int64)
                new_rows=torch.tensor(new_rows,device='cuda');old_rows=torch.tensor(old_rows)
                for name in names:getattr(self.active,name)[new_rows]=old[name][old_rows].to('cuda')
        renderer.prepare_static=self.active.patch
    def run(self):
        self.gradient_smoke()
        metrics=self.train_arm('R4',{'state':self.warm['shadow_state']})
        previous=json.loads((BASE/'runs/deferred_1000_v1/R4/evaluation/metrics.json').read_text())
        write(self.out/'comparison.json',{'single_time_receiver_set':previous,'all49_union':metrics,
          'note':'Expanded candidate selection and fixed SH centers; same data, fresh Adam with ID-matched warm start. Other candidates are not semantic artifact GT.'})
        self.verify_sources();self.update(state='complete',phase='all49_union_comparison_ready',metrics=metrics,
          parent_checkpoint_unchanged=True,source_parameters_unchanged=True,total_additional_updates_after_pilot=3000,
          further_training_not_scheduled=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();a.warmup=1
    assert a.steps==1000;e=None
    try:e=UnionExperiment(a);e.run()
    except BaseException as error:
        if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if e:renderer.prepare_static=e.original_prepare;e.slot.close()
if __name__=='__main__':main()
