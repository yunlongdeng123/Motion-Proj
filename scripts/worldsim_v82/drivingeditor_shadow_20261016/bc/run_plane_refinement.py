"""Targeted geometry probe/refinement after the completed paired B+C pilot."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
import torch
from bc_model import LocalRoad,ActorShadow
from run_experiments import Experiment,BASE,write,sha,now
import renderer

class PlaneRoad(LocalRoad):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        selection=args[0];device=self.receiver_ids.device
        for name in ['receiver_scaling','receiver_rotation']:
            self.register_buffer(name,torch.as_tensor(selection[name],device=device))
    def patch(self,*args,**kwargs):
        result=super().patch(*args,**kwargs)
        if self.enabled:
            if result['scales'] is None:raise ValueError('Plane adapter requires explicit scales/rotation path')
            result['scales']=result['scales'].index_copy(0,self.receiver_ids,self.receiver_scaling)
            result['rotations']=result['rotations'].index_copy(0,self.receiver_ids,self.receiver_rotation)
        return result
    def addon_state(self):
        state=super().addon_state()
        for name in ['receiver_scaling','receiver_rotation']:state[name]=getattr(self,name).detach().cpu().clone()
        return state

class PlaneExperiment(Experiment):
    def __init__(self,args):
        super().__init__(args)
        with np.load(BASE/'selection_plane_v1/selection.npz') as f:self.selection={k:f[k] for k in f.files}
        self.warm=torch.load(BASE/'runs/pilot_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        self.contract.update(geometry_adapter='PlaneRoad',selection_sha256=sha(BASE/'selection_plane_v1/selection.npz'),
            warm_start_addon=str(BASE/'runs/pilot_1000_v1/R4/addon_001000.pt'),
            warm_start_is_not_Adam_continuation=True,
            geometry_scope='Project only selected existing receiver kernels to supported plane; preserve anchor EWA footprint. No new points.')
        write(self.out/'runtime_contract.json',self.contract)
    def activate(self,arm,shadow=None):
        renderer.prepare_static=self.original_prepare
        self.active=PlaneRoad(self.selection,self.static,self.instances,self.data,arm,shadow).cuda()
        self.active.original_prepare=self.original_prepare
        # Geometry is deliberately new. Transfer only the optimized appearance gates.
        with torch.no_grad():
            for name in ['color_raw','opacity_raw','artifact_raw']:
                getattr(self.active,name).copy_(self.warm['road_addon'][name].to(getattr(self.active,name)))
        renderer.prepare_static=self.active.patch
    def shadow(self):
        s=ActorShadow(self.data['plane_model'].cpu(),self.data['world_scale'],'projection').cuda()
        s.load_state_dict(self.warm['shadow_state']);return s
    def run(self):
        self.gradient_smoke()
        if self.args.steps==0:
            (self.out/'R4').mkdir();self.activate('R4',self.shadow());metrics=self.evaluate('R4')
            self.verify_sources();self.update(state='complete',phase='zero_update_plane_projection_probe',training_updates=0,metrics=metrics)
            print(json.dumps(metrics));return
        metrics=self.train_arm('R4',{'state':self.warm['shadow_state']})
        previous=json.loads((BASE/'runs/pilot_1000_v1/R4/evaluation/metrics.json').read_text())
        write(self.out/'comparison.json',{'previous_R4':previous,'plane_R4':metrics,
            'note':'Different geometry, same observations; warm-start addon and fresh Adam. Local reconstruction diagnostic, not an unseen-frame claim.'})
        self.verify_sources();self.update(state='complete',phase='plane_refinement_ready',metrics=metrics,parent_checkpoint_unchanged=True,source_parameters_unchanged=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();a.warmup=1
    assert 0<=a.steps<=1000;e=None
    try:e=PlaneExperiment(a);e.run()
    except BaseException as error:
        if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if e:renderer.prepare_static=e.original_prepare;e.slot.close()
if __name__=='__main__':main()
