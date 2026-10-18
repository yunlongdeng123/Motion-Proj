"""Declared implementation contract for the direct matched-five second stage."""
from pathlib import Path
import ast,json
import torch
from rigid_spatial_topology import SpatialPolicy,after_backward
from native_camera_sampler import CameraSampler

class EpochSampler:
    def __init__(self,seed,cameras):
        self.native=CameraSampler(seed,cameras)
    def select(self,cameras):
        # The rigid renderer uses genuine pose/time, not PVG time augmentation.
        return self.native.select(cameras,0.,0.)[0]
    def getstate(self):return self.native.capture()
    def setstate(self,state):self.native.restore(state)

def register_policy(policy):
    policy['adapter_constants'].update(
        hybrid_lambda_dynamic_opacity=100.,dynamic_opacity_domain='isolated_all_dynamic_raw_alpha_full_HxW',
        SH_raise_interval=2000,camera_sampling='native_49_without_replacement',
        topology='native_static_and_rigid_spatial_clone_split_prune',anisotropic_split=True,
        native_screen_radius_history_reset_preserved=True,
        new_SGCM_clones=False,SGCM='stage1_trained_background_retained; no new SGCM in rigid stage',
        rigid_temporal_split=False,rigid_time_shift=False)

def state_flags(static):return {'static_no_time_split':bool(static.no_time_split)}

def restore_flags(static,saved,step,config):
    expected=bool(config.no_time_split) if step<=config.densify_until_iter*config.time_split_frac else False
    assert saved=={'static_no_time_split':expected},(saved,expected)
    static.no_time_split=saved['static_no_time_split']

def native_static_function():
    source=Path(__file__).with_name('gn11_train_reference.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='densification_and_optimization')
    scope={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned native topology>','exec'),scope)
    return scope['densification_and_optimization']

def setup_topology(config,scene):
    return SpatialPolicy(scene_extent=float(scene.cameras_extent),percent_dense=config.percent_dense,
        gradient_threshold=config.densify_grad_threshold,min_opacity=config.thresh_opa_prune,
        big_point_threshold=config.big_point_threshold,contracted=config.contract,
        from_step=config.densify_from_iter,until_step=config.dynamic_densify_until_iter,
        interval=config.densification_interval,opacity_reset_interval=config.opacity_reset_interval,
        dynamic_point_limit=config.dynamic_densify_until_num_points,static_point_limit=config.densify_until_num_points)

@torch.no_grad()
def topology_update(static,instances,optimizer,package,config,scene,policy,step,camera,native_function):
    n=package['static_point_count']
    assert n==len(static.get_xyz),'render/static rows changed before topology'
    if step>config.densify_until_iter*config.time_split_frac:static.no_time_split=False
    screen=package['viewspace_points'][:n]
    screen.grad=package['viewspace_points'].grad[:n]
    native_function(static,config,step,screen,package['visibility_filter'][:n],scene,package['radii'][:n],False)
    return after_backward(instances,optimizer,package,policy,camera.timestamp,step)

def record_topology(output,record):
    if record.get('mutated'):
        with (Path(output)/'topology.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
