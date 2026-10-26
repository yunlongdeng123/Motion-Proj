"""在同一次深度排序中增加深度二阶矩，约束表面一致性而不指定分支责任。"""
import types
from editor_common import *


def install(native):
    original=native.render
    def render(self,camera,road=None,suppress=None,dynamic=False,static_multiplier=None):
        if road is None:return original(camera,road,suppress,dynamic,static_multiplier)
        kwargs=prepare_rasterization(camera,self.static,self.args);kwargs['means2D']=torch.zeros_like(kwargs['means3D'])
        if suppress is not None:kwargs['mask']=kwargs['mask']&~suppress
        if static_multiplier is not None:kwargs['opacities']=kwargs['opacities']*static_multiplier[:,None]
        if dynamic:
            dy=prepare_rasterization(camera,self.dynamic,self.args,dynamic=True);dy['means2D']=torch.zeros_like(dy['means3D']);kwargs=merge_rows(kwargs,dy)
        depth=(torch.cat((kwargs['means3D'],torch.ones_like(kwargs['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3])
        kwargs['features']=torch.cat((kwargs['features'],torch.zeros_like(depth),depth.square()),1)
        channels=kwargs['shs'].shape[1] if kwargs['shs'] is not None else 1
        rr=road_kwargs(camera,road,self.args,channels,precomputed_color=kwargs['colors_precomp'] is not None)
        rr['features']=torch.cat((rr['features'],torch.ones_like(rr['features'][:,:1]),rr['features'][:,:1].square()),1)
        kwargs=merge_rows(kwargs,rr);out=raster(camera,self.args,self.static.active_sh_degree,kwargs,self.bg)
        env=self.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        out['rgb']=out['rgb']+(1-out['alpha'])*env
        out['contribution']=out['extra'][:1];out['depth2_sum']=out['extra'][1:2]
        return out
    # 实际CUDA检查额外统计没有改变RGB；在新实验中执行，不干预健康基线。
    with torch.no_grad():
        probe=Road(FULL/'full199_initialization.npz');camera=native.camera(102)
        expected=original(camera,probe)['rgb'];actual=render(native,camera,probe)['rgb']
        error=float((expected-actual).abs().max());assert error<=1e-6,error
    native.render=types.MethodType(render,native)
    return dict(rgb_max_difference=error,statistic='贡献加权深度一/二阶矩，非首交面',branch_agnostic=True)


def loss(package,reference_depth_m,mask,scale,tolerance_m):
    alpha=package['alpha'].clamp_min(1e-6)
    mean=(package['depth_sum']/alpha)[0,mask];second=(package['depth2_sum']/alpha)[0,mask]
    target=reference_depth_m[mask]*scale
    assert torch.isfinite(target).all()
    squared_distance=(second-2*target*mean+target.square()).clamp_min(0)
    # E[(z-z_ref)^2]同时约束期望位置和分布宽度；不通过逼道路占比取代几何。
    return torch.log1p(squared_distance/(scale*tolerance_m)**2).mean()
