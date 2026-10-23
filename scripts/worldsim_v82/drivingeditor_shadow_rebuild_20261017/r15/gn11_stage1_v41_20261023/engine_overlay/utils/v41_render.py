"""额外道路训练前向：冻结原生输入，但保留道路对透射和环境光的梯度。"""

import torch
from utils.v4_road_render import road_kwargs, merge_rows, raster


@torch.no_grad()
def native_snapshot(camera, static, dynamic, pipe, background, environment):
    from gaussian_renderer import prepare_rasterization
    rows = prepare_rasterization(camera, static, pipe)
    if dynamic is not None:
        rows = merge_rows(rows, prepare_rasterization(camera, dynamic, pipe, dynamic=True))
    # prepare_rasterization 的输入可能直接是 Parameter；no_grad 不替代 detach。
    rows = {key: value.detach() if torch.is_tensor(value) else value for key, value in rows.items()}
    rows['means2D'] = torch.zeros_like(rows['means3D'])  # 不共享 V1 屏幕梯度/增密缓冲。
    env_rgb = None
    if environment is not None:
        env_rgb = environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1).detach()
    return dict(rows=rows, degree=static.active_sh_degree,
        background=(background.detach() if environment is not None else torch.zeros_like(background)),
        environment=env_rgb)


def render_snapshot(camera, pipe, snapshot, road=None, for_training=False):
    rows = snapshot['rows']
    if any(value.requires_grad for value in rows.values() if torch.is_tensor(value)):
        raise ValueError('原 V1 光栅输入未完全 detach')
    if road is not None:
        channels = rows['shs'].shape[1] if rows['shs'] is not None else 1
        added = road_kwargs(camera, road, pipe, channels,
            precomputed_color=rows['colors_precomp'] is not None, for_training=for_training)
        rows = dict(rows)
        rows['features'] = torch.cat((rows['features'], torch.zeros_like(rows['features'][:, :1])), 1)
        added['features'] = torch.cat((added['features'], torch.ones_like(added['features'][:, :1])), 1)
        rows = merge_rows(rows, added)
    package = raster(camera, pipe, snapshot['degree'], rows, snapshot['background'])
    package['road_visible_alpha'] = package['extra'] if road is not None else torch.zeros_like(package['alpha'])
    if snapshot['environment'] is not None:
        # 环境颜色冻结，(1-alpha) 仍必须对道路 opacity 可导。
        package['rgb'] = package['rgb']+(1-package['alpha'])*snapshot['environment']
    return package


def render_training_pair(camera, static, dynamic, road, pipe, background, environment):
    snapshot = native_snapshot(camera, static, dynamic, pipe, background, environment)
    with torch.no_grad():
        native = render_snapshot(camera, pipe, snapshot)['rgb']
    joint = render_snapshot(camera, pipe, snapshot, road, for_training=True)
    return native, joint


def render_training_deleted(camera, static, road, pipe, background, environment):
    # 与既有验收默认的 all_dynamic 删除一致；不改写静态分层或任何 Gaussian 归属。
    snapshot = native_snapshot(camera, static, None, pipe, background, environment)
    return render_snapshot(camera, pipe, snapshot, road, for_training=True)
