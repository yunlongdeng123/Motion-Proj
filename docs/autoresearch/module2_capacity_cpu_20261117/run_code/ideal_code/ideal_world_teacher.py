"""CPU解析世界教师；不导入Torch、GS或CUDA，不读取已渲染RGB构造目标。"""
from __future__ import annotations
import numpy as np


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def world_coordinates(world_xyz, config):
    frame = config['world_frame']
    xy = np.asarray(world_xyz, np.float64)[..., :2] - np.asarray(frame['origin_xy'])
    return xy @ np.asarray(frame['longitudinal_xy']), xy @ np.asarray(frame['lateral_xy'])


def curve_coordinates(world_xyz, config):
    s, lateral = world_coordinates(world_xyz, config)
    radius = float(config['curve']['radius_m'])
    rho = np.hypot(s, radius - lateral)
    u = radius * np.arctan2(s, radius - lateral)
    v = radius - rho
    es = np.asarray(config['world_frame']['longitudinal_xy'])
    el = np.asarray(config['world_frame']['lateral_xy'])
    gu = (radius * (radius-lateral) / rho**2)[..., None] * es + (radius*s/rho**2)[..., None] * el
    gv = (-s/rho)[..., None] * es + ((radius-lateral)/rho)[..., None] * el
    return u, v, gu, gv


def evaluate_world(world_xyz, case, config):
    """返回原生sRGB外观和白漆比例；所有相机调用同一世界函数。"""
    s, lateral = world_coordinates(world_xyz, config)
    asphalt = np.broadcast_to(np.asarray(config['asphalt']['constant_srgb']), (*s.shape, 3)).copy()
    if case['asphalt'] == 'lowfreq':
        values = config['asphalt']['lowfreq']
        modulation = np.sin(2*np.pi*s/values['longitudinal_period_m']) * np.cos(2*np.pi*lateral/values['lateral_period_m'])
        asphalt += modulation[..., None] * np.asarray(values['amplitude_srgb'])
    if case['asphalt'] == 'stress':
        modulation = np.sin(2*np.pi*s/.04) * np.cos(2*np.pi*lateral/.08)
        asphalt += modulation[..., None] * np.asarray([.07,.05,.03])
    paint = np.zeros(s.shape, np.float64)
    if case.get('marking'):
        u, v, _, _ = curve_coordinates(world_xyz, config)
        marking = case['marking']
        across = smoothstep((marking['width_m']/2 - np.abs(v))/marking['cross_edge_m'])
        phase = (u-marking['phase_m']+marking['period_m']/2) % marking['period_m'] - marking['period_m']/2
        along = smoothstep((marking['on_m']/2-np.abs(phase))/marking['longitudinal_edge_m'])
        paint = across * along
    rgb = asphalt * (1-paint[..., None]) + np.asarray(config['paint_srgb']) * paint[..., None]
    return np.clip(rgb,0,1), paint


def srgb_to_linear(value):
    value = np.asarray(value)
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def sampling_qualification(world_xyz, pixel_world_xy_jacobian, case, config):
    """Jacobian轴为(..., pixel[x,y], world[x,y])，资格仅分层，不改主域。"""
    u, v, gu, gv = curve_coordinates(world_xyz, config)
    ju = np.einsum('...w,...pw->...p', gu, pixel_world_xy_jacobian)
    jv = np.einsum('...w,...pw->...p', gv, pixel_world_xy_jacobian)
    shape = u.shape
    result = dict(width_px=np.full(shape,np.nan),on_px=np.full(shape,np.nan),
                  off_px=np.full(shape,np.nan),period_px=np.full(shape,np.nan),
                  resolvable=np.zeros(shape,bool),nyquist_ok=np.ones(shape,bool),
                  line_corridor=np.zeros(shape,bool))
    es=np.asarray(config['world_frame']['longitudinal_xy'])
    el=np.asarray(config['world_frame']['lateral_xy'])
    texture_frequency=np.zeros(shape,np.float64)
    periods=None
    if case['asphalt']=='lowfreq':
        low=config['asphalt']['lowfreq'];periods=(low['longitudinal_period_m'],low['lateral_period_m'])
    elif case['asphalt']=='stress':
        periods=(.04,.08)
    if periods:
        for sign in (-1,1):
            wave=es/periods[0]+sign*el/periods[1]
            projected=np.einsum('w,...pw->...p',wave,pixel_world_xy_jacobian)
            texture_frequency=np.maximum(texture_frequency,np.max(np.abs(projected),axis=-1))
    result['asphalt_max_cycles_per_pixel_axis']=texture_frequency
    result['nyquist_ok']=texture_frequency<=.5
    if case.get('marking'):
        mark = case['marking']
        du = np.maximum(np.linalg.norm(ju,axis=-1),1e-12)
        dv = np.maximum(np.linalg.norm(jv,axis=-1),1e-12)
        result.update(width_px=mark['width_m']/dv,on_px=mark['on_m']/du,
                      off_px=(mark['period_m']-mark['on_m'])/du,period_px=mark['period_m']/du,
                      line_corridor=np.abs(v)<=mark['width_m'],
                      nyquist_ok=result['nyquist_ok']&(np.max(np.abs(ju)/mark['period_m'],axis=-1)<=.5))
        threshold = config['sampling']['detail_minimum_pixels']
        result['resolvable']=(result['width_px']>=threshold)&(result['on_px']>=threshold)&(result['off_px']>=threshold)&result['nyquist_ok']
    else:
        result['resolvable'] = np.ones(shape,bool)
    return result


def plane_ray_jacobian(origin, rays, du, dv, normals, point):
    """固定局部表面面片与针孔光线的解析像素→world XY导数。"""
    denominator = np.einsum('...i,...i->...',normals,rays)
    distance = np.einsum('...i,...i->...',point-origin,normals)
    t = distance / denominator
    derivatives=[]
    for direction in (du,dv):
        change = t[...,None]*direction - (t*np.einsum('...i,i->...',normals,direction)/denominator)[...,None]*rays
        derivatives.append(change[...,:2])
    return np.stack(derivatives,axis=-2)
