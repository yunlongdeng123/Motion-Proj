"""已知距离的解析场景检查原生LiDAR通道；不作为自然badcase。"""
import json
from pathlib import Path
import torch
from gsplat.rendering import lidar_rasterization

device='cuda'
az=torch.linspace(-180,180,33,device=device)[:-1]
el=torch.arange(-4,4,device=device,dtype=torch.float32)
ee,aa=torch.meshgrid(el,az,indexing='ij')
raster=torch.stack([aa,ee,torch.ones_like(aa)*10,torch.zeros_like(aa)],-1)[None]
rows=[]
for opacity in [0.2,0.5,0.9]:
    render,alpha,_,info=lidar_rasterization(
        means=torch.tensor([[10.,0.,0.]],device=device),
        quats=torch.tensor([[1.,0.,0.,0.]],device=device),
        scales=torch.tensor([[.2,.2,.2]],device=device),
        opacities=torch.tensor([opacity],device=device),
        lidar_features=torch.zeros(1,1,3,device=device),velocities=None,
        viewmats=torch.eye(4,device=device)[None],raster_pts=raster,
        tile_elevation_boundaries=torch.tensor([-4.5,3.5],device=device),
        min_elevation=-4.5,max_elevation=3.5,n_elevation_channels=8,azimuth_resolution=11.25,
        near_plane=.2,far_plane=300,rasterize_mode='antialiased',compute_alpha_sum_until_points=False)
    raw=render[0,4,16,-1];a=alpha[0,4,16,0]
    median=info['median_depths'][0,4,16,0]+(a<=.5)*(raw/a.clamp_min(1e-10))
    rows.append({'opacity':opacity,'alpha':float(a),'raw_depth_m':float(raw),
        'normalized_depth_m':float(raw/a.clamp_min(1e-10)),
        'native_pointcloud_median_depth_m':float(median),'known_center_range_m':10.})
    assert a>0 and torch.isfinite(median)
out={'role':'Synthetic measurement contract only; Gaussian center range, not an opaque physical surface','rows':rows,
    'native_readout':'SplatAD filter_lidar_pred_and_gt(output_point_cloud=True) returns point_cloud from raw depth and median_point_cloud from median_depth, both using the learned ray-drop threshold. Keep and evaluate both native readouts; median is not the sole official point-cloud path.'}
p=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1/lidar_measurement_contract.json')
p.write_text(json.dumps(out,indent=2));print(p.read_text(),flush=True)
