"""显式Argoverse适配器调用原版Ludus池与CUDA渲染；独立针孔投影检查。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from ludus_renderer import FThetaCamera, TimestampedScene
from ludus_renderer._ops.primitives import (
    CubePool, CUBE_FLAG_WIREFRAME, PRIM_OBSTACLE, PRIM_ROAD_BOUNDARY,
    PRIM_CROSSWALK, PRIM_LANE_LINE_WHITE_SOLID, PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID, PRIM_LANE_LINE_YELLOW_DASHED)
from ludus_renderer.clipgt import OBSTACLE_COLORS_V3, _polylines_to_pool, _polygons_to_pool
from ludus_renderer.torch import LudusCudaTimestampedContext
from prepare_argoverse import OUT, W, H, N, CORNERS

def tensor(value, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device='cuda')

def make_camera(K):
    fx,fy,cx,cy=K
    limit = np.arctan(np.hypot(max(cx,W-cx)/fx,max(cy,H-cy)/fy))*1.05
    samples=np.linspace(0,limit,4097)
    # 约束常数项为0；在整个可见像素域拟合针孔tan(theta)。
    A=np.stack([samples**i for i in range(1,6)],axis=1)
    poly=np.r_[0,np.linalg.lstsq(A,fx*np.tan(samples),rcond=None)[0]]
    check=np.linspace(0,limit,8193)
    errors=np.abs(np.polynomial.polynomial.polyval(check,poly)-fx*np.tan(check))
    assert errors.max()<0.05, errors.max()
    camera=FThetaCamera(tensor([cx,cy]),tensor([W,H]),tensor(poly),float(limit),
                        linear_distortion=tensor([[1,0],[0,fy/fx]]),depth_max=200)
    return camera, {'max_radial_residual_px':float(errors.max()),'max_angle_rad':float(limit),
                    'fifth_order_coefficients':poly.tolist(),'tolerance_px':0.05}

def pool_for_tracks(tracks,times):
    ends=np.cumsum([len(t['frames']) for t in tracks])
    colors=[]
    for t in tracks:
        category=t['category']
        group=('Car' if category in ['REGULAR_VEHICLE','LARGE_VEHICLE','WHEELED_RIDER'] else
               'Truck' if category in ['BUS','BOX_TRUCK','TRUCK','TRUCK_CAB','VEHICULAR_TRAILER','SCHOOL_BUS','ARTICULATED_BUS'] else
               'Cyclist' if category in ['BICYCLIST','MOTORCYCLIST','BICYCLE','MOTORCYCLE'] else
               'Pedestrian' if category in ['PEDESTRIAN','STROLLER','WHEELCHAIR'] else 'Other')
        colors.append(np.asarray(OBSTACLE_COLORS_V3[group]).ravel())
    return CubePool(tensor(np.unique(np.concatenate([times[t['frames']] for t in tracks])),torch.int64),
                    tensor(ends,torch.int32),tensor(np.concatenate([times[t['frames']] for t in tracks]),torch.int64),
                    tensor(np.concatenate([t['centers'] for t in tracks])),
                    tensor(np.concatenate([t['quaternions'] for t in tracks])),
                    tensor([t['dimensions'] for t in tracks]),tensor(np.asarray(colors)),
                    prim_type_id=PRIM_OBSTACLE,render_flags=CUBE_FLAG_WIREFRAME)

def context(camera):
    ctx=LudusCudaTimestampedContext(device=torch.device('cuda'))
    ctx.set_depth_scaling(True)
    ctx.set_msaa_samples(4)
    ctx.set_max_tessellation_levels(cube=0)
    # 官方默认允许500ms外推；本协议缺失轨迹不外推。
    ctx._max_extrapolation_us=0
    ctx.upload_cameras([camera])
    return ctx

def render(ctx, sid, timestamps, camera_world):
    n=len(timestamps)
    # Argoverse相机是RDF；C++入口内部还会执行FLU→RDF，因此输入须先转FLU。
    flu_from_rdf=np.eye(4)
    flu_from_rdf[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
    world_to_flu=flu_from_rdf @ np.linalg.inv(camera_world)
    images=ctx.render(tensor([sid]*n,torch.int32),tensor([0]*n,torch.int32),
                      tensor(timestamps,torch.int64),tensor([0]*n,torch.int32),
                      tensor(world_to_flu),resolution=(H,W))[:,:,:,:3]
    if ctx.needs_vflip:
        images=images.flip(1)
    return images.cpu().numpy()

def independent_check(camera,K):
    ctx=context(camera)
    result=[]
    for xyz in [[0,0,20],[4,2,20],[-4,-2,15]]:
        t={'frames':[0,1],'centers':[xyz,xyz],'quaternions':[[0,0,0,1]]*2,
           'dimensions':[2,2,2],'category':'REGULAR_VEHICLE'}
        scene=TimestampedScene([],[],[pool_for_tracks([t],np.array([1,2]))])
        sid=ctx.upload_scene(scene)
        image=render(ctx,sid,[1],np.eye(4)[None])[0]
        ys,xs=np.where(image.max(-1)>0)
        assert len(xs)>0
        actual=np.array([xs.min(),ys.min(),xs.max(),ys.max()])
        points=CORNERS*2+xyz
        uv=points[:,:2]/points[:,2,None]*K[:2]+K[2:]
        expected=np.r_[uv.min(0),uv.max(0)]
        error=float(np.max(np.abs(actual-expected)))
        assert error<3, (xyz,actual,expected,error)
        result.append({'center':xyz,'actual_bounds':actual.tolist(),'pinhole_bounds':expected.tolist(),
                       'max_bound_residual_px':error})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=OUT)
    a=p.parse_args();out=a.run_dir
    assert not (out/'conditions.npy').exists(), '拒绝覆盖条件'
    result={'status':'started','world_model_generation_calls':0,'human_verdict':None}
    began=time.monotonic()
    try:
        data=json.loads((out/'scene.json').read_text());traj=np.load(out/'trajectory.npz')
        camera,fit=make_camera(traj['K'])
        result['pinhole_approximation']=fit
        result['independent_cube_projection_checks']=independent_check(camera,traj['K'])
        buckets={}
        for line in data['lines']:
            if line['kind']=='road_boundary':
                kind=PRIM_ROAD_BOUNDARY
            else:
                yellow='YELLOW' in line['mark'];dashed='DASH' in line['mark']
                kind=([PRIM_LANE_LINE_YELLOW_SOLID,PRIM_LANE_LINE_YELLOW_DASHED] if yellow else
                      [PRIM_LANE_LINE_WHITE_SOLID,PRIM_LANE_LINE_WHITE_DASHED])[int(dashed)]
            buckets.setdefault(kind,[]).append(tensor(line['xyz']))
        line_pools=[_polylines_to_pool(lines,kind,torch.device('cuda')) for kind,lines in buckets.items()]
        polygon=_polygons_to_pool([tensor(x) for x in data['crossings']],PRIM_CROSSWALK,torch.device('cuda'))
        scene=TimestampedScene(line_pools,[polygon] if polygon else [],
                               [pool_for_tracks(data['tracks'],traj['timestamps_us'])])
        ctx=context(camera);sid=ctx.upload_scene(scene)
        output=np.lib.format.open_memmap(out/'conditions.npy',mode='w+',dtype=np.uint8,shape=(N,H,W,3))
        for start in range(0,N,8):
            output[start:start+8]=render(ctx,sid,traj['timestamps_us'][start:start+8],traj['camera_world'][start:start+8])
        output.flush()
        sheet=Image.new('RGB',(1280,4*378),'#151c29')
        for row,f in enumerate([0,60,150,234]):
            condition=Image.fromarray(output[f]);condition.save(out/f'condition-{f:03d}.png')
            rgb=Image.open(out/f'reference-{f:03d}.png')
            y=row*378
            ImageDraw.Draw(sheet).text((8,y+5),f'Logged RGB | Adapted GT condition | t={f/30:.2f}s',fill='white')
            sheet.paste(rgb.resize((640,352)),(0,y+26));sheet.paste(condition.resize((640,352)),(640,y+26))
        sheet.save(out/'condition-review.jpg',quality=94)
        result.update(status='passed',frames=N,renderer='official Ludus CUDA; custom Argoverse-to-pool adapter',
                      official_source_changed=False,shader_uses_full_box_dimensions=True,
                      interface_pose='world-to-FLU; official C++ converts FLU to RDF',
                      max_extrapolation_us=0,
                      nonzero_pixels_first=int(np.any(output[0]!=0,axis=-1).sum()))
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        result['wall_s']=time.monotonic()-began
        (out/'render_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
