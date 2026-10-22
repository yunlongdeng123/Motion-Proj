"""V4初始化只读验收；不执行训练，不修改冻结模型或151个原V1文件。"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,ast,base64,importlib.util,json,shutil,sys,time,zlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from audit_math import footprint_regions,in_region,grid_membership,region_grid_keys,projected_mask,pixel_metrics,horizontal_footprints

HERE=Path(__file__).resolve().parent
V4=HERE.parent
R15=V4.parent
DIAG=R15/'gn11_stage1_v3_20261020/diagnostics_20261021/src'
CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v4_20261022')
sys.path.insert(0,str(DIAG))
from backend import Adapter,clean
from cloud_sampling import pack
from export_inspection import prepare_depth,native_raster,measured_frame
from ray_reference import covariance


def native_module():
    path=V4/'engine_overlay/utils/v4_road.py'
    spec=importlib.util.spec_from_file_location('v4_audit_native_road',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2),encoding='utf-8')


def write_js(path,lhs,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(lhs+'='+json.dumps(clean(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+';',encoding='utf-8')


def ply(path):
    v=PlyData.read(str(path))['vertex'];return np.c_[v['x'],v['y'],v['z']].astype(float),np.asarray(v['nx'],np.float32)


def native_initial(config,meta,plane):
    """按当前V4原生loader及3NN公式复原；不是历史V1训练后候选。"""
    source=Path(config['source_path']);world,t=ply(source/'points3d.ply')
    if not config['including_dynamic'] or config['random_init_point']!=0:raise ValueError('unsupported native initializer')
    dynamic=np.concatenate([ply(p)[0] for p in sorted(source.glob('points3d_dynamic_*.ply'))])
    keep=cKDTree(dynamic).query(world,k=1,distance_upper_bound=.1,workers=4)[0]>=.1
    ids=np.flatnonzero(keep).astype(np.int32);world=world[keep];t=t[keep]
    A=np.asarray(meta['world_to_model']);anchor=np.asarray(meta['coordinate_anchor']);inv=np.linalg.inv(anchor)
    model=(world@A[:3,:3].T+A[:3,3]).astype(np.float32)
    # 原生create_from_pcd对float32融合坐标计算3NN均方间距，CPU精确近邻公式。
    distances=cKDTree(model).query(model,k=4,workers=4)[0][:,1:]
    sigma=np.sqrt(np.maximum((distances**2).mean(1),1e-7)).astype(np.float32)
    local=world@inv[:3,:3].T+inv[:3,3]
    ground=(abs(local@plane[:3]+plane[3])<=.15)&(local[:,0]>=3)&(local[:,0]<=90)&(abs(local[:,1])<=8)
    # Scene会用loader的49帧实际时长覆盖GaussianModel的配置时长，再创建高斯。
    # 本数据frame_interval=.02，实际[-.48,.48]，不是配置文件的[-.5,.5]。
    dt=float(config['frame_interval']*48 if config['frame_interval']>0 else config['time_duration'][1]-config['time_duration'][0]);sigma_t=np.sqrt(dt*config['t_init'])
    result=dict(xyz=model[ground],scale=np.repeat(sigma[ground,None],3,axis=1),q=np.tile([1,0,0,0],(ground.sum(),1)).astype(np.float32),
        opacity=np.full(ground.sum(),.01,np.float32),t=t[ground].astype(np.float32),sigma_t=np.full(ground.sum(),sigma_t,np.float32),vel=np.zeros((ground.sum(),3),np.float32),
        road=np.full(ground.sum(),-1,np.int64),cycle=float(config['cycle']))
    return result,local[ground],ids[ground],dict(total_static=len(world),ground_count=int(ground.sum()),initial_opacity=.01,initial_sigma_t=sigma_t,actual_loader_time_duration=[-dt/2,dt/2],native_cycle=float(config['cycle']),
        definition='current V4 source points3d.ply, exact native dynamic-neighbor<0.1m exclusion; ground diagnostic subset plane±0.15m and x3..90m/y±8m; no trained historical V1 points',
        sigma_source='CPU reconstruction of native 3NN mean-square formula, not saved CUDA step0',point_source=str(source/'points3d.ply'))


def ground_pool(meta,plane):
    source=Path(meta['original_engineering']).parent/'inputs/scene0008_089_137';anchor=np.asarray(meta['coordinate_anchor']);inv=np.linalg.inv(anchor);parts=[];times=[]
    for raw in range(89,138):
        fid=f'0008{raw:03d}';pose=np.loadtxt(source/'pose'/f'{fid}.txt');lidar=np.fromfile(source/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(float)
        world=lidar@pose[:3,:3].T+pose[:3,3];local=world@inv[:3,:3].T+inv[:3,3];occupied=np.zeros(len(world),bool)
        for row in json.loads((source/'annotations'/f'{fid}.json').read_text())['laser_labels']:
            box=row['box'];half=np.array([box['length'],box['width'],box['height']])/2
            if min(half)<=0:continue
            c,s=np.cos(box['heading']),np.sin(box['heading']);axes=pose[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
            center=pose[:3,:3]@np.array([box['center_x'],box['center_y'],box['center_z']])+pose[:3,3]
            occupied|=(abs((world-center)@axes)<=half).all(1)
        ground=(abs(local@plane[:3]+plane[3])<=.15)&~occupied&(local[:,0]>=3)&(local[:,0]<=90)&(abs(local[:,1])<=8)
        parts.append(local[ground]);times.append(np.full(ground.sum(),raw,np.int16))
    allpoints=np.concatenate(parts);allraw=np.concatenate(times);_,ids=np.unique(np.floor(allpoints[:,:2]/.1).astype(int),axis=0,return_index=True)
    return allpoints[ids],allraw[ids],len(allpoints)


def road_model(layer):
    n=len(layer.xyz)
    return dict(xyz=layer.xyz.detach().numpy().copy(),scale=layer.scaling.numpy().copy(),q=layer.rotation.numpy().copy(),opacity=layer.opacity.sigmoid().detach().numpy().reshape(-1).copy(),
        t=np.zeros(n,np.float32),sigma_t=np.ones(n,np.float32),vel=np.zeros((n,3),np.float32),road=layer.unit_ids.numpy().copy(),cycle=1.)


def render(a,model,raw,root,mask=None,capacity=False):
    b=model
    if mask is not None:b={k:(v[mask] if isinstance(v,np.ndarray) and v.shape[0]==len(model['xyz']) else v) for k,v in model.items()}
    if capacity:
        b={**b,'opacity':np.ones(len(b['xyz']),np.float32),'t':np.zeros(len(b['xyz']),np.float32),'sigma_t':np.ones(len(b['xyz']),np.float32),'road':np.arange(len(b['xyz']),dtype=np.int64)}
    p=prepare_depth(a,b,raw,root);p['road']=np.zeros(len(p['depth']),bool)
    alpha,ds=native_raster(root,p);depth=np.divide(ds,alpha,out=np.zeros_like(ds),where=alpha>1e-8)/a.scale
    return alpha,depth


def arrays_to_js_maps(maps):
    # 展示包用已声明的16bitalpha/厘米z；所有统计在量化前计算。
    return {key:(np.rint(np.clip(value,0,1)*65535).astype(np.uint16) if key.startswith(('alpha','cap')) else np.rint(np.clip(value,0,655.34)*100).astype(np.uint16)) for key,value in maps.items()}


def export(output,campaign=CAMPAIGN,raws=range(89,138),resume=False):
    started=time.time();output=Path(output);root=output/'v4_init';root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((campaign/'plan.json').read_text());config=json.loads((campaign/'configs/R.json').read_text());native=native_module();meta,z=native.load_bundle(plan['road_bundle']);layer=native.RoadLayer(z,meta['meter_to_model'],'RP','cpu')
    import torch
    if abs(meta['policy']['cell_m']-.2)>1e-10:raise ValueError('audit requires the requested0.2m common world grid')
    params=[k for k,_ in layer.named_parameters()];assert params==['dc','opacity'];assert int(layer.steps)==0 and not layer.output_mask().any()
    a=Adapter();assert Path(plan['road_bundle']).resolve()==a.bundle.resolve();np.testing.assert_allclose(a.A,meta['world_to_model'],atol=1e-10)
    ready=json.loads(Path(meta['original_readiness']).read_text());plane=np.array(ready['SGCM_geometry']['ground_fit']['plane_anchor']);anchor=np.array(meta['coordinate_anchor']);M=np.linalg.inv(a.A);L=np.linalg.inv(anchor)@M
    unit_ids=layer.unit_ids.numpy();N=len(z['unit_ids']);supported=z['surface_supported'];observed,prior=native.classify_units(z)
    model=road_model(layer)
    # V3原安装公式独立复原，逐项对照实际V4 RoadLayer初始buffer。
    v3_path=R15/'gn11_stage1_v3_20261020/engine_overlay/utils/v3_multiview_road.py'
    node=next(n for n in ast.parse(v3_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quaternion_from_normal')
    namespace={'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=[node],type_ignores=[]),str(v3_path),'exec'),namespace)
    q=namespace['quaternion_from_normal'](torch.as_tensor(z['normal_model'][supported],dtype=torch.float32)).numpy()
    v3_sigma=(torch.tensor([.13,.13,.015])*a.scale).log().exp().numpy()
    v3={**model,'xyz':z['model_xyz'][supported].astype(np.float32),'q':q,'scale':np.tile(v3_sigma,(supported.sum(),1)),'opacity':np.full(supported.sum(),.5,np.float32)}
    geometry={k:float(np.abs(model[k]-v3[k]).max()) for k in ['xyz','q','scale','opacity']}
    for key in ['xyz','scale','opacity']:assert geometry[key]<2e-7,(key,geometry[key])
    qa=model['q'].astype(float);qb=q.astype(float);qa/=np.linalg.norm(qa,axis=1)[:,None];qb/=np.linalg.norm(qb,axis=1)[:,None]
    rotation_error=float(np.degrees(2*np.arccos(np.clip(abs((qa*qb).sum(1)),0,1))).max())
    # 法线预归一化顺序可产生微小FP32方向差；记录真实差异，不把V3图直接拷作V4图。
    assert rotation_error<.1,rotation_error
    native_init,native_local,native_rows,native_meta=native_initial(config,meta,plane)
    points,point_raw,all_ground=ground_pool(meta,plane);tree=cKDTree(points[:,:2]);dist,near=tree.query(z['xy_local'],k=8,distance_upper_bound=.8,workers=4);used=np.isfinite(dist).sum(1)
    np.testing.assert_array_equal(used>=3,supported);np.testing.assert_allclose(tree.query(z['xy_local'],workers=4)[0],z['surface_distance_m'],atol=1e-5)
    static_binding=json.loads((Path(plan['V1_source']).parent.parent/'runs/scene0008_089_137_hybrid_30000_seed0_GN11/actual_sgcm_binding.json').read_text())
    assert native_meta['total_static']==static_binding['static_points']
    # 元数据只读来自同一组49相机；每帧真实OBB用于固定世界选区。
    from cloud_sampling import Sampler
    sampler=Sampler(a);vehicles=sampler.metadata['vehicles'];regions=footprint_regions(vehicles['102'],M,anchor,102)
    # 将选区里原候选网格没有包含的0.2m单元也纳入共同分母，单独标记原因。
    original_count=N;lookup=set(map(tuple,z['keys'].tolist()));extra=sorted({tuple(key) for r in regions.values() for key in region_grid_keys(r,1.2)}-lookup)
    outside=np.r_[np.zeros(N,bool),np.ones(len(extra),bool)]
    if extra:
        k=np.asarray(extra,np.int64);xynew=(k+.5)*.2;height=-(xynew@plane[:2]+plane[3])/plane[2];localnew=np.c_[xynew,height];worldnew=localnew@anchor[:3,:3].T+anchor[:3,3]
        modelnew=worldnew@a.A[:3,:3].T+a.A[:3,3];nworld=(anchor[:3,:3]@plane[:3]);nworld/=np.linalg.norm(nworld);nmodel=a.A[:3,:3]@nworld/a.scale
        for name in ['keys','xy_local','world','model_xyz','normal_world','normal_model']:
            new={'keys':k,'xy_local':xynew,'world':worldnew,'model_xyz':modelnew,'normal_world':np.tile(nworld,(len(extra),1)),'normal_model':np.tile(nmodel,(len(extra),1))}[name]
            z[name]=np.concatenate([z[name],new])
        for name in ['base_weight','source_uv','source_rgb','source_depth']+[k for k in z if k.startswith('gate_')]:
            shape=list(z[name].shape);shape[1]=len(extra);z[name]=np.concatenate([z[name],np.zeros(shape,z[name].dtype)],axis=1)
        supported=np.r_[supported,np.zeros(len(extra),bool)];observed=np.r_[observed,np.zeros(len(extra),bool)];prior=np.r_[prior,np.zeros(len(extra),bool)]
        z['surface_supported']=supported
        z['surface_distance_m']=np.r_[z['surface_distance_m'],tree.query(xynew,workers=4)[0]]
        dist,near=tree.query(z['xy_local'],k=8,distance_upper_bound=.8,workers=4);used=np.isfinite(dist).sum(1)
        N=len(supported)
    xy=z['xy_local'];keys=z['keys'];membership=grid_membership(native_local[:,:2],keys);counts=np.bincount(membership[membership>=0],minlength=N).astype(np.int32)
    support=tree.query_ball_point(xy,.8,return_length=True,workers=4).astype(np.int32)
    # 0.2米格子内4x4固定采样检查3sigma足迹，不把屏幕方点当真实尺度。
    offsets=np.array([[x,y] for x in [-.075,-.025,.025,.075] for y in [-.075,-.025,.025,.075]])
    probes=(xy[:,None,:]+offsets).reshape(-1,2)
    B=L[:2,:3];native_cov=np.einsum('ij,njk,lk->nil',B,covariance(native_init['scale'],native_init['q']),B)
    road_cov=np.einsum('ij,njk,lk->nil',B,covariance(model['scale'],model['q']),B)
    footprint_native=horizontal_footprints(probes,native_local[:,:2],native_cov).reshape(N,16).mean(1)
    footprint_road=horizontal_footprints(probes,xy[supported],road_cov).reshape(N,16).mean(1)
    footprint_R=horizontal_footprints(probes,xy[observed],road_cov[observed[unit_ids]]).reshape(N,16).mean(1)
    footprint_RP=horizontal_footprints(probes,xy[observed|prior],road_cov[(observed|prior)[unit_ids]]).reshape(N,16).mean(1)
    v3cov=np.einsum('ij,njk,lk->nil',B,covariance(v3['scale'],v3['q']),B)
    footprint_v3=horizontal_footprints(probes,xy[supported],v3cov).reshape(N,16).mean(1)
    roadcovfull=np.zeros((N,2,2),np.float32);roadcovfull[supported]=road_cov
    v3covfull=np.zeros((N,2,2),np.float32);v3covfull[supported]=v3cov
    native_height=np.divide(np.bincount(membership[membership>=0],weights=native_local[membership>=0,2],minlength=N),counts,out=np.zeros(N),where=counts>0)
    plane_height=-(xy@plane[:2]+plane[3])/plane[2];local_z=(z['world']-anchor[:3,3])@anchor[:3,2]
    group=np.zeros(N,np.uint8);group[supported]=3;group[observed]=1;group[prior]=2;group[outside]=4
    gates=sorted(k for k in z if k.startswith('gate_'));bits=np.zeros((49,N),np.uint16)
    for i,key in enumerate(gates):bits|=z[key].astype(np.uint16)<<i
    arrays=dict(keys=keys.astype(np.int32),xy=xy.astype(np.float32),world=z['world'],xyz=z['model_xyz'].astype(np.float32),normal=z['normal_model'].astype(np.float32),supported=supported.astype(np.uint8),group=group,outside_original_grid=outside.astype(np.uint8),original_unit_id=np.r_[np.arange(original_count,dtype=np.int32),np.full(N-original_count,-1,np.int32)],
        native_count=counts,support_count=support,used_k8_count=used.astype(np.uint8),nearest_m=z['surface_distance_m'].astype(np.float32),source_count=(z['base_weight']>0).sum(0).astype(np.uint8),
        source_weight=z['base_weight'],source_uv=np.where((z['base_weight']>0)[:,:,None],z['source_uv'],0),source_rgb=np.where((z['base_weight']>0)[:,:,None],z['source_rgb'],0),source_depth=np.where(z['base_weight']>0,z['source_depth'],0),gate_bits=bits,footprint_native=footprint_native.astype(np.float32),footprint_road=footprint_road.astype(np.float32),footprint_R=footprint_R.astype(np.float32),footprint_RP=footprint_RP.astype(np.float32),footprint_v3=footprint_v3.astype(np.float32),
        native_xy=native_local[:,:2].astype(np.float32),native_sigma_m=(native_init['scale'][:,0]/a.scale).astype(np.float32),native_ply_row=native_rows,road_cov_xy=roadcovfull,v3_cov_xy=v3covfull,
        plane_delta_m=(local_z-plane_height).astype(np.float32),native_plane_delta_m=(native_height-plane_height).astype(np.float32))
    initial_sigma=native_init['scale'][:,0]/a.scale;summary_regions={}
    for actor,region in regions.items():
        row={}
        for scope,halo in [('core',0),('context',1.2)]:
            mask=in_region(xy,region,halo);nm=in_region(native_local[:,:2],region,halo);roadmask=mask&supported
            row[scope]=dict(grid_cells=int(mask.sum()),outside_original_candidate_grid=int((mask&outside).sum()),unsupported_inside_original_grid=int((mask&~supported&~outside).sum()),native_centers=int(nm.sum()),native_occupied_cells=int(np.count_nonzero(counts[mask])),road_created=int(roadmask.sum()),road_missing=int((mask&~supported).sum()),
                R_train=int((mask&observed).sum()),RP_train=int((mask&(observed|prior)).sum()),unknown_created=int((mask&supported&~observed&~prior).sum()),output_step0=0,
                native_sigma_m=np.quantile(initial_sigma[nm],[.1,.5,.9]).tolist() if nm.any() else None,road_sigma_m=[.13,.13,.015],
                native_footprint_3sigma=float(footprint_native[mask].mean()) if mask.any() else None,road_footprint_3sigma=float(footprint_road[mask].mean()) if mask.any() else None,R_footprint_3sigma=float(footprint_R[mask].mean()) if mask.any() else None,RP_footprint_3sigma=float(footprint_RP[mask].mean()) if mask.any() else None,v3_footprint_3sigma=float(footprint_v3[mask].mean()) if mask.any() else None)
        summary_regions[actor]=row
    summary=dict(version=1,task='WS-V82-V4-INITIALIZATION-AUDIT-20261022',code_reference='c9385395',scene='scene0008_089_137',anchor_raw=102,frames=list(range(89,138)),cell_m=.2,meter_to_model=a.scale,plane_anchor=plane.tolist(),scope='initialization only; no trained V4 result',
        counts=dict(candidate_cells=original_count,common_audit_grid_cells=N,outside_original_candidate_grid=int(outside.sum()),created=int(supported.sum()),unsupported=int((~supported&~outside).sum()),observed=int(observed.sum()),connected_prior=int(prior.sum()),supported_unknown=int((supported&~observed&~prior).sum()),R_train=int(observed.sum()),RP_train=int((observed|prior).sum()),output_step0=0),
        V4_native_initial=native_meta,V4_V3_max_abs=geometry,V4_V3_rotation_max_degrees=rotation_error,V4_V3_comparison='same cells/positions/scales; explicit small float32 quaternion difference from normal normalization order; separate raster maps, not bitwise identical claim',ground_support=dict(frames=49,all_raw_ground_points=all_ground,used_dedup_points=len(points),xy_dedup_m=.1,radius_m=.8,minimum_k8_points=3),
        regions=summary_regions,region_definitions=regions,gate_names=gates,geometry_parameters_fixed=True,trainable_parameters=params,
        map_precision=dict(alpha='uint16/65535',depth='uint16 centimetres;0 means unavailable',statistics='full float32 before display quantization',domain='only fixed raw102 a5/a37/a6 context projection union;outside is not exported,not zero coverage evidence',source_RGB='only actual positive-weight sources packed;rejected values omitted while all49 gate flags retained'),
        thresholds=dict(alpha50=.5,alpha95=.95,geometric_footprint='horizontal projection3sigma,16fixed probes per0.2m cell;not rendered alpha',capacity='diagnostic counterfactual opacity=1,timeweight=1 with native0.99 alpha clamp/early termination;not a strict pointwise upper bound,training or output result'),
        world_region_denominator='fixed raw102 world OBB horizontal core and separate1.2m halo;common0.2m grid includes cells outside original candidate grid, marked separately;no shrinking by support or prediction',
        pixel_denominator='projected common world cell quadrilateral union;fixed before prediction,not car mask/not hidden clean GT;independent road/static-subset rendering,no vehicle occlusion',
        GPU_used=False,optimizer_updates=0,training_started_by_audit=False,human_verdict=None,quality_accepted=False,failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none')
    camera_meta=[]
    for raw in range(89,138):
        cam=a.camera(raw);camera_meta.append(dict(raw=raw,W=cam['W'].tolist(),K=cam['K'].tolist(),timestamp=cam['timestamp']))
    write_js(root/'data.js','window.V4AUDIT',dict(summary=summary,data=pack(arrays),cameras=camera_meta,model_to_world=M.tolist(),coordinate_anchor=anchor.tolist()))
    # 每格四角采用该格局部切平面，投影分母与模型输出无关。
    model_axes=a.A[:3,:3]@anchor[:3,:3];normal_local=np.asarray(z['normal_world'])@anchor[:3,:3]
    corner_offset=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    dz=-np.einsum('ni,ki->nk',normal_local[:,:2],corner_offset)/np.where(abs(normal_local[:,2])>1e-8,normal_local[:,2],1)[:,None]
    corner_local=np.concatenate([np.broadcast_to(corner_offset,(N,4,2)),dz[:,:,None]],axis=2)
    corners=z['model_xyz'][:,None,:]+corner_local@model_axes.T
    previous=json.loads((root/'receipt.json').read_text()) if resume and (root/'receipt.json').exists() else {}
    receipt=dict(status='RUNNING',summary=summary,frames={},V4_V3_alpha_max_error=previous.get('V4_V3_alpha_max_error',0.),GPU_used=False,optimizer_updates=0)
    masks={'native':None,'all':np.ones(len(unit_ids),bool),'v3':np.ones(len(unit_ids),bool),'R':observed[unit_ids],'RP':(observed|prior)[unit_ids]}
    for raw in raws:
        t0=time.time();stats_path=root/f'stats_{raw}.json';map_path=root/f'frame_{raw}.js'
        if resume and stats_path.exists() and map_path.exists():receipt['frames'][str(raw)]=json.loads(stats_path.read_text());continue
        maps={};full={}
        for name,mask in masks.items():
            use=native_init if name=='native' else v3 if name=='v3' else model
            alpha,depth=render(a,use,raw,root,None if name=='native' else mask)
            capacity,_=render(a,use,raw,root,None if name=='native' else mask,True)
            # 原生早停在接受最后一项之前退出，全1参考不是逐像素严格上界；只核对95%覆盖能力。
            assert not np.any((alpha>=.95)&(capacity<.95)),(name,raw,'capacity95 regression')
            maps['alpha_'+name]=alpha;maps['depth_'+name]=depth;maps['cap_'+name]=capacity;full[name]=(alpha,depth,capacity)
        # 以独立V3初始化公式做每帧真数值回归，不只比较点数。
        err=float(abs(full['v3'][0]-full['all'][0]).max());receipt['V4_V3_alpha_max_error']=max(receipt['V4_V3_alpha_max_error'],err)
        maps['lidar_depth']=measured_frame(a,raw)['lidar_nearest_m']
        cam=a.camera(raw);frame_stats={}
        for actor,region in regions.items():
            frame_stats[actor]={}
            for scope,halo in [('core',0),('context',1.2)]:
                selected=in_region(xy,region,halo);pixel_mask,excluded=projected_mask(corners[selected],cam['W'],cam['K']);arrays_mask=pixel_mask.astype(np.uint8)
                maps[f'mask_{actor}_{scope}']=arrays_mask
                frame_stats[actor][scope]=dict(near_clipped_cells=excluded,models={name:pixel_metrics(*data[:2],pixel_mask,data[2]) for name,data in full.items()})
        # 只打包预先固定的三个世界区域投影并集；计算时仍光栅全部高斯，保留外部覆盖贡献。
        # 区外没有导出，不能当作alpha=0的测量。每个共同分母及全量浮点统计保留。
        domain=np.logical_or.reduce([maps[f'mask_{actor}_context'].astype(bool) for actor in regions])
        packed=arrays_to_js_maps({k:np.where(domain,v,0) for k,v in maps.items() if not k.startswith('mask_')})
        packed.update({k:v for k,v in maps.items() if k.startswith('mask_')});packed['domain_mask']=domain.astype(np.uint8)
        write_js(map_path,f'window.V4AUDITFRAMES=window.V4AUDITFRAMES||{{}};window.V4AUDITFRAMES[{raw}]',dict(**pack(packed),stats=frame_stats))
        write_json(stats_path,frame_stats);receipt['frames'][str(raw)]=frame_stats
        print(json.dumps(dict(raw=raw,seconds=round(time.time()-t0,2),native_count=len(native_init['xyz']),V4_V3_alpha_max_error=err)),flush=True)
        write_json(root/'receipt.json',receipt)
    receipt.update(status='COMPLETE',elapsed_seconds=round(time.time()-started,2),frame_count=len(receipt['frames']),full_float_statistics=True,display_only_quantization=True)
    if torch.cuda.is_initialized():raise AssertionError('read-only CPU audit initialized CUDA')
    write_json(root/'summary.json',summary);write_json(root/'receipt.json',receipt)
    for name in ['v4_initialization.html','v4_initialization.js','v4_initialization.css','v4_embed.js','export_v4_initialization.ps1']:
        if (HERE/name).exists():shutil.copy2(HERE/name,output/name)
    index=output/'index.html'
    if index.exists():
        content=index.read_text(encoding='utf-8')
        if 'v4_initialization.html' not in content and '#v4-initialization' not in content:content=content.replace('</nav>','<a href="v4_initialization.html">V4初始化验收</a></nav>',1)
        if 'v4_embed.js' not in content:content=content.replace('</body>','<script src="v4_embed.js?v=embed1"></script></body>',1)
        index.write_text(content,encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],counts=summary['counts'],regions=summary_regions,seconds=receipt['elapsed_seconds']),ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--campaign',type=Path,default=CAMPAIGN);p.add_argument('--raws',nargs='+',type=int);p.add_argument('--resume',action='store_true');o=p.parse_args()
    export(o.output,o.campaign,o.raws or range(89,138),o.resume)
