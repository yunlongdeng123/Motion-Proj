"""Read-only CPU audit of the frozen silver-car DrivingEditor implementation.

Only writes to this independent campaign audit directory. Does not import any
renderer, call CUDA, edit formal code/data, run training, or use GT actor boxes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import ast
import gc
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from plyfile import PlyData
from PIL import Image, ImageDraw

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
OUT = BASE/'campaign/vgpu_bugfix_20261012/sgcm_data_audit'
DATA = BASE/'data/waymo_paper_scenes/scene0017_000_048'
SRC = OUT/'frozen_source'
RUN = BASE/'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'
OUT.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(1)


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sg = load_module(SRC/'scene/paper_sgcm.py', 'frozen_sgcm_audit')


def read_ply(path):
    v = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([v[k] for k in ['x', 'y', 'z']]).astype(np.float64)
    time = np.asarray(v['nx'], np.float64) if 'nx' in v.data.dtype.names else None
    return xyz, time


def qsummary(values):
    a = np.asarray(values)
    return {str(p):float(np.quantile(a,p)) for p in [0,.01,.05,.25,.5,.75,.95,.99,1]} if len(a) else {}


def ast_functions(path, class_name, names):
    tree = ast.parse(path.read_text())
    container = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name) if class_name else tree
    chosen = [n for n in container.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    module = ast.Module(body=chosen, type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(path)+':isolated_cpu_functions', 'exec')
    return code


def tiny_cpu_checks(controller):
    # Invoke actual frozen optimizer routines through a CPU-only adapter. The
    # sole adapter intercepts four postfix zeros(device='cuda'); mathematical
    # code, parent indices, parameter order, and optimizer transforms are exact.
    class TorchCPUAdapter:
        def __getattr__(self, name):
            return getattr(torch, name)
        def zeros(self, *args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return torch.zeros(*args, **kwargs)
    ns = dict(torch=TorchCPUAdapter(), nn=nn, np=np,
              inverse_sigmoid=lambda x:torch.log(x/(1-x)))
    names = ['replace_tensor_to_optimizer','_prune_optimizer','prune_points',
             'cat_tensors_to_optimizer','densification_postfix','reset_opacity']
    exec(ast_functions(SRC/'scene/gaussian_model.py','GaussianModel',names),ns)
    class Harness:
        get_xyz = property(lambda self:self._xyz)
        get_scaling = property(lambda self:torch.exp(self._scaling))
        get_opacity = property(lambda self:torch.sigmoid(self._opacity))
        scaling_inverse_activation = staticmethod(torch.log)
    for name in names:
        setattr(Harness,name,ns[name])
    h = Harness()
    fields = ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_t','_scaling_t','_velocity']
    optnames = ['xyz','f_dc','f_rest','opacity','scaling','rotation','t','scaling_t','velocity']
    # 0=prune at .8m; 1=clone at .3m; 2=split at .1m;
    # 3=unchanged .1m with small footprint; 4=outside region.
    xyz = torch.tensor([[0,0,.8],[0,0,.3],[0,0,.1],[0,0,.1],[10,0,.8]],dtype=torch.float64)
    vals = [xyz,torch.arange(5,dtype=torch.float64)[:,None,None]+10,
            torch.arange(15,dtype=torch.float64).reshape(5,1,3)+100,
            torch.zeros(5,1,dtype=torch.float64),
            torch.log(torch.tensor([[.1,.2,.3]]*5,dtype=torch.float64)),
            torch.tensor([[1.,0,0,0]]*5,dtype=torch.float64),
            torch.arange(5,dtype=torch.float64)[:,None],
            torch.zeros(5,1,dtype=torch.float64),
            torch.arange(15,dtype=torch.float64).reshape(5,3)]
    for field,val in zip(fields,vals):
        setattr(h,field,nn.Parameter(val.clone()))
    h.optimizer = torch.optim.Adam([{'params':[getattr(h,f)],'name':n,'lr':.001} for f,n in zip(fields,optnames)])
    for idx,field in enumerate(fields):
        p = getattr(h,field)
        h.optimizer.state[p] = {'step':torch.tensor(7.),'exp_avg':torch.ones_like(p)*(idx+1),
                                'exp_avg_sq':torch.ones_like(p)*(idx+11)}
    oldmom = {f:h.optimizer.state[getattr(h,f)]['exp_avg'].clone() for f in fields}
    for name in ['max_radii2D','xyz_gradient_accum','t_gradient_accum','denom']:
        setattr(h,name,torch.arange(5,dtype=torch.float64) if name=='max_radii2D' else torch.arange(5,dtype=torch.float64)[:,None])
    c = sg.SGCMController(sg.SGCMConfig(start=0,end=1000,offset=0),1,np.array([0.,0,1,0]),np.eye(4),
                          [np.array([[-1.,-1.,-1.],[1.,1.,1.]])],{})
    stats = c.refine(h,torch.tensor([1.,1.,30.,1.,1.],dtype=torch.float64),100)
    expected_parent = [1,3,4,1,2,2]
    feature_ok = bool(torch.equal(h._features_dc.detach(),vals[1][expected_parent]))
    temporal_ok = all(torch.equal(getattr(h,f).detach(),vals[fields.index(f)][expected_parent]) for f in ['_t','_scaling_t','_velocity'])
    optimizer_ok = all(h.optimizer.param_groups[i]['params'][0] is getattr(h,f) and
                       torch.equal(h.optimizer.state[getattr(h,f)]['exp_avg'][:3],oldmom[f][[1,3,4]]) and
                       torch.count_nonzero(h.optimizer.state[getattr(h,f)]['exp_avg'][3:])==0 and
                       int(h.optimizer.state[getattr(h,f)]['step'])==7 for i,f in enumerate(fields))
    stats_ok = torch.equal(h.denom[:3,0],torch.tensor([1.,3.,4.],dtype=torch.float64)) and torch.count_nonzero(h.denom[3:])==0
    # Identity quaternion: split axis must be z, inherited SH must match parent2.
    expected_offset=.3/np.sqrt(2)
    split_axis_ok = np.allclose(h._xyz.detach().numpy()[-2:], [[0,0,.1-expected_offset],[0,0,.1+expected_offset]])
    split_sigma_ok = np.allclose(torch.exp(h._scaling).detach().numpy()[-2:],[[.1,.2,expected_offset]]*2)
    tiny = {'stats':stats,'feature_and_SH_parent_order_pass':feature_ok,'temporal_copy_pass':temporal_ok,
            'survivor_Adam_moments_and_new_zero_moments_pass':bool(optimizer_ok),
            'statistics_survivor_index_pass':bool(stats_ok),'split_axis_quaternion_pass':bool(split_axis_ok),
            'split_scaling_pass':bool(split_sigma_ok),
            'scope':'Exact frozen routines with device-only CPU allocation adapter; no CUDA.'}
    assert all(tiny[k] for k in tiny if k.endswith('_pass'))
    # Frozen schedule function: zero densification needed at30k, but static resets.
    ns2={}
    exec(ast_functions(SRC/'train.py',None,['densification_and_optimization']),ns2)
    schedule=[]
    class Counter:
        get_xyz=torch.zeros(1,3)
        def __init__(self): self.resets=0
        def reset_opacity(self): self.resets+=1
    args=SimpleNamespace(dynamic_densify_until_iter=15000,densify_until_iter=15000,densify_until_num_points=3000000,
                         dynamic_densify_until_num_points=1000000,opacity_reset_interval=3000,
                         white_background=False,densify_from_iter=500)
    for it in [15000,18000,20000,21000,24000,27000,29999,30000,33000,36000,48000,50000]:
        s,d=Counter(),Counter()
        ns2['densification_and_optimization'](s,args,it,None,None,None,None,False)
        ns2['densification_and_optimization'](d,args,it,None,None,None,None,True)
        schedule.append({'iteration':it,'static_resets':s.resets,'dynamic_resets':d.resets,'sgcm_due':controller.due(it)})
    tiny['post_densify_reset_schedule']=schedule
    tiny['cuda_initialized']=torch.cuda.is_initialized()
    assert not tiny['cuda_initialized']
    save('tiny_cpu_checks.json',tiny)


def sgcm_and_data():
    class Args:
        def __init__(self,data): self.__dict__.update(data)
    import yaml
    args=Args(yaml.safe_load((RUN/'effective_config.yaml').read_text()))
    controller=sg.create_sgcm_controller(DATA,args)
    save('sgcm_runtime_geometry.json',controller.metadata)
    hashes={str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.rglob('*.py'))}
    save('source_hashes.json',hashes)
    tiny_cpu_checks(controller)
    static_world,static_time=read_ply(DATA/'points3d.ply')
    dynamic_world,dynamic_time=read_ply(DATA/'points3d_dynamic_0.ply')
    anchor_to_model=np.linalg.inv(controller.model_to_anchor)
    firstpose=np.loadtxt(DATA/'pose/0017000.txt')
    world_to_anchor=np.linalg.inv(firstpose)
    initial_static_dist,_=cKDTree(dynamic_world).query(static_world,k=1,distance_upper_bound=.1,workers=1)
    keep=initial_static_dist>=.1
    model=sg.transform_points(static_world[keep],np.array(controller.metadata['world_to_model']))
    classification=controller.classify_numpy(model,np.zeros(len(model)))
    dynamic_anchor=sg.transform_points(dynamic_world,world_to_anchor)
    dynamic_ground_distance=np.abs(dynamic_anchor@np.array(controller.metadata['ground_fit']['plane_anchor'])[:3]+
                                   controller.metadata['ground_fit']['plane_anchor'][3])
    static_anchor=sg.transform_points(static_world[keep],world_to_anchor)
    signed=static_anchor@np.array(controller.metadata['ground_fit']['plane_anchor'])[:3]+controller.metadata['ground_fit']['plane_anchor'][3]
    prep=json.loads((DATA/'preparation_manifest.json').read_text())
    actorprep=json.loads((DATA/'dynamic_actor_preparation.json').read_text())
    framestats=[]
    for row in actorprep['frames']:
        frame=row['frame']; t=float(row['model_time'])
        these=np.isclose(dynamic_time,t,atol=1e-7)
        framestats.append({'frame':frame,'model_time':t,'points_from_PLY_time':int(these.sum()),
                          'manifest_retained_points':row['retained_visible_lidar_points'],
                          'points_ground_distance_lt_015m':int((dynamic_ground_distance[these]<.15).sum()),
                          'ground_distance_quantiles_m':qsummary(dynamic_ground_distance[these])})
    summary={'static_PLY_points':len(static_world),'dynamic_PLY_points':len(dynamic_world),
             'static_excluded_near_dynamic':int((~keep).sum()),'static_initialized':int(keep.sum()),
             'initial_SGCM_counts':{k:int(v.sum()) for k,v in classification.items() if v.dtype==bool},
             'initial_static_signed_ground_distance_quantiles_m':qsummary(signed),
             'initial_static_inside_region_ground_distance_quantiles_m':qsummary(signed[classification['inside']]),
             'dynamic_ground_distance_quantiles_m':qsummary(dynamic_ground_distance),
             'dynamic_ground_lt_015m':int((dynamic_ground_distance<.15).sum()),
             'dynamic_ground_lt_025m':int((dynamic_ground_distance<.25).sum()),
             'dynamic_ground_lt_050m':int((dynamic_ground_distance<.5).sum()),
             'dynamic_times_unique':len(np.unique(dynamic_time)),
             'static_times_unique':len(np.unique(static_time)),
             'per_frame':framestats,
             'time_assignment_matches_actor_manifest':all(x['points_from_PLY_time']==x['manifest_retained_points'] for x in framestats),
             'ground_proximity_caveat':'Ground-plane proximity alone does not classify actor leakage: tires and lower car body legitimately approach road. Pixel/RGB and raw-return evidence required.'}
    save('initial_geometry_and_time.json',summary)
    # Parse logged SGCM operations by regex, avoiding CR-delimited tqdm noise.
    logs=[BASE/'campaign/resume_vgpu_baseline_20261011/resume_training.log',BASE/'campaign/failures/oom_20261011T062945/logs/paper50k.log']
    allstats=[]
    for log in logs:
        if not log.exists(): continue
        text=log.read_text(errors='replace')
        rows=[json.loads(x) for x in re.findall(r'SGCM_STATS (\{[^\r\n]+\})',text)]
        allstats.append({'path':str(log),'count':len(rows),'first':rows[:3],'last':rows[-3:],
                         'sums':{k:sum(x.get(k,0) for x in rows) for k in ['pruned','cloned','split_parents']},
                         'rows':rows})
    save('sgcm_log_summary.json',allstats)
    checkpoints=[]
    paths=[BASE/'runs/scene0017_000_048_paper_50000_seed0/chkpnt10000.pth',RUN/'chkpnt20000.pth',RUN/'chkpnt30000.pth']
    for path in paths:
        cp,it=torch.load(path,map_location='cpu')
        xyz=cp[1].detach().numpy().astype(np.float64)
        opacity=torch.sigmoid(cp[6]).detach().numpy()[:,0]
        cl=controller.classify_numpy(xyz,cp[10].detach().numpy())
        inr=cl['inside']
        anchor=sg.transform_points(xyz,controller.model_to_anchor)
        velocity=cp[9].detach().numpy()/controller.model_scale
        times=cp[7].detach().numpy()[:,0]
        scale_t=np.exp(cp[8].detach().numpy()[:,0])
        p={'path':str(path),'iteration':it,'static_points':len(xyz),
           'SGCM_counts_at_checkpoint':{k:int(v.sum()) for k,v in cl.items() if v.dtype==bool},
           'opacity_quantiles':qsummary(opacity),'opacity_gt_001':int((opacity>.0100001).sum()),
           'inside_region_distance_quantiles_m':qsummary(cl['distance_m'][inr]),
           'inside_region_opacity_quantiles':qsummary(opacity[inr]),
           'velocity_norm_quantiles_m_per_model_time':qsummary(np.linalg.norm(velocity,axis=1)),
           'inside_region_velocity_norm_quantiles':qsummary(np.linalg.norm(velocity[inr],axis=1)),
           'time_center_quantiles':qsummary(times),'temporal_sigma_quantiles':qsummary(scale_t),
           'spatial_scales_quantiles_m':qsummary(np.exp(cp[4].detach().numpy()).max(axis=1)/controller.model_scale)}
        checkpoints.append(p)
        del cp,xyz,opacity,cl,anchor,velocity,times,scale_t
        gc.collect()
    save('checkpoint_geometry.json',checkpoints)
    print(json.dumps({'phase':'sgcm_data_done','initial':{k:v for k,v in summary.items() if k!='per_frame'},
                      'tiny_cpu_pass':True,'checkpoints':[{k:v for k,v in p.items() if k in ['iteration','static_points','SGCM_counts_at_checkpoint','opacity_gt_001']} for p in checkpoints]},ensure_ascii=False),flush=True)
    return controller


def calibration(frame):
    values={}
    for line in (DATA/'calib'/f'{frame}.txt').read_text().splitlines():
        name,val=line.split(':',1)
        values[name]=np.fromstring(val,sep=' ')
    K=values['P0'].reshape(3,4)[:,:3]
    tr=np.eye(4); tr[:3]=values['Tr_velo_to_cam_0'].reshape(3,4)
    ego=np.loadtxt(DATA/'pose'/f'{frame}.txt')
    return K,tr,ego


def project(world,K,world_to_cam,W,H):
    xyz=sg.transform_points(world,world_to_cam)
    uv=xyz@K.T
    uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
    rounded=np.rint(uv).astype(np.int64)
    inside=(xyz[:,2]>0)&(rounded[:,0]>=0)&(rounded[:,0]<W)&(rounded[:,1]>=0)&(rounded[:,1]<H)
    return xyz,uv,rounded,inside


def coverage(controller):
    frames=sorted(p.stem for p in (DATA/'pose').glob('*.txt'))
    firstpose=np.loadtxt(DATA/'pose'/f'{frames[0]}.txt')
    world_to_anchor=np.linalg.inv(firstpose)
    plane=np.array(controller.metadata['ground_fit']['plane_anchor'])
    region=np.array(controller.regions_anchor[0]); lower=region[0,:2]-3; upper=region[1,:2]+3
    observation=[]
    samples=[]
    manifest=json.loads((DATA/'dynamic_actor_preparation.json').read_text())
    alignment=[]
    for frame in frames:
        K,tr,ego=calibration(frame)
        rgb=np.asarray(Image.open(DATA/'image/image_0'/f'{frame}.png').convert('RGB'))
        mask=np.asarray(Image.open(DATA/'dynamic_masks/dynamic_mask_0'/f'{frame}.png'))>0
        if mask.ndim==3: mask=mask[...,0]
        sky=np.asarray(Image.open(DATA/'sky/sky_0'/f'{frame}.png'))
        sky=(sky[...,0] if sky.ndim==3 else sky)>0
        dynrgb=np.asarray(Image.open(DATA/'dynamic_image/dynamic_image_0'/f'{frame}.png').convert('RGB'))
        align={'frame':frame,'size':list(rgb.shape),'mask_pixels':int(mask.sum()),
               'mask_equals_nonzero_dynamic_RGB':bool(np.array_equal(mask,np.any(dynrgb!=0,axis=2))),
               'masked_RGB_equal_input':bool(np.array_equal(dynrgb,rgb*mask[...,None])),
               'mask_sky_overlap_pixels':int((mask&sky).sum())}
        alignment.append(align)
        H,W=mask.shape
        velo=np.fromfile(DATA/'velodyne'/f'{frame}.bin',dtype=np.float32).reshape(-1,6)[:,:3].astype(np.float64)
        world=sg.transform_points(velo,ego)
        anchor=sg.transform_points(world,world_to_anchor)
        ground=(np.abs(anchor@plane[:3]+plane[3])<=.15)
        pointcam,_,rounduv,inframe=project(world,K,tr@np.linalg.inv(ego),W,H)
        binsize=8
        bins=np.full(((H+binsize-1)//binsize,(W+binsize-1)//binsize),np.inf)
        inds=rounduv[inframe]
        np.minimum.at(bins,(inds[:,1]//binsize,inds[:,0]//binsize),pointcam[inframe,2])
        observation.append({'frame':frame,'K':K,'w2c':tr@np.linalg.inv(ego),'mask':mask,'sky':sky,
                            'groundtree':cKDTree(world[ground]),'bins':bins,'size':(W,H),'binsize':binsize})
        # Four-pixel grid through the actual actor silhouette. Fit-ground ray
        # intersections define candidate occluded-road points, not GT road.
        yy,xx=np.mgrid[0:H:4,0:W:4]
        selected=mask[yy,xx]
        pixels=np.column_stack((xx[selected],yy[selected]))
        rays=np.column_stack((pixels,np.ones(len(pixels))))@np.linalg.inv(K).T
        cam2world=ego@np.linalg.inv(tr)
        cam2anchor=world_to_anchor@cam2world
        rays_anchor=rays@cam2anchor[:3,:3].T
        origin=cam2anchor[:3,3]
        denom=rays_anchor@plane[:3]
        raydepth=-(origin@plane[:3]+plane[3])/denom
        anchorpoints=origin[None]+raydepth[:,None]*rays_anchor
        valid=(raydepth>0)&(raydepth<100)&np.isfinite(anchorpoints).all(axis=1)&np.all((anchorpoints[:,:2]>=lower)&(anchorpoints[:,:2]<=upper),axis=1)
        worldpoints=sg.transform_points(anchorpoints[valid],firstpose)
        samples.append({'frame':frame,'world':worldpoints,'pixels':pixels[valid],
                        'actor_sample_pixels':len(pixels),'road_candidate_count':int(valid.sum())})
    save('mask_alignment_49frames.json',{'all_exact_masked_RGB':all(x['masked_RGB_equal_input'] for x in alignment),
                                         'total_mask_sky_overlap':sum(x['mask_sky_overlap_pixels'] for x in alignment),
                                         'rows':alignment})
    rows=[]
    detailed=[]
    for qi,sample in enumerate(samples):
        world=sample['world']; n=len(world)
        candidate=np.zeros((49,n),dtype=bool); confirmed=candidate.copy()
        for oi,ob in enumerate(observation):
            if oi==qi: continue
            W,H=ob['size']
            xyz,uv,xy,inside=project(world,ob['K'],ob['w2c'],W,H)
            ids=np.flatnonzero(inside)
            if not len(ids): continue
            px,py=xy[ids].T
            ids=ids[~ob['mask'][py,px]&~ob['sky'][py,px]]
            if not len(ids): continue
            px,py=xy[ids].T
            depthmin=ob['bins'][py//ob['binsize'],px//ob['binsize']]
            # Reject if a measured nearer surface lies along the same coarse bin.
            ids=ids[depthmin>=xyz[ids,2]-.3]
            candidate[oi,ids]=True
            near,_=ob['groundtree'].query(world[ids],k=1,distance_upper_bound=.35,workers=1)
            confirmed[oi,ids[np.isfinite(near)]]=True
        # Measure independent frame observations, no imputed pixels.
        num_candidate=candidate.sum(axis=0); num_confirm=confirmed.sum(axis=0)
        row={'query_frame':sample['frame'],'actor_grid_samples':sample['actor_sample_pixels'],
             'nearby_ground_plane_candidate_samples':n,
             'observed_outside_actor_with_no_measured_nearer_occluder_samples':int((num_candidate>0).sum()),
             'also_supported_by_nearby_real_ground_LiDAR_samples':int((num_confirm>0).sum()),
             'confirmed_fraction':float((num_confirm>0).mean()) if n else None,
             'confirmed_at_least_3_frames':int((num_confirm>=3).sum()),
             'max_independent_observing_frames':int(num_confirm.max()) if n else 0,
             'best_support_frame':frames[int(confirmed.sum(axis=1).argmax())] if n else None,
             'best_support_samples':int(confirmed.sum(axis=1).max()) if n else 0}
        rows.append(row)
        if qi in [0,10,17,24,40,48]:
            np.savez_compressed(OUT/f'coverage_{sample["frame"]}.npz',world=world,pixels=sample['pixels'],
                                candidate=candidate,confirmed=confirmed,frames=frames)
            best=int(confirmed.sum(axis=1).argmax())
            bestobs=observation[best]
            imgq=Image.open(DATA/'image/image_0'/f'{sample["frame"]}.png').convert('RGB')
            imgc=Image.open(DATA/'image/image_0'/f'{frames[best]}.png').convert('RGB')
            drawq=ImageDraw.Draw(imgq); drawc=ImageDraw.Draw(imgc)
            ids=np.flatnonzero(confirmed[best])
            _,_,supportxy,_=project(world[ids],bestobs['K'],bestobs['w2c'],*bestobs['size'])
            for p in sample['pixels'][ids][::max(1,len(ids)//700)]:
                x,y=map(int,p);drawq.ellipse((x-3,y-3,x+3,y+3),fill=(50,255,80))
            for p in supportxy[::max(1,len(ids)//700)]:
                x,y=map(int,p);drawc.ellipse((x-3,y-3,x+3,y+3),fill=(50,255,80))
            imgq.thumbnail((768,512));imgc.thumbnail((768,512))
            panel=Image.new('RGB',(1536,550),(20,20,20));panel.paste(imgq,(0,35));panel.paste(imgc,(768,35))
            dp=ImageDraw.Draw(panel)
            dp.text((10,8),f'Query {sample["frame"]}: green rays through car -> fitted road',fill='white')
            dp.text((778,8),f'Observed {frames[best]}: same 3D road points, mask-free + LiDAR support',fill='white')
            panel.save(OUT/f'coverage_evidence_{sample["frame"]}.jpg',quality=92)
            detailed.append(row)
        print(json.dumps({'phase':'coverage','query':sample['frame'],'fraction':row['confirmed_fraction']},ensure_ascii=False),flush=True)
    save('road_observation_coverage.json',{'method':
         'Ray-plane intersections through 4px-grid actor-mask pixels, limited to validated actor-neighborhood plane (region XY plus3m), positive camera depth<100m. Reproject to each of48 independent RGB frames. Candidate requires in-frame, outside silver-body mask, non-sky, and no closer measured LiDAR in8px bin beyond0.3m tolerance. Confirmed also requires actual same-frame LiDAR within0.35m whose distance to fitted ground<=0.15m.',
         'no_gt_boxes':True,'gpu_used':False,'not_a_groundtruth_image':True,
         'limitations':['Fitted-plane samples are a geometric diagnostic and cannot supply dense edited-image GT.',
                        'The supplied dynamic mask covers one car body, so unselected occluders are excluded only when measured by LiDAR; visual evidence must be checked.',
                        '0.15m plane tolerance,8px bin,.3m depth tolerance and .35m nearby-return tolerance are audit parameters, not training thresholds.',
                        'No static feature correspondence alone is used to label visibility.'],
         'rows':rows,'highlight_queries':detailed,
         'at_least_one_query_has_independent_observed_road':any(x['also_supported_by_nearby_real_ground_LiDAR_samples']>0 for x in rows),
         'cuda_initialized':torch.cuda.is_initialized()})


if __name__=='__main__':
    controller=sgcm_and_data()
    coverage(controller)
    assert not torch.cuda.is_initialized()
    print('CPU_AUDIT_COMPLETE',flush=True)
