"""离线初始化/49帧地面证据与深度标记；不修改冻结训练输入。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,gc,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from cloud_sampling import Sampler,pack
from backend import Adapter,CAMPAIGN,ARMS,clean
from ray_reference import project,covariance,candidates,composite

def js(root,path,key,value):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.js.tmp');temp.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(clean(value),separators=(',',':'),allow_nan=False)+';')
    temp.replace(target)

def ground_history(a,root):
    meta=a.meta;engineering=Path(meta['original_engineering']);source=engineering.parent/'inputs/scene0008_089_137'
    ready=json.loads(Path(meta['original_readiness']).read_text());plane=np.array(ready['SGCM_geometry']['ground_fit']['plane_anchor'])
    anchor=np.array(meta['coordinate_anchor']);inv=np.linalg.inv(anchor);parts=[];raws=[];counts={}
    for raw in range(89,138):
        fid=f'0008{raw:03d}';pose=np.loadtxt(source/'pose'/f'{fid}.txt');points=np.fromfile(source/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(float)
        world=points@pose[:3,:3].T+pose[:3,3];local=world@inv[:3,:3].T+inv[:3,3];occupied=np.zeros(len(world),bool)
        for label in json.loads((source/'annotations'/f'{fid}.json').read_text())['laser_labels']:
            b=label['box'];half=np.array([b['length'],b['width'],b['height']])/2
            if min(half)<=0:continue
            th=b['heading'];c,s=np.cos(th),np.sin(th);axes=pose[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
            center=pose[:3,:3]@np.array([b['center_x'],b['center_y'],b['center_z']])+pose[:3,3]
            occupied|=(np.abs((world-center)@axes)<=half).all(1)
        ground=(abs(local@plane[:3]+plane[3])<=.15)&~occupied&(local[:,0]>=3)&(local[:,0]<=90)&(abs(local[:,1])<=8)
        parts.append(local[ground]);raws.append(np.full(ground.sum(),raw,np.int32));counts[str(raw)]=int(ground.sum())
    allpoints=np.concatenate(parts);allraw=np.concatenate(raws)
    _,indices=np.unique(np.floor(allpoints[:,:2]/.1).astype(int),axis=0,return_index=True)
    points=allpoints[indices];raws=allraw[indices];tree=cKDTree(points[:,:2]);z=np.load(a.bundle.parent/meta['units'])
    xy=z['xy_local'];dist,idx=tree.query(xy,k=8,distance_upper_bound=meta['policy']['surface_support_radius_m']);used=np.isfinite(dist).sum(1)
    assert np.array_equal(used>=3,z['surface_supported']), 'initial ground support reconstruction mismatch'
    nearest=tree.query(xy)[0];assert np.allclose(nearest,z['surface_distance_m'],atol=1e-5)
    all_count=np.array([len(v) for v in tree.query_ball_point(xy,.8)],np.int32)
    normal=z['normal_model'].astype(np.float32);q=np.c_[1+normal[:,2],-normal[:,1],normal[:,0],np.zeros(len(normal))].astype(np.float32);bad=np.linalg.norm(q,axis=1)<1e-8;q[bad]=[0,1,0,0];q/=np.linalg.norm(q,axis=1)[:,None]
    world=points@anchor[:3,:3].T+anchor[:3,3];model=world@a.A[:3,:3].T+a.A[:3,3]
    arrays=dict(init_xyz=z['model_xyz'].astype(np.float32),init_q=q,init_xy=xy.astype(np.float32),supported=z['surface_supported'].astype(np.uint8),
        ground_xyz=model.astype(np.float32),ground_xy=points[:,:2].astype(np.float32),ground_raw=raws.astype(np.uint16),support_count=all_count,used_k8_count=used.astype(np.uint8),
        nearest_m=nearest.astype(np.float32),eighth_m=np.where(np.isfinite(dist[:,-1]),dist[:,-1],-1).astype(np.float32))
    metadata=dict(frames=49,raw_start=89,raw_end=137,raw_ground_count=len(allpoints),used_ground_count=len(points),per_frame_raw_count=counts,
        per_frame_used_count={str(r):int((raws==r).sum()) for r in range(89,138)},radius_m=.8,minimum_neighbors=3,k_nearest_cap=8,dedup_xy_m=.1,cell_m=meta['policy']['cell_m'],
        initial_opacity=.5,initial_sigma_m=[meta['policy']['cell_m']*.65]*2+[meta['policy']['normal_sigma_m']],initial_color=meta['policy']['initial_color'],
        cell_axes_model=(a.A[:3,:3]@anchor[:3,:3]*meta['policy']['cell_m']/2).tolist(),
        source='frozen49 original raw LiDAR, exact original all-3D-box exclusion/plane/corridor/10cm dedup; matched all frozen support flags and distances',
        initialization='deterministic reconstruction from frozen bundle and original install_road_units; not a saved step0 checkpoint')
    js(root,'data/road_history.js','roadHistory',dict(**pack(arrays),metadata=metadata));z.close()
    return metadata

def measured_frame(a,raw):
    cam=a.camera(raw);source=Path(a.config['source_path']);fid=f'0008{raw:03d}'
    pts=np.fromfile(source/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(float);pose=np.loadtxt(source/'pose'/f'{fid}.txt')
    world=pts@pose[:3,:3].T+pose[:3,3];W=np.array(cam['meta']['world_to_camera']);cp=world@W[:3,:3].T+W[:3,3];K=cam['K']
    uv=cp[:,:2]/cp[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];valid=(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)
    pixels=uv[valid].astype(int);depth=np.full((320,480),np.inf);np.minimum.at(depth,(pixels[:,1],pixels[:,0]),cp[valid,2]);depth[~np.isfinite(depth)]=0
    yy,xx=np.where(depth>0);sel=np.linspace(0,len(xx)-1,min(6000,len(xx))).astype(int);x,y=xx[sel],yy[sel];z=depth[y,x];v=np.c_[(x+.5-K[0,2])/K[0,0]*z,(y+.5-K[1,2])/K[1,1]*z,z]*a.scale
    inv=np.linalg.inv(cam['W']);xyz=v@inv[:3,:3].T+inv[:3,3]
    return dict(lidar_nearest_m=depth.astype(np.float32),current_xyz=xyz.astype(np.float32),current_pixel=np.c_[x,y].astype(np.uint16),current_total=np.array([len(xx)],np.int32))

def prepare_depth_numpy(a,b,raw):
    cam=a.camera(raw);parts=[]
    for start in range(0,len(b['xyz']),100000):
        sl=slice(start,start+100000);delta=cam['timestamp']-b['t'][sl];omega=np.float32(2*np.pi/b['cycle']);m=np.exp(-.5*(delta/b['sigma_t'][sl])**2);xyz=b['xyz'][sl]+b['vel'][sl]*(np.sin(delta*omega)/omega)[:,None]
        road=b['road'][sl]>=0;m[road]=1;xyz[road]=b['xyz'][sl][road]
        p=project(xyz,covariance(b['scale'][sl],b['q'][sl]),b['opacity'][sl]*m,m>.05,cam['W'],cam['K'],bool(a.config['neg_fov']))
        p['road']=road[p['row']];parts.append({k:p[k] for k in ['uv','conic','opacity','depth','lo','hi','road']})
    return {k:np.ascontiguousarray(np.concatenate([p[k] for p in parts])) for k in parts[0]}

def library(root):
    source=Path(__file__).with_name('depth_raster.cpp');lib=root/'depth_raster.so'
    if not lib.exists() or lib.stat().st_mtime<source.stat().st_mtime:subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    return ctypes.CDLL(str(lib))

def prepare_depth(a,b,raw,root,include_rows=False):
    cam=a.camera(raw);n=len(b['xyz']);arrays={k:np.empty((n,stride),dtype) if stride>1 else np.empty(n,dtype) for k,stride,dtype in [('uv',2,np.float32),('conic',3,np.float32),('opacity',1,np.float32),('depth',1,np.float32),('lo',2,np.int32),('hi',2,np.int32),('road',1,np.uint8)]}
    ins=[np.ascontiguousarray(b[k]) for k in ['xyz','scale','q','opacity','t','sigma_t','vel','road']];W=np.ascontiguousarray(cam['W'],np.float32);K=np.ascontiguousarray(cam['K'],np.float32)
    fn=library(root).project_depth_rows if include_rows else library(root).project_depth;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*8+[ctypes.c_float,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*7;fn.restype=ctypes.c_int
    if include_rows:
        arrays['source_row']=np.empty(n,np.int32);fn.argtypes.append(ctypes.c_void_p)
    count=fn(n,*[v.ctypes.data for v in ins],cam['timestamp'],b['cycle'],W.ctypes.data,K.ctypes.data,int(bool(a.config['neg_fov'])),*[v.ctypes.data for v in arrays.values()])
    out={k:np.ascontiguousarray(v[:count]) for k,v in arrays.items()};out['road']=out['road'].astype(bool);return out

def native_raster(root,p):
    fn=library(root).raster_depth;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*8;fn.restype=ctypes.c_int
    mask=~p['road'];inputs=[np.ascontiguousarray(p[k][mask]) for k in ['uv','conic','opacity','depth','lo','hi']]
    alpha=np.zeros((320,480),np.float32);ds=np.zeros_like(alpha);fn(len(inputs[0]),*[v.ctypes.data for v in inputs],alpha.ctypes.data,ds.ctypes.data)
    return alpha,ds

def export(root,arms,raws=range(89,138),resume=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);started=time.time();a=Adapter();sampler=Sampler(a);receipt=dict(status='INSPECTION_EXPORT_RUNNING',arms=arms,frames=list(raws),gpu_used=False,optimizer_updates=0,checks=[])
    if resume and (root/'data/road_history.js').exists():
        packed=json.loads((root/'data/road_history.js').read_text().split(']=',1)[1].rstrip(';'));receipt['ground_history']=packed['metadata']
    else:receipt['ground_history']=ground_history(a,root)
    print('ground49 complete',flush=True)
    for raw in raws:
        path=f'data/inspection/shared_{raw}.js'
        if not(resume and (root/path).exists()):js(root,path,f'inspection/shared/{raw}',pack(measured_frame(a,raw)))
    for arm in arms:
        if arm=='gn11_v1':
            # V1 has no road, so its native static depth is already ordinary depth; UI reads existing maps.
            continue
        if resume and all((root/f'data/inspection/{arm}_{raw}.js').exists() for raw in raws):continue
        config=json.loads((CAMPAIGN/f'arms/{arm}/run30k/effective_config.json').read_text())
        if Path(config['v3_bundle']).resolve()!=a.bundle.resolve():raise ValueError('experiment uses a different initialization bundle; export a separate road history')
        b=sampler.get_model(arm)['branches'][0]
        for raw in raws:
            path=f'data/inspection/{arm}_{raw}.js'
            if resume and (root/path).exists():continue
            t=time.time();p=prepare_depth(a,b,raw,root);alpha,ds=native_raster(root,p)
            # Independent Python compositor checks dense CPU path; combined/road rays also compared to saved native buffers.
            errs=[];native=[]
            with np.load(CAMPAIGN/f'final49/{arm}_r2/0008{raw:03d}/render_float.npz') as z:
                road_alpha=z['road_alpha'][0];road_depth=z['road_depth_sum'][0]/np.maximum(road_alpha,1e-8)/a.scale
                for x,y in [(90,255),(255,240),(268,225),(267,297),(365,280),(240,100),(30,30)]:
                    ids,al=candidates(p,x,y);mask=~p['road'][ids];c=composite(al[mask],np.zeros((mask.sum(),3),np.float32),p['depth'][ids][mask]);errs.extend([abs(c['alpha']-alpha[y,x]),abs(c['depth_sum']-ds[y,x])])
                    c=composite(al,np.zeros((len(ids),3),np.float32),p['depth'][ids]);native.extend([abs(c['alpha']-z['static_alpha'][0,y,x]),abs(c['depth_sum']-z['static_depth_sum'][0,y,x])])
                assert max(errs)<.002 and max(native)<.002,(arm,raw,max(errs),max(native))
                arrays=dict(ordinary_depth_m=np.where(alpha>1e-8,ds/np.maximum(alpha,1e-8)/a.scale,0).astype(np.float32),ordinary_alpha=alpha,road_depth_m=np.where(road_alpha>1e-8,road_depth,0).astype(np.float32))
            js(root,path,f'inspection/{arm}/{raw}',pack(arrays));receipt['checks'].append(dict(arm=arm,raw=raw,cpu_compositor_max_error=max(errs),saved_native_combined_max_error=max(native),seconds=round(time.time()-t,2)))
            print(arm,raw,'seconds',round(time.time()-t,2),flush=True);del p;gc.collect()
            (root/'inspection_receipt.json').write_text(json.dumps(clean(receipt),indent=2))
        sampler.model=None;sampler.current=None;a.models.clear();gc.collect()
    receipt.update(status='INSPECTION_EXPORT_COMPLETE',elapsed_seconds=round(time.time()-started,1),pixel_convention='integer index labels; all 3D ray/depth backprojections use x+0.5,y+0.5; native rendering untouched')
    (root/'inspection_receipt.json').write_text(json.dumps(clean(receipt),indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--arms',nargs='+',default=list(ARMS));p.add_argument('--raws',nargs='+',type=int,default=list(range(89,138)));p.add_argument('--resume',action='store_true');q=p.parse_args();export(q.output,q.arms,q.raws,q.resume)
