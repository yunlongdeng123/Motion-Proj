"""完整片段道路初始化：CPU、原始传感器、独立输出；不载入训练模型。

同20cm查询网格比较49/199帧；不把CPU初始化覆盖称为删车渲染成功。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, json, sys, time, resource
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates, laplace
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/drivingeditor_repro')
OLD=ROOT/'campaign/nearby_actors_20261015_v1'
BUNDLE=ROOT/'campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'
DENSE=ROOT/'campaign/r15_gn11_stage1_v41_20261023_dense_r7'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT/'campaign/r15_gn11_stage1_v41_20261023/tools'))
from v41_dense_geometry import project, transform, measured_patch_evidence, ray_box_blocked, FixedSurface

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def log(**x):print(json.dumps(x),flush=True)
def loadmeta():
    m=read(BUNDLE);anchor=np.asarray(m['coordinate_anchor']);inv=np.linalg.inv(anchor)
    plane=np.asarray(read(m['original_readiness'])['SGCM_geometry']['ground_fit']['plane_anchor'])
    return m,anchor,inv,plane
def sample(a,uv):
    if a.ndim==3:return np.stack([sample(a[...,k],uv) for k in range(a.shape[-1])],-1)
    return map_coordinates(a.astype(float),[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0)

def extract(out):
    import waymo_numpy_cpu
    from fast_range_matrix import matrix
    waymo_numpy_cpu.matrix=matrix
    from waymo_numpy_cpu import protobuf,records,lidar,front_rgb,calibration_text
    from prepare_v41_dense import masks
    from google.protobuf.json_format import MessageToDict
    m,anchor,inv,plane=loadmeta();pb=protobuf()
    source_meta=read(OLD/'inputs/scene0008_089_137/preparation_manifest.json')
    out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True);(out/'rgb').mkdir(exist_ok=True)
    protocol=dict(task_id='WS-V82-FULLCLIP-ROAD-CPU-20261026',raw_range=[0,198],baseline_range=[89,137],
        query_grid='Original 26088 world cells, 0.2m; preserves original footprint including unsupported cells',
        ground_rule='Original anchor plane distance <=0.15m; anchor x3..90,y+-8; exclude ALL labelled 3D objects',
        support='>=3 of nearest8 10cm-deduplicated ground returns within0.8m; original inverse-distance height rule',
        visibility='nearest actual return; small measured triangles; all 3D boxes; front 2D annotated boxes plus original mask union where available',
        comparison='same CPU initializer and observation gates on49 vs199; NOT bitwise repeat of old R eligibility',
        no_gt_invention=True,GPU_used=False,optimizer_updates=0,quality_accepted=False,human_verdict=None,
        failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none',
        old_assets_modified=False,external_models_used=False,seed=0,
        limitation='extra frames are new build inputs, not heldout tests; rectangles may underestimate real visibility; no CUDA render')
    if not (out/'protocol.json').exists():write(out/'protocol.json',protocol)
    roots=[OLD/'inputs/scene0008_089_137',OLD/'inputs/scene0008_077_125',ROOT/'data/waymo_paper_scenes/scene0008_023_071']
    start=time.time();rows=[]
    for idx,payload,crc in records(source_meta['raw_path'],198):
        fp=out/'frames'/f'{idx:03d}.npz';rp=out/'rgb'/f'{idx:03d}.png'
        if fp.exists() and rp.exists() and (out/'frames'/f'{idx:03d}.json').exists():
            rows.append(read(out/'frames'/f'{idx:03d}.json'));continue
        fr=pb.Frame.FromString(payload)
        assert fr.context.name==source_meta['segment_context_name']
        fid=f'0008{idx:03d}';pose=np.asarray(fr.pose.transform).reshape(4,4)
        cal={line.split(':')[0]:np.fromstring(line.split(':')[1],sep=' ') for line in calibration_text(fr).splitlines()}
        vc=np.eye(4);vc[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
        w2l=vc@np.linalg.inv(pose)@anchor;K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=.25
        rgb,_=front_rgb(fr);rgb=np.asarray(Image.fromarray(rgb).resize((480,320),Image.Resampling.LANCZOS))
        Image.fromarray(rgb).save(rp)
        reused=None
        for r in roots:
            p=r/'velodyne'/f'{fid}.bin'
            if p.exists():cloud=np.fromfile(p,np.float32).reshape(-1,6);reused=str(p);break
        if reused is None:cloud,_=lidar(fr,pb)
        local=transform(cloud[:,:3],inv@pose)
        ground=(np.abs(local@plane[:3]+plane[3])<=.15)&(local[:,0]>=3)&(local[:,0]<=90)&(np.abs(local[:,1])<=8)
        boxes=[]
        for lab in fr.laser_labels:
            b=lab.box
            if min(b.length,b.width,b.height)<=0:continue
            cs,sn=np.cos(b.heading),np.sin(b.heading)
            axes=(inv@pose)[:3,:3]@np.array([[cs,-sn,0],[sn,cs,0],[0,0,1.]])
            box=dict(center=transform([[b.center_x,b.center_y,b.center_z]],inv@pose)[0],half=np.array([b.length,b.width,b.height])/2,axes=axes)
            boxes.append(box);ii=np.flatnonzero(ground)
            ground[ii]&=~(np.abs((local[ii]-box['center'])@axes)<=box['half']).all(1)
        guard=np.zeros((320,480),bool)
        nrect=0
        for group in fr.camera_labels:
            if group.name!=1:continue
            for lab in group.labels:
                b=lab.box
                x0=max(0,int(np.floor((b.center_x-b.length/2)/4)));x1=min(480,int(np.ceil((b.center_x+b.length/2)/4)))
                y0=max(0,int(np.floor((b.center_y-b.width/2)/4)));y1=min(320,int(np.ceil((b.center_y+b.width/2)/4)))
                if x1>x0 and y1>y0:guard[y0:y1,x0:x1]=True;nrect+=1
        guard=binary_dilation(guard,iterations=2)
        if fid in m['frames']:guard|=masks(m['frames'][fid])
        # 只存当前相机每像素最近回波，地面池另存，逐帧内存有界。
        uv,z=project(local,K,w2l);ok=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        ids=np.flatnonzero(ok);pix=np.floor(uv[ids]+.5).astype(int);order=np.argsort(z[ids],kind='stable')
        _,ii=np.unique(pix[order,1]*480+pix[order,0],return_index=True);ids=ids[order[ii]]
        np.savez(fp,ground_local=local[ground].astype(np.float32),cloud=local[ids].astype(np.float32),ground=ground[ids],K=K,w2l=w2l,guard=guard,
            box_center=np.array([b['center'] for b in boxes]),box_half=np.array([b['half'] for b in boxes]),box_axes=np.array([b['axes'] for b in boxes]))
        row=dict(raw=idx,timestamp_micros=int(fr.timestamp_micros),ground_points=int(ground.sum()),visible_returns=len(ids),reused_lidar=reused,
                 rgb_rectangles=nrect,guard_pixels=int(guard.sum()),sharpness=float(np.var(laplace(rgb.astype(float).mean(2)/255.))),record=crc)
        write(out/'frames'/f'{idx:03d}.json',row);rows.append(row)
        log(stage='extract',raw=idx,ground=row['ground_points'],reused=bool(reused),elapsed_s=round(time.time()-start,1))
    assert len(rows)==199
    write(out/'extraction.json',dict(frames=rows,elapsed_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         raw_path=source_meta['raw_path'],segment=source_meta['segment_context_name'],GPU_used=False))

class Surface:
    def __init__(self,points,plane):self.p=points;self.tree=cKDTree(points[:,:2]);self.plane=-plane[[0,1,3]]/plane[2]
    def height(self,xy):
        d,ids=self.tree.query(xy,k=8,distance_upper_bound=.8);valid=np.isfinite(d);pts=self.p[np.minimum(ids,len(self.p)-1)]
        residual=pts[...,2]-pts[...,:2]@self.plane[:2]-self.plane[2];w=np.where(valid,1/np.maximum(d,.05)**2,0)
        return xy@self.plane[:2]+self.plane[2]+(w*residual).sum(1)/np.maximum(w.sum(1),1e-12),valid.sum(1)>=3
    def normal(self,xy):
        dx=(self.height(xy+[.05,0])[0]-self.height(xy-[.05,0])[0])/.1
        dy=(self.height(xy+[0,.05])[0]-self.height(xy-[0,.05])[0])/.1
        n=np.c_[-dx,-dy,np.ones(len(xy))];return n/np.linalg.norm(n,axis=1)[:,None]

def pool(out,raws):
    chunks=[]
    for raw in raws:
        with np.load(out/'frames'/f'{raw:03d}.npz') as f:chunks.append(f['ground_local'])
    p=np.concatenate(chunks).astype(float)
    _,ii=np.unique(np.floor(p[:,:2]/.1).astype(int),axis=0,return_index=True)
    return p[ii],len(p)

def initialize(out,variant='all'):
    if (out/'initialization.json').exists():raise FileExistsError('初始化已完成，不覆盖')
    m,anchor,inv,plane=loadmeta();u=np.load(BUNDLE.parent/m['units']);keys=u['keys'];xy=(keys+.5)*.2
    extraction=read(out/'extraction.json');sharp=np.array([r['sharpness'] for r in extraction['frames']]);quality=np.minimum(sharp/max(np.median(sharp[89:138]),1e-9),1.)
    start=time.time();variants={};saved={}
    for name,raws in [('window49',list(range(89,138))),('full199',list(range(199)))]:
        if variant!='all' and name!=variant:continue
        p,total=pool(out,raws);surface=Surface(p,plane);height,support=surface.height(xy);normal=surface.normal(xy);xyz=np.c_[xy,height]
        np.savez(out/f'{name}_ground_pool.npz',local_xyz=p.astype(np.float32))
        count=np.zeros(len(keys),np.int16);sparse_count=np.zeros_like(count);first=np.full(len(keys),-1,np.int16);second=first.copy()
        cfirst=np.zeros((len(keys),3),np.float32);csum=cfirst.copy();wsum=np.zeros(len(keys));last=np.full(len(keys),-1,np.int16)
        all_valid=[];all_rgb=[];rows=[];first_tri=np.full((len(keys),3),-1,np.int32)
        for raw in raws:
            f=np.load(out/'frames'/f'{raw:03d}.npz');K=f['K'];w=f['w2l'];cloud=f['cloud'];ground=f['ground'];rgb=np.asarray(Image.open(out/'rgb'/f'{raw:03d}.png'))/255.
            uv,z=project(xyz,K,w);origin=np.linalg.inv(w)[:3,3];direction=origin-xyz;distance=np.linalg.norm(direction,axis=1)
            cosine=np.abs((direction/np.maximum(distance[:,None],1e-9)*normal).sum(1));area=K[0,0]*K[1,1]*.2**2*cosine/np.maximum(z*z,1e-8)
            boxes=[dict(center=c,half=h,axes=a) for c,h,a in zip(f['box_center'],f['box_half'],f['box_axes'])]
            inside=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
            candidate=support&inside&(sample(f['guard'],uv)==0)&(cosine>=.12)&(area>=.25)&(quality[raw]>=.2)
            ids=np.flatnonzero(candidate);ok=np.zeros(len(keys),bool);sp=ok.copy()
            if len(ids):
                evidence,dep,tri,conflict=measured_patch_evidence(cloud,ground,K,w,xyz[ids],uv[ids],normal[ids])
                clear=~ray_box_blocked(xyz[ids],origin,boxes)
                ok[ids]=evidence&clear
                cuv,cz=project(cloud,K,w);dist,nn=cKDTree(cuv).query(uv[ids]);sp[ids]=clear&(dist<=1.5)&ground[nn]&(np.abs(cz[nn]-z[ids])<.3)
                fresh=ok[ids]&(first[ids]<0);first_tri[ids[fresh]]=tri[fresh]
            color=sample(rgb,uv).astype(np.float32);fresh=ok&(first<0);first[fresh]=raw;cfirst[fresh]=color[fresh]
            # 不选最佳颜色帧；首个几何合格来源与至少隔2帧的首个来源形成可审计配对。
            pair=ok&(second<0)&(first>=0)&(raw-first>=2);second[pair]=raw
            wt=np.where(ok,cosine**2*quality[raw]/np.maximum(distance,1)**2,0);csum+=color*wt[:,None];wsum+=wt
            count+=ok;sparse_count+=sp;last[ok]=raw
            all_valid.append(ok);all_rgb.append(color)
            rows.append(dict(raw=raw,triangle_sources=int(ok.sum()),nearest_return_sources=int(sp.sum()),candidates=int(candidate.sum())))
            if raw%10==0 or raw==raws[-1]:log(stage='sources',variant=name,raw=raw,valid=int(ok.sum()),elapsed_s=round(time.time()-start,1))
        all_valid=np.asarray(all_valid);all_rgb=np.asarray(all_rgb);lookup={r:i for i,r in enumerate(raws)}
        pair_rgb=np.zeros_like(cfirst);pair_ids=np.flatnonzero(second>=0)
        for raw in np.unique(second[pair_ids]):
            ids=pair_ids[second[pair_ids]==raw];pair_rgb[ids]=all_rgb[lookup[int(raw)],ids]
        mae=np.abs(pair_rgb-cfirst).mean(1);eligible=(second>=0)&(mae<=.05)
        color=np.divide(csum,wsum[:,None],out=np.full_like(csum,.18),where=wsum[:,None]>0)
        model=np.asarray(m['world_to_model']);world=transform(xyz,anchor);scale=m['meter_to_model']
        sigma=np.tile([.13,.13,.015],(len(keys),1));normal_world=normal@anchor[:3,:3].T
        # 薄Gaussian的真实初始化参数；只输出CPU资产，无优化器/训练状态。
        zaxis=np.array([0.,0.,1.]);q=np.c_[1+normal[:,2],np.cross(np.tile(zaxis,(len(normal),1)),normal)];q/=np.linalg.norm(q,axis=1)[:,None]
        arrays=dict(keys=keys,local_xyz=xyz.astype(np.float32),world_xyz=world,model_xyz=transform(world,model),normal_local=normal.astype(np.float32),
            normal_world=normal_world.astype(np.float32),gaussian_sigma_m=sigma.astype(np.float32),rotation_local_wxyz=q.astype(np.float32),
            rgb=color,opacity=np.full(len(keys),.5,np.float32),surface_supported=support,source_count=count,sparse_source_count=sparse_count,
            first_source=first,second_source=second,last_source=last,first_pair_rgb_mae=mae,eligible=eligible,first_triangle_indices=first_tri,
            source_valid=all_valid,source_raw=np.array(raws,np.int16),source_rgb=all_rgb)
        np.savez_compressed(out/f'{name}_initialization.npz',**arrays)
        variants[name]=dict(raw_count=len(raws),ground_returns=total,deduplicated_ground_points=len(p),query_cells=len(keys),supported=int(support.sum()),
            two_triangle_sources=int((count>=2).sum()),two_sparse_sources=int((sparse_count>=2).sum()),color_confirmed=int(eligible.sum()),single_source=int((count==1).sum()),
            source_records=rows,initial_opacity=.5,gaussian_sigmas_m=[.13,.13,.015])
        saved[name]=arrays
    write(out/('initialization.json' if variant=='all' else f'{variant}_summary.json'),dict(variants=variants,elapsed_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        original_R=dict(created=14034,eligible=1919),same_grid_cells=len(keys),GPU_used=False,optimizer_updates=0,quality_accepted=False,human_verdict=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['extract','init'],required=True);p.add_argument('--variant',default='all',choices=['all','window49','full199']);a=p.parse_args()
    if a.stage=='extract':extract(a.output)
    else:initialize(a.output,a.variant)
