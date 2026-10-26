import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from fullclip_road_cpu import loadmeta,read,write,ROOT,OLD,BUNDLE,DENSE,transform,project,sample,measured_patch_evidence,ray_box_blocked,FixedSurface

def subset_rays(surface,K,w2c,pixels,plane):
    c2w=np.linalg.inv(w2c);ray=np.c_[pixels+.5,np.ones(len(pixels))]@np.linalg.inv(K).T@c2w[:3,:3].T@surface.inverse[:3,:3].T
    origin=transform(c2w[None,:3,3],surface.inverse)[0];p=-plane[[0,1,3]]/plane[2]
    den=ray[:,2]-ray[:,:2]@p[:2];num=origin[:2]@p[:2]+p[2]-origin[2]
    t=np.divide(num,den,out=np.full(len(pixels),1e4),where=np.abs(den)>1e-7)
    for _ in range(8):
        xyz=origin+t[:,None]*ray;h,s=surface.height(xyz[:,:2]);dt=np.divide(h-xyz[:,2],den,out=np.zeros_like(t),where=(np.abs(den)>1e-7)&s)
        t+=np.clip(dt,-10,10)
    xyz=origin+t[:,None]*ray;h,s=surface.height(xyz[:,:2]);ids=surface.ids(np.floor(xyz[:,:2]/.2).astype(int))
    return xyz,t,s&(np.abs(h-xyz[:,2])<.03)&(t>.1)&(t<120)&(ids>=0),ids

def run(out):
    m,anchor,inv,plane=loadmeta();(out/'review').mkdir(exist_ok=True)
    old=np.load(BUNDLE.parent/m['units']);arrays={n:dict(np.load(out/f'{n}_initialization.npz')) for n in ['window49','full199']}
    surfaces={}
    for n,a in arrays.items():
        surfaces[n]=FixedSurface(m,dict(keys=a['keys'],world=a['world_xyz'],surface_supported=a['surface_supported'],provenance=a['eligible'].astype(np.uint8),base_weight=a['source_valid'],normal_world=a['normal_world']))
    # 固定raw102真实车辆世界足迹；中心为长/宽各中间50%，不随结果改变。
    ann=read(OLD/'inputs/scene0008_089_137/annotations/0008102.json');pose=np.loadtxt(OLD/'inputs/scene0008_089_137/pose/0008102.txt')
    tracks=dict(zip([5,26,18,6,37],read(Path(m['original_engineering'])/'matched5_input_receipt.json')['track_ids']))
    xy=arrays['full199']['local_xyz'][:,:2];foot={};regions={}
    for actor in [37,6]:
        label=next(l for l in ann['laser_labels'] if l['id']==tracks[actor]);b=label['box'];c,s=np.cos(b['heading']),np.sin(b['heading'])
        axes=(inv@pose)[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        center=transform([[b['center_x'],b['center_y'],b['center_z']]],inv@pose)[0];half=np.array([b['length'],b['width']])/2
        axes2=axes[:2,:2];axes2/=np.linalg.norm(axes2,axis=0);loc=(xy-center[:2])@axes2
        whole=(np.abs(loc)<=half).all(1);core=(np.abs(loc)<=half*.5).all(1);regions[actor]=(whole,core)
        foot[str(actor)]={}
        for region,mask in [('whole',whole),('core',core)]:
            foot[str(actor)][region]=dict(cells=int(mask.sum()),area_m2=float(mask.sum()*.04),original_created=int((mask&old['surface_supported']).sum()),original_R=int((mask&(old['provenance']==1)).sum()))
            for name,a in arrays.items():
                foot[str(actor)][region][name]=dict(supported=int((mask&a['surface_supported']).sum()),two_sources=int((mask&(a['source_count']>=2)).sum()),color_confirmed=int((mask&a['eligible']).sum()))
    rows=[];critical=[]
    for raw in range(89,138):
        fid=f'0008{raw:03d}';fm=m['frames'][fid];K=np.asarray(fm['intrinsic']);w=np.asarray(fm['world_to_camera'])
        with np.load(DENSE/'frames'/f'{fid}.npz') as z:labels=z['actor_labels'].copy();legacy=z['deletion_valid'].copy()
        yy,xx=np.nonzero(labels);pixels=np.c_[xx,yy];lab=labels[yy,xx]
        coremap=np.zeros_like(labels,bool)
        for k in range(1,6):coremap|=binary_erosion(labels==k,iterations=5)
        row=dict(raw=raw,actors={})
        maps={};details={}
        for name,a in arrays.items():
            xyz,depth,sup,uid=subset_rays(surfaces[name],K,w,pixels,plane);safe=np.maximum(uid,0);valid=sup&a['eligible'][safe]
            state=np.zeros((320,480),np.uint8);state[yy,xx]=1;state[yy[sup],xx[sup]]=2;state[yy[valid],xx[valid]]=3;maps[name]=state
            details[name]=dict(xyz=xyz,depth=depth,supported=sup,uid=uid,valid=valid)
        for k,actor in enumerate(m['actors'],1):
            mask=lab==k;core=mask&coremap[yy,xx];row['actors'][str(actor)]={}
            for region,sel in [('whole',mask),('core',core)]:
                result=dict(pixels=int(sel.sum()),original_crossview_labels=int((legacy[yy,xx]&sel).sum()))
                for name,d in details.items():result[name]=dict(supported=int((sel&d['supported']).sum()),eligible_cell_projection=int((sel&d['valid']).sum()))
                row['actors'][str(actor)][region]=result
        rgb=np.asarray(Image.open(out/'rgb'/f'{raw:03d}.png'));panel=Image.new('RGB',(1440,350),(15,25,40));draw=ImageDraw.Draw(panel)
        for j,name in enumerate(['target','window49','full199']):
            overlay=rgb.copy().astype(float)
            if name!='target':
                st=maps[name]
                for code,col in [(1,[240,70,90]),(2,[255,180,30]),(3,[30,240,150])]:
                    mask=st==code;overlay[mask]=overlay[mask]*.45+np.array(col)*.55
            panel.paste(Image.fromarray(overlay.astype(np.uint8)),(j*480,30));draw.text((j*480+10,8),f'raw{raw} / {name}',fill='white')
        panel.save(out/'review'/f'coverage_{raw}.jpg',quality=92)
        if raw==102:
            for pixel in [[255,240],[256,219]]:
                index=np.flatnonzero((pixels==pixel).all(1))[0];record=dict(pixel=pixel,variants={})
                for name,d in details.items():
                    uid=int(d['uid'][index]);a=arrays[name];q=d['xyz'][index]
                    record['variants'][name]=dict(local_xyz=q.tolist(),depth_m=float(d['depth'][index]),four_corner_supported=bool(d['supported'][index]),unit_id=uid,
                        unit_created=bool(a['surface_supported'][uid]),unit_eligible=bool(a['eligible'][uid]),cell_sources=a['source_raw'][a['source_valid'][:,uid]].tolist(),
                        two_first_sources=[int(a['first_source'][uid]),int(a['second_source'][uid])],first_pair_mae=float(a['first_pair_rgb_mae'][uid]))
                critical.append(record)
        rows.append(row)
    # 精确选中像素的新曲面点，逐帧核验，不把邻格有来源当作该像素有来源。
    xyz=np.array([c['variants']['full199']['local_xyz'] for c in critical]);normal=np.array([arrays['full199']['normal_local'][c['variants']['full199']['unit_id']] for c in critical])
    sources=[[],[]];source_points={}
    for raw in range(199):
        f=np.load(out/'frames'/f'{raw:03d}.npz');uv,z=project(xyz,f['K'],f['w2l']);rgb=np.asarray(Image.open(out/'rgb'/f'{raw:03d}.png'))/255.
        ok,depth,tri,conflict=measured_patch_evidence(f['cloud'],f['ground'],f['K'],f['w2l'],xyz,uv,normal)
        boxes=[dict(center=c,half=h,axes=a) for c,h,a in zip(f['box_center'],f['box_half'],f['box_axes'])]
        clear=~ray_box_blocked(xyz,np.linalg.inv(f['w2l'])[:3,3],boxes)
        inside=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        valid=ok&clear&inside&(sample(f['guard'],uv)==0)
        colors=sample(rgb,uv)
        for i in range(2):
            if valid[i]:sources[i].append(dict(raw=raw,pixel=uv[i].tolist(),rgb=colors[i].tolist(),measured_depth_m=float(depth[i]),lidar_triangle=tri[i].tolist()))
        source_points[raw]=uv
    for i,c in enumerate(critical):c['exact_pixel_geometry_witnesses']=sources[i]
    # 保存原始露出帧的图示：紫色圆圈标的是同一世界道路点，不是屏幕同坐标。
    witness_raws=sorted(set([102]+[r['raw'] for s in sources for r in s[::max(1,len(s)//3)]][:12]))
    board=Image.new('RGB',(480*3,350*((len(witness_raws)+2)//3)),(15,25,40));draw=ImageDraw.Draw(board)
    for j,raw in enumerate(witness_raws):
        im=Image.open(out/'rgb'/f'{raw:03d}.png').copy();d=ImageDraw.Draw(im)
        for i,(x,y) in enumerate(source_points[raw]):
            if 0<=x<480 and 0<=y<320:d.ellipse((x-5,y-5,x+5,y+5),outline=['magenta','cyan'][i],width=2);d.text((x+6,y),str(i+1),fill='white')
        ox=(j%3)*480;oy=(j//3)*350;board.paste(im,(ox,oy+30));draw.text((ox+8,oy+8),f'raw {raw} / same WORLD road targets',fill='white')
    board.save(out/'review/witnesses.jpg',quality=93)
    # 世界俯视图固定范围/固定分母，显示核心并非外围。
    region=(xy[:,0]>=8)&(xy[:,0]<=40)&(np.abs(xy[:,1])<=4)
    board=Image.new('RGB',(1000,660),(15,25,40));d=ImageDraw.Draw(board)
    for j,name in enumerate(['window49','full199']):
        a=arrays[name];d.text((j*500+12,10),name+' / 20 cm cells',fill='white')
        for idx in np.flatnonzero(region):
            x=j*500+250+xy[idx,1]*45;y=630-(xy[idx,0]-8)*18
            col=(70,70,80)
            if a['surface_supported'][idx]:col=(220,155,25)
            if a['eligible'][idx]:col=(30,225,145)
            d.rectangle((x-2,y-2,x+2,y+2),fill=col)
        for actor,(whole,core) in regions.items():
            for idx in np.flatnonzero(core):
                x=j*500+250+xy[idx,1]*45;y=630-(xy[idx,0]-8)*18;d.rectangle((x-3,y-3,x+3,y+3),outline=(230,100,250))
    board.save(out/'review/world_coverage.png')
    totals={}
    for actor in map(str,m['actors']):
        totals[actor]={}
        for region in ['whole','core']:
            r=dict(pixels=sum(row['actors'][actor][region]['pixels'] for row in rows),original_crossview_labels=sum(row['actors'][actor][region]['original_crossview_labels'] for row in rows))
            for name in arrays:r[name]={key:sum(row['actors'][actor][region][name][key] for row in rows) for key in ['supported','eligible_cell_projection']}
            totals[actor][region]=r
    merged=read(out/'window49_summary.json');merged['variants'].update(read(out/'full199_summary.json')['variants']);write(out/'initialization.json',merged)
    report=dict(fixed_raw102_world_footprints=foot,per_frame=rows,pixel_frame_totals=totals,critical=critical,
        eligibility_projection_is_not_dense_ground_truth=True,rendering_validated=False,quality_accepted=False,human_verdict=None)
    write(out/'coverage.json',report)
    print(json.dumps(dict(footprints=foot,critical=[dict(pixel=c['pixel'],variants=c['variants'],exact_sources=[s['raw'] for s in c['exact_pixel_geometry_witnesses']]) for c in critical],totals={k:totals[k] for k in ['37','6']}),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
