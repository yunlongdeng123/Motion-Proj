"""撤源组件评价：实测可见点、米制核心、简单平面/邻域颜色与固定视频后端。

不读取已有高斯或完整道路编码。拟合与颜色只读input_*，target_*仅用于计分。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
RUNS={
    'scene0017': ROOT.parent/'hybrid30k_20261012/runs/scene0017_000_048_hybrid_30000_seed0_GN11',
    'scene0029': ROOT.parent/'followup_scene_prep_20261013/runs/scene0029_069_117_hybrid_30000_seed0_GN11'}


def read(path):return json.loads(Path(path).read_text())
def transform(points,T):return points@T[:3,:3].T+T[:3,3]
def project(points,T,K):
    cam=transform(points,T);uv=(cam@K.T)[:,:2]/np.maximum(cam[:,2:],1e-9)-.5
    ij=np.floor(uv+.5).astype(np.int64)
    ok=(cam[:,2]>.01)&(ij[:,0]>=0)&(ij[:,0]<480)&(ij[:,1]>=0)&(ij[:,1]<320)
    return cam,ij,ok


def depth_map(cam,ij,valid):
    image=np.full(320*480,np.inf)
    np.minimum.at(image,ij[valid,1]*480+ij[valid,0],cam[valid,2])
    return image.reshape(320,480)


def fit_input_plane(folder,audit):
    center=np.array(audit['hole_center_anchor_xy']);half=np.array(audit['hole_half_extent_m'])
    all_points=[]
    for path in sorted((folder/'input_points').glob('*.npz')):
        a=np.load(path)['local_xyz'];near=(np.abs(a[:,:2]-center)<=half+3).all(1)
        all_points.append(a[near])
    p=np.concatenate(all_points).astype(np.float64)
    assert len(p)>100
    # 仅由保留下来的原始回波选择低处带，不读teacher plane/target points。
    level=np.percentile(p[:,2],20);p=p[np.abs(p[:,2]-level)<.7]
    _,ids=np.unique(np.floor(p/.1).astype(np.int64),axis=0,return_index=True);p=p[ids]
    random=np.random.default_rng(0);sample=p[random.choice(len(p),min(20000,len(p)),replace=False)]
    best=None;score=0
    for _ in range(256):
        q=sample[random.choice(len(sample),3,replace=False)]
        n=np.cross(q[1]-q[0],q[2]-q[0]);length=np.linalg.norm(n)
        if length<1e-8:continue
        n/=length
        if abs(n[2])<.9:continue
        plane=np.r_[n,-n@q[0]];current=int((np.abs(sample@plane[:3]+plane[3])<.15).sum())
        if current>score:score=current;best=plane
    assert best is not None and score>=.3*len(sample),'输入地面支撑不足，不能伪造平面基线'
    for _ in range(3):
        inlier=p[np.abs(p@best[:3]+best[3])<.15];middle=inlier.mean(0)
        _,_,vh=np.linalg.svd(inlier-middle,full_matrices=False);n=vh[-1]
        if n[2]<0:n=-n
        best=np.r_[n,-n@middle]
    return best,dict(input_ring_points=len(p),inlier_fraction=score/len(sample),teacher_plane_used=False,target_points_used=False,seed=0)


def main():
    for name,run in RUNS.items():
        folder=ROOT/'evaluation_withdrawal_v2'/name;audit=read(folder/'audit.json')
        dest=folder/'measurement_evaluation';dest.mkdir(exist_ok=True)
        assert not (dest/'summary.json').exists(),'不覆盖已有评价'
        source=Path(read(run/'effective_config.json')['source_path'])
        cameras=read(run/'cameras.json');byraw={int(c['img_name'][-3:]):c for c in cameras}
        anchor=np.array(audit['anchor_local_to_world']);inv_anchor=np.linalg.inv(anchor)
        model=np.array(audit['world_to_model']);scale=np.linalg.norm(model[:3,0])
        center=np.array(audit['hole_center_anchor_xy']);half=np.array(audit['hole_half_extent_m'])
        plane,fit=fit_input_plane(folder,audit)
        input_color_points=[];input_colors=[];cache=[]
        # 颜色池只来自已撤源输入，目标GT与已有高斯完全不参与。
        for row in audit['rows']:
            raw=row['raw'];T=np.array(row['world_to_camera'])@model@anchor;K=np.array(row['K'])
            points=np.load(folder/'input_points'/f'{raw:03d}.npz')['local_xyz'].astype(float)
            ground=(np.abs(points@plane[:3]+plane[3])<.15)&(np.abs(points[:,:2]-center)<=half+3).all(1)
            cam,ij,ok=project(points,T,K);nearest=depth_map(cam,ij,ok)
            mask=np.array(Image.open(folder/'input_mask'/f'{raw:03d}.png'))>0
            rgb=np.array(Image.open(folder/'input_rgb'/f'{raw:03d}.png'))/255.
            ids=np.flatnonzero(ground&ok);uv=ij[ids]
            keep=(~mask[uv[:,1],uv[:,0]])&(np.abs(cam[ids,2]-nearest[uv[:,1],uv[:,0]])<.1*scale)
            ids=ids[keep];uv=ij[ids]
            input_color_points.append(points[ids]);input_colors.append(rgb[uv[:,1],uv[:,0]])
            cache.append((row,T,K,mask,rgb))
        pool=np.concatenate(input_color_points);colors=np.concatenate(input_colors)
        assert len(pool)>20
        keys=np.floor(pool[:,:2]/.1).astype(np.int64);_,inverse=np.unique(keys,axis=0,return_inverse=True)
        count=np.bincount(inverse);locations=np.zeros((len(count),2));color=np.zeros((len(count),3))
        np.add.at(locations,inverse,pool[:,:2]);np.add.at(color,inverse,colors)
        locations/=count[:,None];color/=count[:,None];tree=cKDTree(locations)
        totals={method:{layer:dict(pixels=0,abs_sum=0.,square_sum=0.) for layer in ['all','boundary','transition','core']} for method in ['zero_masked_input','input_plane_neighbor_color','fixed_propainter']}
        rows=[];geometry=[]
        for row,T,K,input_mask,input_rgb in cache:
            raw=row['raw'];c=byraw[raw];frame=c['img_name']
            # GT侧独立重载原始全部回波，核验每像素最近可见回波，不能从撤源后的点云推断无遮挡。
            cloud=np.fromfile(source/'velodyne'/(frame+'.bin'),np.float32).reshape(-1,6)
            pose=np.loadtxt(source/'pose'/(frame+'.txt'));original=transform(cloud[:,:3],inv_anchor@pose)
            ocam,oij,ook=project(original,T,K);nearest=depth_map(ocam,oij,ook)
            target=np.load(folder/'target_points'/f'{raw:03d}.npz')['measured_ground_xyz'].astype(float)
            within=(np.abs(target[:,:2]-center)<=half).all(1);target=target[within]
            tcam,tij,tok=project(target,T,K)
            body=np.array(Image.open(source/'dynamic_masks/dynamic_mask_0'/(frame+'.png')).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
            guard=cv2.dilate(body.astype('uint8'),np.ones((7,7),np.uint8))>0
            ids=np.flatnonzero(tok);uv=tij[ids]
            visible=(np.abs(tcam[ids,2]-nearest[uv[:,1],uv[:,0]])<=.3*scale)&~guard[uv[:,1],uv[:,0]]&input_mask[uv[:,1],uv[:,0]]
            ids=ids[visible]
            # 一个像素只计最前面的目标实测点，不能重复LiDAR点放大分母。
            ordered=ids[np.argsort(tcam[ids,2],kind='stable')];pixels=tij[ordered,1]*480+tij[ordered,0]
            _,first=np.unique(pixels,return_index=True);ids=ordered[first];uv=tij[ids]
            distance=np.maximum(0,np.min(half-np.abs(target[ids,:2]-center),axis=1))
            layers={'all':np.ones(len(ids),bool),'boundary':distance<.25,'transition':(distance>=.25)&(distance<.75),'core':distance>=.75}
            valid=np.zeros((320,480),bool);valid[uv[:,1],uv[:,0]]=True
            labels=np.zeros((320,480),np.uint8)
            labels[uv[:,1],uv[:,0]]=np.where(distance<.25,1,np.where(distance<.75,2,3))
            depth=np.full((320,480),np.nan,np.float32);depth[uv[:,1],uv[:,0]]=tcam[ids,2]/scale
            np.savez_compressed(dest/f'{raw:03d}_reference.npz',valid=valid,layer=labels,depth_m=depth,measured_xyz=target[ids],core_distance_m=distance)
            # 最简单3D平面+输入邻域颜色外推，作为②的确定性强对照，不声称算法创新。
            gy,gx=np.mgrid[:320,:480];rays=np.stack([(gx+.5-K[0,2])/K[0,0],(gy+.5-K[1,2])/K[1,1],np.ones_like(gx)],-1)
            c2a=np.linalg.inv(T);origin=c2a[:3,3];direction=rays@c2a[:3,:3].T
            denominator=direction@plane[:3];time=-(origin@plane[:3]+plane[3])/np.where(np.abs(denominator)>1e-8,denominator,1e-8)
            query=origin+time[...,None]*direction
            fill=input_mask&(time>0)&(np.abs(query[...,:2]-center)<=half+.8).all(-1)
            result=input_rgb.copy();dist,nn=tree.query(query[fill,:2],k=min(8,len(locations)))
            weight=1/np.maximum(dist,.05)**2;weight/=weight.sum(1,keepdims=True)
            result[fill]=(color[nn]*weight[...,None]).sum(1)
            result=np.clip(np.round(result*255),0,255)/255.
            (dest/'deterministic_rgb').mkdir(exist_ok=True)
            Image.fromarray(np.uint8(np.clip(np.round(result*255),0,255))).save(dest/'deterministic_rgb'/f'{raw:03d}.png')
            original_rgb=np.array(Image.open(folder/'target_rgb'/f'{raw:03d}.png'))/255.
            generated_file=folder/'propainter/model_png'/f'{raw:03d}.png';assert generated_file.exists()
            generated=np.array(Image.open(generated_file))/255.
            current={}
            for method,image in [('zero_masked_input',input_rgb),('input_plane_neighbor_color',result),('fixed_propainter',generated)]:
                error=image[uv[:,1],uv[:,0]]-original_rgb[uv[:,1],uv[:,0]]
                current[method]={}
                for layer,select in layers.items():
                    if select.any():
                        abs_sum=float(np.abs(error[select]).sum());square_sum=float((error[select]**2).sum());pixels=int(select.sum())
                        t=totals[method][layer];t['pixels']+=pixels;t['abs_sum']+=abs_sum;t['square_sum']+=square_sum
                        current[method][layer]=dict(pixels=pixels,MAE=abs_sum/(3*pixels),PSNR=-10*np.log10(max(square_sum/(3*pixels),1e-12)))
            measured_difference=np.abs(target[ids]@plane[:3]+plane[3]);geometry.extend(measured_difference.tolist())
            rows.append(dict(raw=raw,sparse_reference_pixels=len(ids),core_pixels=int(layers['core'].sum()),masked_pixels=int(input_mask.sum()),metrics=current))
            if raw==audit['rows'][24]['raw']:
                sheet=Image.new('RGB',(480*4,350),'#152638');draw=ImageDraw.Draw(sheet)
                for i,(title,image) in enumerate([('observed target - scoring only',original_rgb),('all-source masked input',input_rgb),('input-plane + nearby RGB',result),('fixed ProPainter',generated)]):
                    draw.text((480*i+8,7),title,fill='white');sheet.paste(Image.fromarray(np.uint8(np.clip(image*255,0,255))),(480*i,27))
                sheet.save(dest/'comparison.jpg',quality=95)
        for method,layers in totals.items():
            for layer,value in layers.items():
                n=value['pixels'];value['MAE']=value['abs_sum']/(3*n) if n else None;value['PSNR']=-10*np.log10(max(value['square_sum']/(3*n),1e-12)) if n else None
        report=dict(scene=name,status='component_reference_and_baselines_complete',frames=49,
                    methods=totals,rows=rows,input_plane=plane.tolist(),input_plane_fit=fit,input_color_voxels=len(locations),
                    withheld_measured_point_to_input_plane_MAE_m=float(np.mean(geometry)),
                    scope='strict raw-input component; sparse measured near-ground targets, not full-scene B retraining or dense semantic-road GT',
                    target_visibility='same-frame original nearest LiDAR within0.3m; known actor mask dilated3px; nearest target per pixel',
                    layer_rule_m={'boundary':[0,.25],'transition':[.25,.75],'core':[.75,1.2]},
                    input_only_deterministic_fit=True,target_RGB_or_points_used_in_fit=False,
                    previous_Gaussians_or_latents_used=False,generated_output_is_3d_state=False,
                    limitations=['same sensor-frame RGB target, PNG benchmark',
                                 'sparse LiDAR/reference sampling; unseen pixels remain unknown',
                                 'coarse teacher ground selection, no independent semantic-road GT',
                                 'deterministic output uses fit plane; not a learned 3D completion result',
                                 'ProPainter yields 2D frames only; no generated geometry claim'],
                    new_training_updates=0,quality_accepted=False,human_verdict=None)
        (dest/'summary.json').write_text(json.dumps(report,indent=2));print(name,json.dumps(totals),flush=True)


if __name__=='__main__':main()
