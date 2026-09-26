"""固定两种读出和24对象；不调阈值、不补全。渲染采用固定3x3点splat。"""
import argparse, json, pathlib, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from geometry import box_mask, edit_actor, fit_sim3, project_bbox, resized_intrinsics, target_pose, transform, unproject


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pose_error(rotation, predicted, known):
    errors=[]
    for p,g in zip(predicted,known):
        relative=(rotation @ p[:3,:3]).T @ g[:3,:3]
        errors.append(float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
    return errors


def read_geometry(scene, prediction, out):
    dep=prediction['depth'][0,...,0].astype(np.float64);h,w=dep.shape[1:]
    origin=np.asarray(scene['origin_world'])
    cams=np.array([v['c2w'] for v in scene['views']]);cams[:,:3,3]-=origin
    kg=np.array([resized_intrinsics(v['intrinsics'],v['original_wh'],[h,w]) for v in scene['views']])
    ec=prediction['extrinsics'][0];pcams=np.tile(np.eye(4),(6,1,1));pcams[:,:3,:4]=ec;pcams=np.linalg.inv(pcams)
    scale,rotation,translation,residual=fit_sim3(pcams[:,:3,3],cams[:,:3,3])
    raw=np.array([unproject(d,k,c) for d,k,c in zip(dep,prediction['intrinsics'][0],pcams)])
    native=raw @ rotation.T * scale + translation
    root=pathlib.Path(scene['root']);lidar=np.fromfile(root/'lidar'/f"{scene['frame']:03d}.bin",np.float32).reshape(-1,4)[:,:3]
    lidar=transform(lidar,np.loadtxt(root/'lidar_pose'/f"{scene['frame']:03d}.txt"))-origin
    boxes=[];background=np.ones(len(lidar),bool)
    for box in scene['all_boxes']:
        pose=np.asarray(box['pose']);pose[:3,3]-=origin
        background &= ~box_mask(lidar,pose,box['size_lwh'])
        boxes.append((box,pose))
    ratios=[];anchor_counts=[]
    for d,c,k in zip(dep,cams,kg):
        cp=transform(lidar,np.linalg.inv(c));uv=cp @ k.T
        uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:3],1e-12)).astype(np.int64)
        ok=(cp[:,2]>1) & (cp[:,2]<80) & (uv[:,0]>=0) & (uv[:,0]<w) & (uv[:,1]>=0) & (uv[:,1]<h)
        # 先对全部LiDAR取像素最近回波，再判断其是否背景，避免越过目标选背景。
        ids=np.flatnonzero(ok);sort=ids[np.argsort(cp[ids,2],kind='stable')]
        _,first=np.unique(uv[sort,1]*w+uv[sort,0],return_index=True);ids=sort[first]
        ids=ids[background[ids]]
        pred=d[uv[ids,1],uv[ids,0]];good=np.isfinite(pred)&(pred>1e-6)
        ratio=cp[ids,2][good]/pred[good];ratios.extend(ratio.tolist());anchor_counts.append(int(good.sum()))
    assert len(ratios)>100,'背景米制锚点不足'
    calibrated_scale=float(np.median(ratios))
    calibrated=np.array([unproject(d*calibrated_scale,k,c) for d,k,c in zip(dep,kg,cams)])
    report={'native_sim3_scale':scale,'native_camera_center_error_m':residual.tolist(),
            'native_camera_center_rmse_m':float(np.sqrt(np.mean(residual**2))),
            'native_rotation_error_deg':pose_error(rotation,pcams,cams),
            'native_predicted_intrinsics':prediction['intrinsics'][0].tolist(),
            'calibrated_global_depth_scale':calibrated_scale,'calibration_background_anchor_count':len(ratios),
            'anchors_per_camera':anchor_counts,'background_scale_ratio_q10_q50_q90':np.quantile(ratios,[.1,.5,.9]).tolist(),
            'input_roles':{'native':'RGB Ω + GT相机中心Sim3','calibrated_control':'相同RGB Ω + GT相机内外参 + 同帧背景LiDAR单尺度','actor_lidar':'仅评价；没有参与背景尺度拟合'},
            'metric_anchor_selection':'全部GT框以外，深度1..80m，像素最近回波；无confidence筛选'}
    dump(out/'alignment.json',report)
    colors=np.rint(np.moveaxis(prediction['images'],1,-1)*255).clip(0,255).astype(np.uint8)
    sources=np.broadcast_to(np.arange(6)[:,None,None],dep.shape)
    variants={}
    for name,points,metric_depth in [('native',native,dep*scale),('calibrated_control',calibrated,dep*calibrated_scale)]:
        valid=np.isfinite(points).all(-1)&(metric_depth>.5)&(metric_depth<80)
        variants[name]=(points[valid],colors[valid],sources[valid])
    return variants, cams, kg, lidar, colors, report


def render(points,colors,c2w,k,hw,radius=1):
    """确定性的点z-buffer；空洞保持深灰，不插值或补全。"""
    h,w=hw;device='cuda'
    xyz=torch.as_tensor(np.ascontiguousarray(points),dtype=torch.float32,device=device)
    mat=torch.as_tensor(np.linalg.inv(c2w),dtype=torch.float32,device=device)
    cp=xyz @ mat[:3,:3].T+mat[:3,3]
    proj=cp @ torch.as_tensor(k,dtype=torch.float32,device=device).T
    uv=torch.round(proj[:,:2]/proj[:,2:3].clamp(min=1e-6)).long()
    visible=(cp[:,2]>.1)&(uv[:,0]>=-radius)&(uv[:,0]<w+radius)&(uv[:,1]>=-radius)&(uv[:,1]<h+radius)
    ids=torch.nonzero(visible).flatten();uv=uv[visible];z=cp[visible,2]
    index=[];depth=[];pointid=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            u=uv[:,0]+dx;v=uv[:,1]+dy;valid=(u>=0)&(u<w)&(v>=0)&(v<h)
            index.append(v[valid]*w+u[valid]);depth.append(z[valid]);pointid.append(ids[valid])
    index=torch.cat(index);depth=torch.cat(depth);pointid=torch.cat(pointid)
    buffer=torch.full((h*w,),float('inf'),device=device);buffer.scatter_reduce_(0,index,depth,reduce='amin',include_self=True)
    nearest=depth==buffer[index];chosen=torch.full((h*w,),len(points),dtype=torch.long,device=device)
    chosen.scatter_reduce_(0,index[nearest],pointid[nearest],reduce='amin',include_self=True)
    palette=torch.cat([torch.as_tensor(colors,device=device),torch.tensor([[22,27,35]],dtype=torch.uint8,device=device)])
    rgb=palette[chosen].reshape(h,w,3).cpu().numpy();mask=(chosen<len(points)).reshape(h,w).cpu().numpy()
    return rgb,mask


def labeled(images,labels,width=344):
    pieces=[]
    for array,label in zip(images,labels):
        im=Image.fromarray(array);height=round(im.height*width/im.width);im=im.resize((width,height))
        canvas=Image.new('RGB',(width,height+28),'#f0f3f7');canvas.paste(im,(0,28));ImageDraw.Draw(canvas).text((7,7),label,fill='#152438');pieces.append(canvas)
    out=Image.new('RGB',(width*len(pieces),max(p.height for p in pieces)),'white')
    for i,p in enumerate(pieces):out.paste(p,(i*width,0))
    return out


def camera_looking(eye,center):
    z=np.asarray(center)-np.asarray(eye);z=z/np.linalg.norm(z)
    up=np.array([0.,0.,1.])
    if abs(z@up)>.98:up=np.array([0.,1.,0.])
    x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    t=np.eye(4);t[:3,:3]=np.stack([x,y,z],1);t[:3,3]=eye
    return t


def lidar_recall(reference,points):
    if len(reference)==0:return None,None
    if len(points)==0:return 0.,None
    p=torch.as_tensor(points,dtype=torch.float32,device='cuda');dist=[]
    for start in range(0,len(reference),128):
        r=torch.as_tensor(reference[start:start+128],dtype=torch.float32,device='cuda')
        dist.extend(torch.cdist(r,p).min(1).values.cpu().tolist())
    d=np.array(dist)
    return float((d<=.2).mean()),float(np.median(d))


def command_list(config):
    commands=[]
    for axis in [0,1]:
        for delta in config['translation_m']:
            xyz=[0.,0.,0.];xyz[axis]=delta
            commands.append(('MOVE',xyz,0.))
    commands.extend(('MOVE',[0.,0.,0.],yaw) for yaw in config['yaw_degrees'])
    commands += [('DELETE',[0.,0.,0.],0.),('INSERT',config['insert_clone_delta_world_m'],0.)]
    return commands


def evaluate_scene(scene,root,result_root,reg):
    out=result_root/scene['name'];out.mkdir(parents=True,exist_ok=True)
    pred=np.load(root/scene['name']/'prediction.npz')
    assert not (out/'evaluation.json').exists(),'禁止覆盖评价'
    variants,cams,ks,lidar,original,alignment=read_geometry(scene,pred,out)
    h,w=original.shape[1:3];origin=np.array(scene['origin_world']);reports=[];full_renders={}
    for name,(points,colors,sources) in variants.items():
        variant_dir=out/name;variant_dir.mkdir(exist_ok=True)
        np.savez_compressed(variant_dir/'scene_points.npz',points_local_world=points.astype(np.float32),colors=colors,source_camera=sources,origin_world=origin)
        factual=[]
        for cam,k in zip(cams,ks):factual.append(render(points,colors,cam,k,(h,w))[0])
        full_renders[name]=factual
        for actor in scene['actors']:
            pose=np.array(actor['pose']);pose[:3,3]-=origin;size=np.array(actor['size_lwh']);selected=box_mask(points,pose,size)
            ap=points[selected];ac=colors[selected];reference=lidar[box_mask(lidar,pose,size)]
            local=transform(ap,np.linalg.inv(pose));recall,distance=lidar_recall(reference,ap)
            record={'scene':scene['name'],'variant':name,'actor_id':actor['actor_id'],'track_id':actor['track_id'],
                    'category':actor['category'],'distance_m':actor['distance_m'],'primary_camera':actor['primary_camera'],
                    'point_count':len(ap),'lidar_reference_points':len(reference),'visible_lidar_recall_0p2m':recall,'lidar_median_nearest_m':distance,
                    'gt_box_size_lwh':size.tolist(),'geometry_extent_lwh':np.ptp(local,axis=0).tolist() if len(local) else [0,0,0],
                    'source_camera_point_counts':np.bincount(sources[selected],minlength=6).tolist(),
                    'bottom_15cm_point_fraction':float((local[:,2]<-size[2]/2+.15).mean()) if len(local) else None,
                    'isolation_or_completeness_human_verdict':None,'commands':[]}
            ad=variant_dir/('actor_'+actor['actor_id']);ad.mkdir(exist_ok=True)
            np.savez_compressed(ad/'actor.npz',local_points=local.astype(np.float32),colors=ac,source_camera=sources[selected],pose_local_world=pose,size_lwh=size)
            for op,xyz,yaw in command_list(reg['edits']):
                target=target_pose(pose,xyz,yaw)
                result,col,ids=edit_actor(points,colors,selected,pose,target,op)
                # 背景来源索引直接指向原数组，检验完整输出而非仅检查指令矩阵。
                bg=~selected[ids]
                background_error=float(np.max(np.abs(result[bg]-points[ids[bg]]))) if bg.any() else 0.
                if op=='DELETE':
                    target_count=int(np.sum(selected[ids]));metric_error=None
                else:
                    transformed=result[selected] if op=='MOVE' else result[len(points):]
                    expected=transform(local,target)
                    metric_error=float(np.linalg.norm(transformed-expected,axis=1).max()) if len(ap) else None
                    target_count=len(transformed)
                total_expected=len(points)-len(ap) if op=='DELETE' else len(points)+len(ap) if op=='INSERT' else len(points)
                assert len(result)==total_expected and background_error==0
                if metric_error is not None:assert metric_error<1e-8
                record['commands'].append({'operation':op,'delta_world_m':xyz,'yaw_deg':yaw,'output_point_count':len(result),
                    'target_point_count':target_count,'max_point_adherence_error_m':metric_error,'max_background_change_m':background_error,
                    'valid_object_test':len(ap)>0,'donor_preserved':bool(np.array_equal(result[:len(points)],points)) if op=='INSERT' else None})
                del result,col,ids
            c=actor['primary_camera'];images=[original[c],factual[c]];labels=['Input RGB','Factual points']
            # 每对象固定同一MOVE(+3m x,+15 yaw)、DELETE、INSERT(+3m x)，不择优选方向。
            for op,xyz,yaw in [('MOVE',[3,0,0],15),('DELETE',[0,0,0],0),('INSERT',[3,0,0],0)]:
                target=target_pose(pose,xyz,yaw);edited,col,_=edit_actor(points,colors,selected,pose,target,op)
                rgb,mask=render(edited,col,cams[c],ks[c],(h,w));images.append(rgb);labels.append(op)
                Image.fromarray(rgb).save(ad/(op.lower()+'.jpg'),quality=88)
                del edited,col
            strip=labeled(images,labels)
            strip.save(ad/'edit_strip.jpg',quality=91)
            # GT box附近作固定放大图；同一裁剪用于原图和全部编辑。
            b=actor['gt_projected_boxes'][c];lo=np.maximum(np.array(b[:2])-35,0).astype(int);hi=np.minimum(np.array(b[2:])+35,[w,h]).astype(int)
            crops=[a[lo[1]:hi[1],lo[0]:hi[0]] for a in images]
            labeled(crops,labels,width=250).save(ad/'edit_crop.jpg',quality=92)
            canon=[];directions=[(1,0,.2),(-1,0,.2),(0,1,.2),(0,-1,.2),(0,0,1)]
            k=np.array([[220,0,127.5],[0,220,127.5],[0,0,1.]])
            for direction in directions:
                eye=np.array(direction,dtype=float);eye=eye/np.linalg.norm(eye)*max(size)*1.6
                view=camera_looking(eye,[0,0,0]);canon.append(render(local,ac,view,k,(256,256))[0])
            labeled(canon,['Front','Back','Left','Right','Top'],256).save(ad/'canonical.jpg',quality=92)
            dump(ad/'metrics.json',record);reports.append(record)
            print(json.dumps({'scene':scene['name'],'variant':name,'actor':actor['actor_id'],'points':len(ap),'lidar_n':len(reference),'recall':recall}),flush=True)
    comparisons=[]
    for camera in range(6):
        comparisons.append(labeled([original[camera],full_renders['native'][camera],full_renders['calibrated_control'][camera]],['Input camera '+str(camera),'Native Sim3','GT calibrated + bg scale']))
    montage=Image.new('RGB',(comparisons[0].width,sum(x.height for x in comparisons)))
    y=0
    for panel in comparisons:montage.paste(panel,(0,y));y+=panel.height
    montage.save(out/'scene_comparison.jpg',quality=91)
    dump(out/'evaluation.json',{'scene':scene['name'],'alignment':alignment,'actors':reports,'failure_ledger_delta':'pending_evidence_review'})
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True);args=p.parse_args()
    torch.set_num_threads(4);root=pathlib.Path(args.run_dir);reg=json.loads((root/'registration.json').read_text());all_records=[]
    result_root=pathlib.Path(args.output_dir);result_root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for scene in reg['scenes']:all_records.extend(evaluate_scene(scene,root,result_root,reg))
    summary={'actor_count':reg['actual_actor_count'],'readout_count':2,'actor_readouts':len(all_records),'evaluation_s':time.monotonic()-start,'by_variant':{}}
    for variant in ['native','calibrated_control']:
        rows=[r for r in all_records if r['variant']==variant];commands=[c for r in rows for c in r['commands']]
        recalls=[r['visible_lidar_recall_0p2m'] for r in rows if r['visible_lidar_recall_0p2m'] is not None]
        errors=[c['max_point_adherence_error_m'] for c in commands if c['max_point_adherence_error_m'] is not None]
        summary['by_variant'][variant]={'actors':len(rows),'empty_actors':sum(r['point_count']==0 for r in rows),'commands':len(commands),
            'valid_object_commands':sum(c['valid_object_test'] for c in commands),'max_adherence_error_m':max(errors,default=None),
            'max_background_change_m':max(c['max_background_change_m'] for c in commands),'actors_with_lidar':len(recalls),
            'macro_visible_lidar_recall_0p2m':float(np.mean(recalls)) if recalls else None,
            'median_actor_point_count':float(np.median([r['point_count'] for r in rows]))}
    dump(result_root/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
