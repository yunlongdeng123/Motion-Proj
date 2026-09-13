"""V81 CPU atlas：冻结模型前因素统计、参考几何与自然候选，绝不伪造预测。"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,collections,time
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)
from PIL import Image
from scipy.spatial import cKDTree
import pyarrow as pa
import pyarrow.parquet as pq
from motion_proj.worldsim_v81.geometry import transform,apply,project,remove_boxes,reference_filter,plane_fit,roi_mask,coverage

CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

def texture(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    dx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)/8;dy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)/8
    hist=np.histogram(g,bins=32,range=(0,1))[0]; p=hist[hist>0]/g.size
    orb=cv2.ORB_create(nfeatures=400,edgeThreshold=8,fastThreshold=8)
    kp,des=orb.detectAndCompute((g*255).astype('uint8'),None)
    return {'gradient_energy':float(np.mean(dx*dx+dy*dy)),'gray_entropy':float(-np.sum(p*np.log2(p))),
            'feature_density':len(kp)/g.size*10000,'dark_fraction':float((g<.04).mean()),'saturated_fraction':float((g>.96).mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reuse-run');a=p.parse_args();out=Path(a.run)
    idx=json.loads((out/'index.json').read_text());root=Path(idx['root']); rows=[];inputs=[];started=time.time()
    for d in ['reference_geometry','input_manifests','crops','interventions','figures']: (out/d).mkdir(exist_ok=True)
    completed=set()
    if a.reuse_run:
        import shutil
        base=Path(a.reuse_run)
        rows=[json.loads(l) for l in (base/'v81_roi_registry.jsonl').read_text().splitlines()]
        completed={r['window_id'] for r in rows}
        for folder in ['reference_geometry','input_manifests','crops']:
            for f in (base/folder).iterdir():
                dest=out/folder/f.name
                if not dest.exists():os.link(f,dest) if folder!='input_manifests' else shutil.copy2(f,dest)
        inputs=[{'window_id':w,'manifest':str(out/'input_manifests'/f'{w}.json'),'status':'WAIT_GPU'} for w in sorted(completed)]
    def sensor(sample,ch):
        r=idx['sample_data'][sample][ch];c=idx['calibrated'][r['calibrated_sensor_token']]
        T=transform(idx['poses'][r['ego_pose_token']])@transform(c)
        return r,T,np.array(c['camera_intrinsic'])
    def cloud(s):
        r,T,_=sensor(s,'LIDAR_TOP'); xyz=np.fromfile(root/r['filename'],np.float32).reshape(-1,5)[:,:3]
        xyz=apply(xyz.astype(float),T)
        return remove_boxes(xyz,idx['annotations'].get(s,[]))
    for wi,w in enumerate(idx['windows']):
        if w['window_id'] in completed:continue
        s=w['sample_token'];prompt=cloud(s);refpts=[];src=[]
        for ri,n in enumerate(w['reference_samples']):
            q=cloud(n);q=remove_boxes(q,idx['annotations'].get(s,[]));refpts.append(q);src.extend([ri]*len(q))
        if not refpts:continue
        ref=np.concatenate(refpts);src=np.asarray(src);refids=w['reference_samples']
        targetrows=[sensor(s,c) for c in CAMERAS]
        # 相机、原图、变换记录只存输入侧；heldout 文件不进入模型参数。
        manifest={'window_id':w['window_id'],'scene':w['scene'],'log':w['log'],'role':'DISCOVERY','target_sample':s,
                  'views':[{'camera':c,'sample_token':s,'image':str(root/r['filename']),'size':[r['width'],r['height']], 'K':K.tolist(),'world_from_camera':T.tolist(),'timestamp_us':r['timestamp']} for c,(r,T,K) in zip(CAMERAS,targetrows)],
                  'context_views':[],'lidar_input':idx['sample_data'][s]['LIDAR_TOP']['filename']}
        for ns in w['context_samples']:
            for c in CAMERAS:
                r,T,K=sensor(ns,c);manifest['context_views'].append({'camera':c,'sample_token':ns,'image':str(root/r['filename']),'K':K.tolist(),'world_from_camera':T.tolist(),'timestamp_us':r['timestamp']})
        (out/'input_manifests'/f"{w['window_id']}.json").write_text(json.dumps(manifest))
        inputs.append({'window_id':w['window_id'],'manifest':str(out/'input_manifests'/f"{w['window_id']}.json"),'status':'WAIT_GPU'})
        projections=[project(ref,T,K) for r,T,K in targetrows]
        for ci,(cam,(r,T,K)) in enumerate(zip(CAMERAS,targetrows)):
            image=np.array(Image.open(root/r['filename']).convert('RGB'));height,width=image.shape[:2]
            uv,z,xyz=projections[ci];valid=reference_filter(uv,z,src,width,height)
            pu,pz,px=project(prompt,T,K)
            # 当前扫描只作遮挡拒绝，不作为 HELDOUT depth。
            pv=roi_mask(pu,pz,[0,0,width,height]);ptree=cKDTree(pu[pv]) if pv.any() else None
            if ptree and len(valid):
                dist,nn=ptree.query(uv[valid]);reject=(dist<10)&(z[valid]>pz[pv][nn]+.5+.01*z[valid]);valid=valid[~reject]
            validset=np.zeros(len(ref),bool);validset[valid]=True
            # 5x3 等大非重叠 patch，避开 sky 顶部，路面独立分层。
            for yi,y0 in enumerate([180,360,540]):
                for xi,x0 in enumerate(range(0,1600,320)):
                    box=[x0,y0,min(x0+320,width),min(y0+180,height)]
                    if box[2]<=x0 or box[3]<=y0:continue
                    mask=roi_mask(uv,z,box)&validset;pts=xyz[mask];uvr=uv[mask];zr=z[mask]
                    rid=f"{w['window_id']}_{cam}_{yi}{xi}"
                    crop=image[box[1]:box[3],box[0]:box[2]];stats=texture(crop)
                    inprompt=roi_mask(pu,pz,box);pc=pu[inprompt]
                    plane=plane_fit(pts); nref=len(pts); support=coverage(uvr,box)
                    reliable=(nref>=30 and len(np.unique(src[mask]))>=2 and support>=.25 and plane is not None and plane['inlier_fraction']>=.85 and plane['rms']<=.12)
                    normal_world=T[:3,:3]@plane['normal'] if plane else None
                    surface='ground_plane' if normal_world is not None and abs(normal_world[2])>.9 else 'non_ground_plane_candidate'
                    row={'roi_id':rid,'window_id':w['window_id'],'scene':w['scene'],'log':w['log'],'role':'DISCOVERY','camera':cam,'box':box,
                         'natural':True,**stats,'reference_points':nref,'reference_scans':len(np.unique(src[mask])),'reference_coverage':support,
                         'range_m':float(np.median(np.linalg.norm(pts,axis=1))) if nref else None,'depth_m':float(np.median(zr)) if nref else None,
                         'plane_rms_m':plane['rms'] if plane else None,'plane_inlier_fraction':plane['inlier_fraction'] if plane else None,
                         'reference_status':'GEOMETRIC_SCREEN_PASS' if reliable else 'QUALITATIVE_ONLY', 'semantic':surface,'semantic_review':'UNREVIEWED',
                         'deskew':'scan_pose_compensated; per_point_timestamps_unavailable','lidar_prompt_points':int(inprompt.sum()),
                         'lidar_density_per_10k_px':float(inprompt.sum()/crop.shape[0]/crop.shape[1]*10000),'lidar_coverage':coverage(pc,box),
                         'lidar_dispersion_2d':float(np.trace(np.cov((pc-[x0,y0])/[320,180],rowvar=False))) if len(pc)>1 else None,
                         'lidar_dispersion_3d':float(np.trace(np.cov(px[inprompt],rowvar=False))) if len(pc)>1 else None,
                         'lidar_depth_span_m':float(np.ptp(pz[inprompt])) if len(pc) else None,
                         'prior_error':None,'failure_codes':[],'model_status':'NOT_RUN','human_verdict':None}
                    row['nearest_prompt_distance_px']=float(np.median(ptree.query(uvr)[0])) if ptree and len(uvr) else None
                    overlap=[];angles=[];bestj=None
                    for j,(other,(orr,OT,OK)) in enumerate(zip(CAMERAS,targetrows)):
                        if j==ci:continue
                        ou,oz,ox=projections[j];ovis=roi_mask(ou[mask],oz[mask],[0,0,width,height]);ov=float(ovis.mean()) if nref else None
                        overlap.append(ov)
                        if nref and ovis.any():
                            world=ref[mask][ovis];a1=world-T[:3,3];a2=world-OT[:3,3]
                            cos=np.sum(a1*a2,1)/(np.linalg.norm(a1,axis=1)*np.linalg.norm(a2,axis=1));angle=float(np.median(np.degrees(np.arccos(np.clip(cos,-1,1)))))
                        else:angle=None
                        angles.append(angle)
                        if ov is not None and (bestj is None or ov>bestj[0]):bestj=(ov,j)
                    row['overlap_frustum_upper_bound']=max([v for v in overlap if v is not None],default=None)
                    row['parallax_deg']=max([v for v in angles if v is not None],default=None)
                    row['overlap_status']='calibrated_frustum_plus_sparse_depth_rejection; residual_occlusion_unknown'
                    row['repeatable_match_density']=None
                    # 对几何对应目标做相同相机尺度 ORB 特征互检，明确零匹配与不可评估。
                    if reliable and bestj and bestj[0]>.02:
                        j=bestj[1];oimg=np.array(Image.open(root/targetrows[j][0]['filename']).convert('RGB'))
                        orb=cv2.ORB_create(nfeatures=1500,edgeThreshold=8,fastThreshold=8)
                        kp1,d1=orb.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
                        kp2,d2=orb.detectAndCompute(cv2.cvtColor(oimg,cv2.COLOR_RGB2GRAY),None)
                        matches=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(d1,d2) if d1 is not None and d2 is not None else []
                        n=0;tree=cKDTree(uvr)
                        for m in matches:
                            u=np.array(kp1[m.queryIdx].pt);dist,nn=tree.query(u)
                            if not(x0<=u[0]<box[2] and y0<=u[1]<box[3]) or dist>12 or m.distance>48:continue
                            ou,oz,_=projections[j]; target=ou[np.flatnonzero(mask)[nn]]
                            if np.linalg.norm(target-np.array(kp2[m.trainIdx].pt))<12:n+=1
                        row['repeatable_match_density']=n/(320*180)*10000
                    if reliable:
                        Image.fromarray(crop).save(out/'crops'/f'{rid}.jpg',quality=88)
                        np.savez_compressed(out/'reference_geometry'/f'{rid}.npz',uv=uvr.astype('float32'),xyz=pts.astype('float32'),depth_z=zr.astype('float32'),source_scan=src[mask],prompt_uv=pc.astype('float32'),prompt_xyz=px[inprompt].astype('float32'),box=box,K=K,world_from_camera=T,normal=plane['normal'],center=plane['center'],reference_samples=np.array(refids),input_sample=np.array(s))
                    rows.append(row)
        print(json.dumps({'window':wi+1,'total':len(idx['windows']),'rows':len(rows),'elapsed_s':round(time.time()-started)}),flush=True)
        (out/'roi_progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    eligible=[r for r in rows if r['reference_status']=='GEOMETRIC_SCREEN_PASS' and r['dark_fraction']<.2 and r['saturated_fraction']<.2]
    # 两个模型前测度同时处于尾部分位，混合区明确保留，不强制凑四格。
    thresholds={k:np.quantile([r[k] for r in eligible],[.3,.7]).tolist() for k in ['gradient_energy','gray_entropy']}
    for r in rows:
        low=all(r[k]<=thresholds[k][0] for k in thresholds);high=all(r[k]>=thresholds[k][1] for k in thresholds)
        texture_label='low' if low else 'high' if high else 'middle'
        ov=r['overlap_frustum_upper_bound'];overlap_label='low' if ov is not None and ov<=.1 else 'high' if ov is not None and ov>=.4 and (r['parallax_deg'] or 0)>=1 else 'middle'
        r.update(texture_label=texture_label,overlap_label=overlap_label,cohort=None)
        if r in eligible and texture_label!='middle' and overlap_label!='middle':r['cohort']='C'+('1' if low else '0')+('1' if overlap_label=='low' else '0')
    (out/'v81_roi_registry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'thresholds.json').write_text(json.dumps({'texture_quantiles':[.3,.7],'values':thresholds,'overlap_low_max':.1,'overlap_high_min':.4,'high_parallax_min_deg':1.,'frozen_before_model_inference':True,'scope':'DISCOVERY geometric-screen pool','note':'frustum overlap is upper bound; main matched-cohort claim requires visibility/semantic review'},indent=2))
    pq.write_table(pa.Table.from_pylist(rows),out/'texture_overlap_lidar_prior_stats.parquet')
    (out/'method_outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    counts=collections.Counter(r['cohort'] for r in eligible)
    summary={'rows':len(rows),'geometric_screen_pass':sum(r['reference_status']=='GEOMETRIC_SCREEN_PASS' for r in rows),'eligible_no_exposure_extreme':len(eligible),'cohorts':{str(k):v for k,v in counts.items()},'n_logs':len({r['log'] for r in rows}),'n_scenes':len({r['scene'] for r in rows}),'model_inferences':0,'scientific_failures':0,'elapsed_s':round(time.time()-started)}
    (out/'atlas_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
