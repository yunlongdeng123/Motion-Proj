"""只读199帧原图与已有测距来源：原尺寸核心分辨率、清晰度和可见性排名。"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='2'
from pathlib import Path
import json,time,sys,math
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,binary_erosion,map_coordinates,gaussian_filter,sobel,laplace
from scipy.spatial import cKDTree

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_gaussian_source_quality_20261122_r1';OUT=ROOT/'source_view_ranking'
SRC=C/'road_gaussian_transfer_20261122_r1';S=C/'road_multisource_20261122_r1/sources'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1');BANK=B/'inputs'
ANN=C/'waymo_val065_road_completion_20261111_r1/raw_road_annotations'
sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.v44_road_observation import ray_box_blocked,build_foreground_guard

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def srgb(c):return np.where(c<=.0031308,c*12.92,1.055*np.maximum(c,0)**(1/2.4)-.055)
def linear(c):return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
def project(X,K,T):
    q=np.asarray(X)@T[:3,:3].T+T[:3,3];p=q@K.T
    return p[:,:2]/np.maximum(p[:,2:],1e-8)-.5,q[:,2]
def camera(obs,frame_id):
    cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (BANK/'calib'/(frame_id+'.txt')).read_text().splitlines()}
    K=cal['P0'].reshape(3,4)[:,:3].copy();T=np.eye(4);T[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
    return K,T@np.linalg.inv(np.array(obs['ego_world']).reshape(4,4))
def caliper(uv):
    widths=[]
    for edge in np.roll(uv,-1,axis=0)-uv:
        n=np.array([-edge[1],edge[0]]);n/=max(np.linalg.norm(n),1e-9);p=uv@n;widths.append(float(p.max()-p.min()))
    return min(widths)
def roi(uv,shape,pad=0):
    h,w=shape;lo=np.maximum(0,np.floor(uv.min(0)).astype(int)-pad);hi=np.minimum([w,h],np.ceil(uv.max(0)).astype(int)+pad+1)
    if (hi<=lo).any():return None,None
    im=Image.new('L',tuple((hi-lo).astype(int)));ImageDraw.Draw(im).polygon([tuple(x) for x in uv-lo],fill=255)
    return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],np.asarray(im)>0
def sample(a,uv,order=1,cval=0):return map_coordinates(np.asarray(a,np.float32),[uv[:,1],uv[:,0]],order=order,mode='constant',cval=cval)
def polygon_pixels(uv,shape):
    box,mask=roi(uv,shape)
    if box is None or not mask.any():return box,mask,None
    y,x=np.nonzero(mask);return box,mask,np.c_[x+box[0],y+box[1]]
def crop_save(image,uv,path,pad=0):
    box,mask=roi(uv,(image.height,image.width),pad)
    if box is None:return None
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.crop(tuple(box)).save(path)
    return dict(path=str(path),bbox_xyxy=box,pixels_unchanged=True,native_crop_wh=[box[2]-box[0],box[3]-box[1]])

def retrieval_audit(g,f,selection,audit):
    X=g['world_xyz'];r=f['real_same_position'];p=load(C/'road_surface_residual_20261121_r2/appearance_fit/asset/provenance.npz')
    selected=selection['candidates'];reasons={}
    for x in audit['rows']:
        for k in x['rejection_reasons']:reasons[k]=reasons.get(k,0)+1
    return dict(complete=True,prior_algorithm_scope='Only existing fixed road chart queried through five cameras; not a street-wide road-Gaussian search.',
        geometry_query_frames=[int(x) for x in np.unique(g['target_raw'])],geometry_query_pixels=len(X),
        query_world_bounds=[X.min(0).tolist(),X.max(0).tolist()],R2_Gaussian_world_bounds=[p['world_xyz'].min(0).tolist(),p['world_xyz'].max(0).tolist()],
        strict_real_query_count=int(r.sum()),strict_real_world_bounds=[X[r].min(0).tolist(),X[r].max(0).tolist()],
        query_pixels_are_not_independent_world_points=True,clean_asphalt_queries=selection['clean_asphalt_queries'],
        occupied_4cm_cells=selection['occupied_world_grid_cells'],occupied_grid_area_proxy_m2=selection['occupied_area_grid_proxy_m2'],
        proposal_sizes=selection['selection_policy'].get('sizes',[[.6,.6],[.8,.8],[1.,1.],[1.2,.8],[1.2,1.2],[1.6,1.]]),
        proposals_before_cpu_filters=selection['proposals'],eligible_after_cpu_filters=selection['eligible_CPU_proposals'],
        nonoverlap_kept=selection['candidate_count'],rejection_reason_counts_nonexclusive=reasons,
        search_limitations=[
          'Candidate centers required >=70% occupancy of the existing strict-real asphalt grid; no query or parent Gaussian outside this chart was searched.',
          'Density maxima and .18m center spacing yielded only17 proposals; this is the result of bounded inputs/gates, not evidence that the street has only17 feasible road blocks.',
          'CPU photo fidelity admission used existing R2 native caches in only five target cameras; no_cached_native_photo_fidelity_pass can mean too few qualified cached pixels, not intrinsically bad photo texture.',
          'The final native qualifier explicitly hard-coded patch00/02 and raw86/110. Patch01 and other retained/overlapping alternatives had no equivalent native all-frame comparison.',
          'Ranking favored area/coverage and HF alignment, did not require a minimum original-image shortest footprint axis or minimum native-HF energy/retention.',
          'The prior score compared candidates only under current R2 geometry/sigma/opacity, not an upper bound after re-estimating source GS from original photographs.'
        ],
        native_qualified_candidate_ids=sorted({x['patch'] for x in read(SRC/'source_native/receipt.json')['quality']}),
        native_qualified_frames=sorted({x['raw'] for x in read(SRC/'source_native/receipt.json')['quality']}),
        conclusion='Cannot conclude there is no better source on this street from17→4→3, or that Gaussian transfer fails because the chosen bounded source is blurry.',
        new_diffusion_calls=0,GPU_calls=0)

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve completed scan; use a new output for changed criteria')
    g=load(S/'geometry.npz');f=load(S/'fusion.npz');selection=read(SRC/'source_selection/manifest.json');audit=read(SRC/'source_selection/proposal_audit.json')
    write(OUT/'retrieval_audit.json',retrieval_audit(g,f,selection,audit))
    bank_receipt=read(BANK/'receipt.json');frame_ids={int(x['raw']):x['id'] for x in bank_receipt['frames']}
    observations={raw:read(BANK/f'observations/{raw:03d}.json') for raw in bank_receipt['build_raws']}
    patches={};N=len(g['world_xyz'])
    for entry in selection['candidates']:
        cid=entry['candidate_id'];p=load(entry['packet']);R=p['source_tangent_frame_world'][:3,:3];center=p['source_center_world'];size=p['core_size_m'];q=p['fresh_query_ids']
        local=(g['world_xyz'][q]-center)@R;pointkeys=np.floor((local[:,:2]+size/2)/.04).astype(int)
        denominator=len(np.unique(pointkeys,axis=0));uv=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*size/2
        halo=size+2*float(p['source_world_margin_radius_m'].max())
        hxy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*halo/2
        vv=np.linspace(-.45,.45,9);sy,sx=np.meshgrid(vv,vv);sample_uv=np.c_[sx.ravel()*size[0],sy.ravel()*size[1]]
        # Existing fixed-world samples only; nearest point is used for box-ray tests, not new depth certification.
        _,near=cKDTree(local[:,:2]).query(sample_uv)
        samples=g['world_xyz'][q[near]]
        patches[cid]=dict(packet=p,entry=entry,R=R,center=center,size=size,core_world=center+np.c_[uv,np.zeros(4)]@R.T,
            halo_world=center+np.c_[hxy,np.zeros(4)]@R.T,samples=samples,
            world_cells_denominator=denominator,records=[])
    for raw in bank_receipt['build_raws']:
        tic=time.monotonic();d=observations[raw];K,T=camera(d,frame_ids[raw]);origin=np.linalg.inv(T)[:3,3]
        photo_image=Image.open(BANK/d['image_path']).convert('RGB');photo=np.asarray(photo_image,np.float32)/255;H,W=photo.shape[:2]
        small=np.asarray(photo_image.resize((480,320),Image.Resampling.LANCZOS),np.float32)/255
        ann=read(ANN/f'{raw:03d}.json');fg=np.asarray(Image.open(B/f'repaired_masks_all_r4/{raw:03d}/dynamic_union.png'))>0
        labels=np.asarray(Image.open(B/f'repaired_masks_all_r4/{raw:03d}/labels.png'))>0
        rect=np.asarray([x['xyxy'] for x in ann['front_camera_rectangles']]).reshape(-1,4)*.25
        guard=build_foreground_guard((320,480),rect,binary_dilation(fg|labels,iterations=2),2)
        sem=np.asarray(Image.open(B/f'semantics/{raw:03d}.png'));road_small=np.asarray(Image.fromarray((sem==0).astype(np.uint8)).resize((480,320),Image.Resampling.NEAREST))>0
        lum=linear(small)@np.array([.2126,.7152,.0722],np.float32);cleanroad=road_small&~guard
        den=gaussian_filter(cleanroad.astype(np.float32),12);local=gaussian_filter(lum*cleanroad,12)/np.maximum(den,1e-6)
        shadow=(lum<.72*local)&(local-lum>.02)&(den>.2)&road_small
        centers=np.array([x['center'] for x in ann['boxes']]).reshape(-1,3);halves=np.array([x['half'] for x in ann['boxes']]).reshape(-1,3)
        axes=np.array([x['axes'] for x in ann['boxes']]).reshape(-1,3,3)
        for cid,a in patches.items():
            p=a['packet'];uv,z=project(a['core_world'],K,T);huv,hz=project(a['halo_world'],K,T)
            inside=bool((z>.1).all()&(uv[:,0]>=0).all()&(uv[:,0]<W-1).all()&(uv[:,1]>=0).all()&(uv[:,1]<H-1).all())
            axis=caliper(uv) if (z>.1).all() else 0.
            ccam=a['center']@T[:3,:3].T+T[:3,3];J=np.array([[K[0,0]/ccam[2],0,-K[0,0]*ccam[0]/ccam[2]**2],[0,K[1,1]/ccam[2],-K[1,1]*ccam[1]/ccam[2]**2]])@T[:3,:3]@a['R'][:,:2]@np.diag(a['size'])
            singular=np.linalg.svd(J,compute_uv=False)
            view_cos=float(abs((origin-a['center'])@a['R'][:,2])/np.linalg.norm(origin-a['center']))
            wsel=p['witness_source_raw']==raw;qq=p['witness_query_id'][wsel];wuv=p['witness_source_uv_native'][wsel]
            cells=np.floor((((g['world_xyz'][qq]-a['center'])@a['R'])[:,:2]+a['size']/2)/.04).astype(int)
            world_cells=len(np.unique(cells,axis=0));unique_pixels=len(np.unique(np.floor(wuv+.5).astype(np.int32),axis=0))
            sample_blocked=ray_box_blocked(a['samples'],origin,centers,halves,axes)
            r=dict(candidate_id=cid,raw=int(raw),native_wh=[W,H],core_min_caliper_native_px=float(axis),
                   core_center_singular_axes_native_px=singular.tolist(),core_native_uv=uv.tolist(),halo_native_uv=huv.tolist(),
                   core_in_frame=inside,source_center_depth_m=float(ccam[2]),view_cosine=view_cos,
                   box_clear_sample_fraction=float((~sample_blocked).mean()),box_ray_sample_count=len(sample_blocked),
                   accepted_witness_query_records=len(qq),accepted_distinct_native_pixels=unique_pixels,
                   accepted_distinct_world_4cm_cells=world_cells,core_existing_strict_world_4cm_cells=a['world_cells_denominator'],
                   fresh_world_cell_support_fraction=float(world_cells/a['world_cells_denominator']),
                   original_RGB=str(BANK/d['image_path']),camera_K_native=K.tolist(),world_to_camera=T.tolist(),camera_origin_world=origin.tolist(),
                   Brown_intrinsics_retained=d['camera_intrinsic_Brown'],pixel_convention='Pinhole calib@inverse(ego_world), projected pixel minus.5; consistent with inherited native renderer.')
            box,mask,pixels=polygon_pixels(uv,(H,W))
            if pixels is None or not inside or axis<3:
                r.update(photo_legal=False,rejection_reasons=['out_of_frame_or_insufficient_footprint'],ranking_score=0.);a['records'].append(r);continue
            puv=(pixels+.5)/4-.5;x,y=pixels[:,0].astype(int),pixels[:,1].astype(int)
            clear=sample(guard,puv,0,1)<.5;road=sem[y,x]==0;dark=sample(shadow,puv,0,1)>.5
            # Near-object stronger shadow deficit applies only in that search region.
            near=False
            for b,c,h,ax in zip(ann['boxes'],centers,halves,axes):
                if b['type'] in (1,2,4) and (abs(((a['center']-c)@ax)[:2])<=h[:2]+2).all():near=True;break
            if near:dark|=(sample(lum,puv)<.85*sample(local,puv))&(sample(local,puv)-sample(lum,puv)>.012)
            crop=photo[box[1]:box[3],box[0]:box[2]];gray=crop@np.array([.2126,.7152,.0722],np.float32)
            inner=binary_erosion(mask,iterations=2);npx=int(inner.sum())
            if npx<16:inner=mask;npx=int(mask.sum())
            gx=sobel(gray,axis=1)/8;gy=sobel(gray,axis=0)/8;l=laplace(gray);hp=gray-gaussian_filter(gray,1.)
            metrics=dict(laplacian_variance=float(np.var(l[inner])),Tenengrad_mean=float(np.mean(gx[inner]**2+gy[inner]**2)),
                         photo_gray_std=float(np.std(gray[inner])),highpass_sigma1_std=float(np.std(hp[inner])),
                         photo_luma_mean=float(gray[inner].mean()),sharpness_native_pixels=npx)
            r.update(metrics);r.update(core_polygon_native_pixels=int(mask.sum()),dynamic_clear_fraction=float(clear.mean()),
                       road_semantic_fraction=float(road.mean()),shadow_proxy_clear_fraction=float((~dark).mean()))
            rejects=[]
            if axis<12:rejects.append('short_core_axis_under12_native_pixels')
            if view_cos<.12:rejects.append('grazing_view_cosine')
            if clear.mean()<.99:rejects.append('dynamic_or_2D_box_guard')
            if road.mean()<.98:rejects.append('nonroad_semantics')
            if (~dark).mean()<.98:rejects.append('shadow_photometric_proxy')
            if (~sample_blocked).mean()<.99:rejects.append('3D_OBB_ray_occlusion')
            r.update(photo_legal=not rejects,rejection_reasons=rejects,
                     old_fresh_certificate_supported=bool(world_cells>=3))
            a['records'].append(r)
        if raw%20==0:print('SCANNED',raw,'seconds_per_frame',time.monotonic()-tic,flush=True)
        photo_image.close()
    outputs=[];requests=[]
    old_native=read(SRC/'source_native/receipt.json')['quality']
    for cid,a in patches.items():
        valid=[r for r in a['records'] if r['photo_legal']]
        if valid:
            med=max(float(np.median([r['laplacian_variance'] for r in valid])),1e-12)
            for r in valid:
                focus=min(max(r['laplacian_variance']/med,.25),4)**.25
                r['ranking_score']=r['core_min_caliper_native_px']*focus
                r['detail_resolution_score']=r['core_min_caliper_native_px']*math.sqrt(max(r['highpass_sigma1_std'],0))
        valid.sort(key=lambda r:(-r['ranking_score'],-r['core_min_caliper_native_px'],-r['accepted_distinct_world_4cm_cells']))
        for rank,r in enumerate(valid):r['photo_legal_rank']=rank
        # Photo legality and old measured/cross-frame RGB support are kept as independent columns.
        measured=[r for r in valid if r['old_fresh_certificate_supported']]
        export=[]
        for r in valid:
            if len(export)>=3:break
            if all(abs(r['raw']-x['raw'])>=8 for x in export):export.append(r)
        baseline=[r for r in a['records'] if any(x['patch']==cid and x['raw']==r['raw'] for x in old_native)]
        for r in export:
            im=Image.open(r['original_RGB']).convert('RGB');raw=r['raw']
            r['core_original_crop']=crop_save(im,np.array(r['core_native_uv']),OUT/f'photos/{cid}/raw{raw:03d}_core_native.png',0)
            r['halo_original_crop']=crop_save(im,np.array(r['halo_native_uv']),OUT/f'photos/{cid}/raw{raw:03d}_halo_native.png',0)
            im.close()
        selected=[]
        for r in valid:
            origin=np.array(r['camera_origin_world'])
            if not selected or min(np.linalg.norm(origin-np.array(x['camera_origin_world'])) for x in selected)>=.2:selected.append(r)
            if len(selected)>=3:break
        if measured and measured[0]['raw'] not in [r['raw'] for r in selected]:selected.append(measured[0])
        for r in baseline:
            if r['raw'] not in [x['raw'] for x in selected]:selected.append(r)
        for r in selected:
            K=np.array(r['camera_K_native']);K1152=K.copy();K1152[:2]*=.6
            requests.append(dict(candidate_id=cid,raw=r['raw'],camera_K_native=r['camera_K_native'],camera_K_1152=K1152.tolist(),
                 world_to_camera=r['world_to_camera'],native_wh=r['native_wh'],core_native_uv=r['core_native_uv'],
                 original_RGB=r['original_RGB'],core_min_caliper_native_px=r['core_min_caliper_native_px'],
                 photo_legal=r['photo_legal'],accepted_distinct_world_4cm_cells=r['accepted_distinct_world_4cm_cells'],
                 source_packet=str(SRC/'source_selection/candidates'/f'{cid}.npz'),
                 reason='best/diverse actual photograph or original native qualification baseline; no source-frame number hard-coded in photo ranking'))
        cameras=np.array([r['camera_origin_world'] for r in valid])
        max_axis=max((r['core_min_caliper_native_px'] for r in valid),default=0)
        report=dict(candidate_id=cid,core_size_m=a['size'].tolist(),core_existing_strict_world_4cm_cells=a['world_cells_denominator'],
          raw_frames_scanned=len(a['records']),photo_legal_views=len(valid),old_fresh_supported_photo_legal_views=len(measured),
          max_short_core_axis_native_px=max_axis,best_ranked_photo_views=valid[:12],best_old_fresh_supported_views=measured[:8],
          old_native_qualification_views=baseline,exported_original_views=[r['raw'] for r in export],
          legal_camera_world_bounds=[cameras.min(0).tolist(),cameras.max(0).tolist()] if len(cameras) else None,
          strict_view_qualification='in-frame, >=12 native-pixel shortest core axis, viewcos>=.12, >=99% full-mask/rectangle clear, >=98% semantic road, >=98% shadow-proxy clear and >=99% box-ray samples clear',
          support_count_semantics='Old accepted multiview evidence is separately deduplicated by source native pixel and4cm source-world cell. It is not an independent texture count.',
          visibility_limit='Photographs constrain only this source block. More/legal sharper source photos are not target-location ground truth. Current fixed-geometry/depth gating can reject otherwise visible photo pixels.',
          all_frame_records=a['records'])
        write(OUT/f'{cid}_views.json',report);outputs.append({k:v for k,v in report.items() if k!='all_frame_records'})
        print('PATCH',cid,'legal',len(valid),'fresh_supported',len(measured),'max_native_short_axis',max_axis,'top',[r['raw'] for r in valid[:5]],flush=True)
    write(OUT/'native_render_requests.json',dict(complete=True,requests=requests,GPU_calls_by_auditor=0,new_diffusion_calls=0,
      camera_contract='Exact inherited FRONT pinhole; full original1920x1280 calibration plus scaled1152 matrix. Brown coefficients are retained in full records, not applied here.',
      requested_comparison='Render original source GS at these same real cameras and compare original core/halo pixels; do not compare a similar position as target GT.'))
    write(OUT/'manifest.json',dict(complete=True,run_id=ROOT.name,frames_scanned=bank_receipt['build_raws'],
       source_candidates=list(patches),rows=outputs,retrieval_audit=str(OUT/'retrieval_audit.json'),
       native_requests=str(OUT/'native_render_requests.json'),new_diffusion_calls=0,GPU_calls=0,optimizer_updates=0,
       original_images_modified=False,AI_enhancement=False,image_export='Unresized crops copied from original1920x1280 RGB PNG pixels.',
       independent_shadow_labels=False,source_camera_projection_is_inherited_pinhole=True,seconds=time.monotonic()-start))
    print('COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
