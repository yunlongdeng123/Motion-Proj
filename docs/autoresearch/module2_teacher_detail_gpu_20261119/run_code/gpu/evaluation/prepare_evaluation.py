"""Freeze CPU evaluation domains, actual native baselines, ROIs and reprojections."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy import ndimage as ndi

FRAMES=(86,110,134,179,198)
EXPECTED=dict(A=16904,C=10830,S=3131,S_full=3187,S_old=2222,S_new=909,S_thin=56,D=38416,K=77266)
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bbox(mask,pad=8):
    y,x=np.where(mask)
    if not len(y):return None
    return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(480,int(x.max())+pad+1),min(320,int(y.max())+pad+1)]
def project(xyz,m):
    cam=xyz@m['world_to_camera'][:3,:3].T+m['world_to_camera'][:3,3]
    v=cam@m['camera_K'].T
    uv=v[...,:2]/np.where(np.abs(v[...,2:])>1e-20,v[...,2:],np.nan)-float(m['pixel_center_offset'])
    return uv,cam[...,2]
def bilinear(a,uv):
    x,y=uv[:,0],uv[:,1];xx=np.floor(x).astype(int);yy=np.floor(y).astype(int)
    dx,dy=x-xx,y-yy
    return sum(a[yy+oy,xx+ox]*((dx if ox else 1-dx)*(dy if oy else 1-dy))[(slice(None),)+(None,)*(a.ndim-2)]for oy in [0,1]for ox in [0,1])

def prepare(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    Q=root/'road_teacher_detail_cpu_20261119_r1';P=root/'road_multiscale_material_20261118_r1';N=root/'road_activation_20261118_r1';G=root/'road_capacity_gpu_20261118_r1';C=root/'road_capacity_cpu_20261117_r1'
    mats_path=N/'qa/material_masks.npz';freq_path=G/'check3/frozen_frequency_masks.npz';mats=load(mats_path);freq=load(freq_path)
    result=dict(schema='teacher_detail_frozen_evaluation.v1',frames={},fixed_denominators=EXPECTED,CPU_only=True,
        teacher_frames=[86,110],diagnostic_frames=[134,179,198],historical_199BUILD_not_blind=True,
        policy=dict(photo='native premult sRGB P vs frozen alpha*teacher; q1; A/C/S independently normalized',
            detail='mask-normalized Gaussian sigma1/2px truncate3; local support>=.9; boundary counts retained',
            edge='Sobel luma/8, magnitude>=4/255; 1px match; mask eroded1px; no-edge target is N/A',
            stripe='frozen G/check3 37404 domain; directional absolute second differences',
            visibility='road-only support is separate from actual joint eligible alphaT',
            no_hidden_GT='A observed; C/S pseudo; diagnostic views have no teacher targets',
            ROI='before formal outputs: masks or frozen teacher only; no result-selected crops'))
    counts={k:0 for k in EXPECTED};maps={};frozen={};rois=[];sources=[mats_path,freq_path,Q/'protocol.json',Q/'gpu_20261119/authorization.json']
    for raw in FRAMES:
        rb=Q/f'gpu_20261119/native/basis/{raw:03d}/road_basis.npz';jb=rb.with_name('joint_basis.npz')
        road,joint=load(rb),load(jb);mp=Q/f'pixel_render/maps/{raw:03d}.npz';maps[raw]=load(mp)
        dpath=C/f'ideal_targets_r2/constant/{raw:03d}.npz';D=load(dpath)['mask'];sources.extend([rb,jb,mp,dpath])
        assert np.array_equal(road['K'],joint['K']) and not joint['eligible_weight'][joint['K']].any()
        f=dict(baseline_road_rgb=road['baseline_rgb'],baseline_joint_rgb=joint['baseline_rgb'],baseline_road_alpha=road['alpha'],
            baseline_joint_alpha=joint['alpha'],K=road['K'],D=D,
            eligible_weight_road=road['eligible_weight'],eligible_weight_joint=joint['eligible_weight'],
            stripe_domain=freq[f'{raw:03d}_domain'],stripe_vertical=freq[f'{raw:03d}_vertical'],stripe_horizontal=freq[f'{raw:03d}_horizontal'])
        if raw in (86,110):
            tp=P/f'stage0_native_r1/{raw:03d}_targets.npz';t=load(tp);sources.append(tp)
            assert np.array_equal(t['fixed_alpha'],road['alpha'])
            f.update(A=t['A'],C=t['C'],S=t['S'],S_full=mats[f'{raw:03d}_full_S'],S_old=mats[f'{raw:03d}_old_photo_S'],
                S_new=mats[f'{raw:03d}_current_new_support_909'],S_thin=mats[f'{raw:03d}_current_remaining_56'],
                target_A=t['target_A'],target_C=t['target_C'],target_S=t['target_S'])
            assert np.array_equal(f['S_old']|f['S_new'],f['S']) and np.array_equal(f['S']|f['S_thin'],f['S_full'])
            cp=Q/f'pixel_render/read_operator_box_r1/{raw:03d}/constants.npz';cc=load(cp);sources.append(cp)
            mass=np.zeros(320*480,dtype=np.float64);mass[cc['row_ids']]=cc['mass'];mass=mass.reshape(320,480)
            f['reader_source_mass']=mass
            f['A_readonly_floor']=f['A']&(mass==0);f['A_source_available']=f['A']&(mass>0)
            f['A_native_zero_eligible_weight']=f['A']&(road['eligible_weight']==0)
            assert not road['eligible_weight'][f['A_readonly_floor']].any()
            for mask in ['A','C','S_full']:assert not(f[mask]&f['K']).any(),(raw,mask,'K overlap')
            for key,label in [('C','孔洞伪教师细节'),('S_full','目标阴影伪教师')]:
                rois.append(dict(raw=raw,key=key,label=label,bbox_xyxy=bbox(f[key]),selection='frozen full mask bbox +8px',teacher_is_pseudo=True))
            T=t['target_A'];lum=np.einsum('c,chw->hw',np.array([.2126,.7152,.0722]),T)
            white=f['A']&(lum>.65)&((T.max(0)-T.min(0))<.15)
            labs,n=ndi.label(white);sizes=np.bincount(labs.ravel());sizes[0]=0
            if n and sizes.max()>=3:
                k=int(sizes.argmax());rois.append(dict(raw=raw,key='observed_white_proxy',label='实拍亮色结构（不作标线真值）',bbox_xyxy=bbox(labs==k),selection='largest frozen realA component, luma>.65 and RGB spread<.15',pixels=int(sizes[k])))
            else:rois.append(dict(raw=raw,key='observed_white_proxy',label='实拍标线候选不足',bbox_xyxy=None,selection='same fixed realA white proxy rule',available=False))
        else:
            yy,xx=np.where(f['stripe_domain']);cy,cx=int(np.median(yy)),int(np.median(xx));x0=min(400,max(0,cx-40));y0=min(272,max(0,cy-24))
            rois.append(dict(raw=raw,key='diagnostic_stripe',label='诊断视角固定条纹区域（无教师）',bbox_xyxy=[x0,y0,x0+80,y0+48],selection='80x48 centered at frozen stripe-domain coordinate medians'))
        path=out/f'{raw:03d}.npz';np.savez_compressed(path,**f);frozen[raw]=f
        bp=out/f'{raw:03d}_baseline_packet.npz';np.savez_compressed(bp,road_rgb=road['baseline_rgb'],joint_rgb=joint['baseline_rgb'],road_alpha=road['alpha'],joint_alpha=joint['alpha'],K=road['K'])
        result['frames'][str(raw)]=dict(path=str(path),sha256=sha(path),baseline_packet=str(bp),baseline_packet_sha256=sha(bp),role='teacher_source'if raw in (86,110)else'diagnostic_not_blind')
        for k in counts:counts[k]+=int(f[k].sum())if k in f else 0
    assert counts==EXPECTED,counts
    assert sum(int(f.get('A_readonly_floor',np.zeros(1)).sum())for f in frozen.values())==7
    result['source_bindings']=[dict(path=str(p),sha256=sha(p))for p in sources]
    # Only actual rendered road surfaces are compared. Geometry correspondence is frozen before outputs.
    pairs=[]
    for src,dst in [(86,110),(110,86)]:
        ms,md=maps[src],maps[dst];fs,fd=frozen[src],frozen[dst]
        valid=ms['canonical_mesh_hit_valid']&fs['D']&~fs['K']&(fs['baseline_road_alpha']>.9)
        sy,sx=np.where(valid);xyz=ms['surface_world_xyz'][sy,sx];uv,dep=project(xyz,md)
        good=np.isfinite(uv).all(1)&(dep>0)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        sy,sx,xyz,uv=sy[good],sx[good],xyz[good],uv[good]
        xx,yy=np.floor(uv[:,0]).astype(int),np.floor(uv[:,1]).astype(int)
        dstvalid=md['canonical_mesh_hit_valid']&fd['D']&~fd['K']&(fd['baseline_road_alpha']>.9)
        g=np.logical_and.reduce([dstvalid[yy+dy,xx+dx]for dy in [0,1]for dx in [0,1]])
        sy,sx,xyz,uv=sy[g],sx[g],xyz[g],uv[g]
        dstxyz=bilinear(md['surface_world_xyz'],uv);err=np.linalg.norm(dstxyz-xyz,axis=1)
        cycle,_=project(dstxyz,ms);cycleerr=np.linalg.norm(cycle-np.stack([sx,sy],1),axis=1)
        g=np.isfinite(err)&(err<=.05)&(cycleerr<=.75)
        sy,sx,uv,err,cycleerr=sy[g],sx[g],uv[g],err[g],cycleerr[g]
        js=fs['eligible_weight_joint'][sy,sx];jd=bilinear(fd['eligible_weight_joint'],uv)
        roles=np.zeros(len(sy),np.uint8)
        for k,bit in [('A',1),('C',2),('S',4)]:roles[fs[k][sy,sx]]|=bit
        path=out/f'crossview_{src:03d}_{dst:03d}.npz'
        np.savez_compressed(path,src_y=sy,src_x=sx,dst_uv=uv,world_error_m=err,cycle_error_px=cycleerr,src_joint_eligible_alphaT=js,dst_joint_eligible_alphaT=jd,source_roles=roles)
        selfuv,_=project(ms['surface_world_xyz'][sy,sx],ms)
        pairs.append(dict(src=src,dst=dst,path=str(path),sha256=sha(path),matches=len(sy),self_project_error_px_max=float(np.linalg.norm(selfuv-np.stack([sx,sy],1),axis=1).max())if len(sy)else None))
    result['crossview']=dict(pairs=pairs,geometry_world_tolerance_m=.05,cycle_tolerance_px=.75,road_alpha_min=.9,
        actual_joint_visibility_threshold_alphaT=.05,interpretation='actual saved road renders reprojected on fixed surface; high-alpha diagnostic P/alpha only; same field is never re-queried; exposure/view filters and alphaT occlusion remain diagnostic')
    dump(out/'rois.json',dict(schema='before_output_fixed_ROI.v1',rois=rois))
    result['ROIs']=dict(path=str(out/'rois.json'),sha256=sha(out/'rois.json'))
    dump(out/'manifest.json',result)
    print(json.dumps(dict(manifest=str(out/'manifest.json'),fixed_denominators=counts,readonly_A7=7,crossview=pairs,ROI_count=len(rois)),ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--campaign-root',required=True);a.add_argument('--out',required=True);args=a.parse_args();prepare(args.campaign_root,args.out)
