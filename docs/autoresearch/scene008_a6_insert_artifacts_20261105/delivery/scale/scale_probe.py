"""Three-view frozen-asset scale/joint probe; one fixed HR-normalized opacity control.

No optimisation or model/kernel edits. Opacity compensation exists only in render
row copies and only on chart surfels; geometry and colours are immutable.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,hashlib,inspect,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PARENT=CAM/'scene008_a6_insert_artifacts_20261105'
DEST=PARENT/'scale'
OLD=CAM/'scene008_a6_asset_controls_20261105'
FORMAL=CAM/'scene008_a6_appearance_formal_20261105'
ASSET=FORMAL/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
EXPECTED='70826fc7c9c011ee073f7ffea4f22501a57f39ffa2cd0d52cd5ebd9c37b14e53'
ENGINE=CAM/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
DEPS=Path('/root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization')
sys.path[:0]=[str(OLD/'source'),str(ENGINE)]
from scene_runtime import Runtime,select_rows,raster
from sh_asset import SHAsset
import diff_gaussian_rasterization as raster_package
from diff_gaussian_rasterization import _C

RAWS=(106,113,130)
FLOOR=.3
CALLS=0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def tensor_hash(value):
    a=value.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def row_hash(rows,omit=()):
    return {k:None if v is None else tensor_hash(v) for k,v in rows.items() if k not in omit}


def stat(value):
    a=np.asarray(value,dtype=np.float64).ravel()
    if not len(a):return dict(count=0,mean=None,max=None)
    q=np.quantile(a,[0,.1,.5,.9,1])
    return dict(count=len(a),min=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),max=float(q[4]),mean=float(a.mean()))


def cpu(x):return x.detach().float().cpu().numpy()


def camera_hr(cam):
    c=copy.copy(cam);c.image_width=1920;c.image_height=1280
    for k in ('fx','fy','cx','cy'):setattr(c,k,getattr(cam,k)*4)
    return c


def camera_record(cam):
    return dict(wh=[cam.image_width,cam.image_height],fx=float(cam.fx),fy=float(cam.fy),cx=float(cam.cx),cy=float(cam.cy),
                FoVx=float(cam.FoVx),FoVy=float(cam.FoVy),timestamp=float(cam.timestamp),
                world_view=tensor_hash(cam.world_view_transform),full_projection=tensor_hash(cam.full_proj_transform),center=tensor_hash(cam.camera_center))


def environment(rt,cam,data):
    if cam.image_width==480:return cpu(data['env'])
    yy,xx=torch.meshgrid(torch.arange(cam.image_height,device='cuda',dtype=torch.float32),
                        torch.arange(cam.image_width,device='cuda',dtype=torch.float32),indexing='ij')
    cam.grid=torch.stack((xx,yy),-1)
    directions=cam.get_world_directions(False).permute(1,2,0)
    return cpu(rt.n.env(directions).permute(2,0,1))


def decorate(rows,asset,asset_mask):
    # Active CUDA source only accumulates the24 arbitrary feature channels;
    # no private clip metadata is read by the bound original extension.
    assert rows['features'].shape[1]==24
    assert torch.all(rows['features'][:,6:]==0)
    assert torch.all(asset.coefficients[:,1:]==0),'This diagnostic uses the actual SH0 asset RGB'
    features=rows['features'].clone()
    features[asset_mask,6:9]=asset.rgb.clamp_min(0)
    return dict(rows,features=features)


def projected_covariance_chart(A,cam,chart_mask,pipe):
    means=A['means3D'][chart_mask].double()
    view=cam.world_view_transform.double()
    t=means@view[:3,:3]+view[3,:3]
    tx=math.tan(-.5) if pipe.neg_fov else math.tan(float(cam.FoVx)*.5)
    ty=math.tan(-.5) if pipe.neg_fov else math.tan(float(cam.FoVy)*.5)
    fx=cam.image_width/(2*tx);fy=cam.image_height/(2*ty)
    z=t[:,2]
    assert bool((z>0).all())
    qx=torch.minimum(z.new_tensor(1.3*tx),torch.maximum(z.new_tensor(-1.3*tx),t[:,0]/z))
    qy=torch.minimum(z.new_tensor(1.3*ty),torch.maximum(z.new_tensor(-1.3*ty),t[:,1]/z))
    J=torch.zeros((len(z),2,3),device=z.device,dtype=torch.float64)
    J[:,0,0]=fx/z;J[:,0,2]=-fx*qx/z
    J[:,1,1]=fy/z;J[:,1,2]=-fy*qy/z
    if A['scales'] is not None:
        scales=A['scales'][chart_mask].double()
        assert float((scales-scales[:,:1]).abs().max())<1e-8
        s2=scales[:,0].square()
    else:
        cov=A['cov3D_precomp'][chart_mask].double()
        assert float(cov[:,[1,2,4]].abs().max())<1e-8
        assert float((cov[:,[0,3,5]]-cov[:,:1]).abs().max())<1e-8
        s2=cov[:,0]
    rotation_error=float((view[:3,:3].T@view[:3,:3]-torch.eye(3,device=z.device,dtype=z.dtype)).abs().max())
    assert rotation_error<3e-6
    return s2[:,None,None]*(J@J.transpose(1,2)),rotation_error


def compensation(A,cam,reference,chart_mask,pipe):
    factor=torch.ones(len(chart_mask),device='cuda',dtype=torch.float32)
    if cam.image_width==1920:
        return factor,dict(HR_factor_strictly_one=True,chart_factor=stat(np.ones(int(chart_mask.sum()))))
    G,err=projected_covariance_chart(A,cam,chart_mask,pipe)
    H,_=projected_covariance_chart(A,reference,chart_mask,pipe)
    I=torch.eye(2,device='cuda',dtype=torch.float64)[None]
    det=lambda x:torch.linalg.det(x).clamp_min(1e-24)
    f=torch.sqrt(det(G)/det(G+FLOOR*I));fh=torch.sqrt(det(H)/det(H+FLOOR*I))
    k=f/fh
    assert bool(torch.isfinite(k).all() and (k>0).all() and (k<=1+1e-8).all())
    factor[chart_mask]=k.float()
    return factor,dict(HR_factor_strictly_one=True,chart_factor=stat(cpu(k)),rotation_orthogonality_maxabs=err,
                       geometric_eigenvalues_LR=stat(cpu(torch.linalg.eigvalsh(G))),
                       native_filtered_eigenvalues_LR=stat(cpu(torch.linalg.eigvalsh(G+FLOOR*I))),
                       HR_cov_vs_16LR_maxabs=float((H-16*G).abs().max()))


def render(cam,rt,rows):
    global CALLS
    out=raster(cam,rt.pipe,rt.n.static.active_sh_degree,rows,rt.n.bg.new_zeros(3))
    CALLS+=1
    torch.cuda.synchronize()
    group_error=float((out['extra'][:4].sum(0)-out['alpha'][0]).abs().max())
    assert group_error<2e-5
    result=dict(rgb=cpu(out['rgb']),alpha=cpu(out['alpha']),asset_rgb=cpu(out['extra'][4:7]),
                asset_alpha=cpu(out['extra'][:1]),depth_sum=cpu(out['depth_sum']),
                group_alpha=cpu(out['extra'][:4]),group_conservation_maxabs=group_error,
                radii=stat(cpu(out['radii'])))
    assert all(np.isfinite(result[k]).all() for k in ('rgb','alpha','asset_rgb','asset_alpha','depth_sum'))
    del out
    return result


def down(x):
    c,h,w=x.shape;assert(h,w)==(1280,1920)
    return x.reshape(c,320,4,480,4).mean((2,4))


def down_result(o):
    return {k:down(o[k]) for k in ('rgb','alpha','asset_rgb','asset_alpha','depth_sum','group_alpha')}


def compare(o,ref,mask):
    result={}
    for k in ('rgb','alpha','asset_rgb','asset_alpha'):
        d=np.abs(o[k]-ref[k]).mean(0)
        result[k+'_MAE']=float(d[mask].mean()) if mask.any() else None
        result[k+'_maxabs']=float(np.abs(o[k]-ref[k])[:,mask].max()) if mask.any() else None
    fixed=mask&(ref['asset_alpha'][0]>.5)
    a=o['asset_rgb']/np.maximum(o['asset_alpha'],1e-5)
    b=ref['asset_rgb']/np.maximum(ref['asset_alpha'],1e-5)
    result['asset_straight_RGB_MAE_fixed_ref_alpha_gt05']=float(np.abs(a-b).mean(0)[fixed].mean()) if fixed.any() else None
    result['fixed_mask_pixels']=int(mask.sum());result['fixed_normalized_mask_pixels']=int(fixed.sum())
    return result


def dark(o,mask):
    alpha=o['asset_alpha'][0]
    straight=o['asset_rgb']/np.maximum(o['asset_alpha'],1e-5)
    opaque=mask&(alpha>.9)
    return dict(mask_pixels=int(mask.sum()),opaque_pixels=int(opaque.sum()),
                opaque_dark_straight_below02=int((opaque&(straight.mean(0)<.2)).sum()),
                asset_premul_mean=float(o['asset_rgb'].mean(0)[mask].mean()) if mask.any() else None,
                asset_alpha_mean=float(alpha[mask].mean()) if mask.any() else None)


def image(x):
    if x.shape[0]==1:x=np.repeat(x,3,axis=0)
    return Image.fromarray(np.uint8(np.clip(x.transpose(1,2,0),0,1)*255+.5))


def crop(x,box):
    x0,y0,x1,y1=box;return x[:,y0:y1,x0:x1]


def board(tiles,labels,path,target_height=320):
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),15) if fontpath.is_file() else ImageFont.load_default()
    panels=[]
    for x,label in zip(tiles,labels):
        im=image(x);im=im.resize((round(im.width*target_height/im.height),target_height),Image.Resampling.NEAREST)
        q=Image.new('RGB',(im.width,target_height+46),(237,243,247));q.paste(im,(0,46))
        ImageDraw.Draw(q).text((7,7),label,fill=(15,35,45),font=font);panels.append(q)
    out=Image.new('RGB',(sum(p.width for p in panels),target_height+46));x=0
    for p in panels:out.paste(p,(x,0));x+=p.width
    out.save(path)


@torch.no_grad()
def main():
    DEST.mkdir(parents=True,exist_ok=False);start=time.time()
    assert sha(ASSET)==EXPECTED
    build=DEPS/'build/lib.linux-x86_64-3.9/diff_gaussian_rasterization/_C.cpython-39-x86_64-linux-gnu.so'
    forward=DEPS/'cuda_rasterizer/forward.cu'
    text=forward.read_text()
    assert 'cov[0][0] += 0.3f' in text and 'opacities[idx] }' in text
    assert sha(build)==sha(_C.__file__)
    assert 'features[collected_id[j] * S + ch]' in text and 'MAX_FEATURES 24' in (DEPS/'cuda_rasterizer/config.h').read_text()
    evidence=[Path(__file__),Path(_C.__file__),Path(raster_package.__file__),forward,build,OLD/'source/scene_runtime.py',OLD/'source/sh_asset.py',ASSET]
    hashes={str(p):sha(p) for p in evidence}
    rt=Runtime();asset=SHAsset.from_geometry(ASSET).cuda()
    assert len(asset.xyz)==50322
    torch.cuda.reset_peak_memory_stats()
    chart=asset.representation_kind==1
    assert int(chart.sum())==47544
    records=[]
    for raw in RAWS:
        per=DEST/f'raw{raw}';per.mkdir()
        d=rt.raw(raw);lr=d['cam'];hr=camera_hr(lr)
        assert torch.equal(lr.world_view_transform,hr.world_view_transform)
        assert torch.equal(lr.full_proj_transform,hr.full_proj_transform)
        scene=rt.scene(raw,asset,delete=False)
        J=scene['rows'];amask=J['features'][:,2]>.5;m37=J['features'][:,3]>.5
        assert int(amask.sum())==len(asset.xyz) and scene['meta']['native_a6_rows_input']==0
        J=decorate(J,asset,amask);A=select_rows(J,amask)
        before=row_hash(A);nonopacity=row_hash(A,('opacities',))
        factors,factor_stats=compensation(A,lr,hr,chart,rt.pipe)
        hr_factors,_=compensation(A,hr,hr,chart,rt.pipe)
        assert torch.equal(hr_factors,torch.ones_like(hr_factors))
        asset_indices=torch.where(amask)[0]
        ref_outputs=None;record=dict(raw=raw,cameras=dict(LR=camera_record(lr),HR=camera_record(hr)),
            asset_row_hash=before,compensation=factor_stats,arms={},scope='internal development diagnostic, not independent confirmation')
        for resolution,cam in (('HR',hr),('LR',lr)):
            E=environment(rt,cam,d)
            B=render(cam,rt,dict(J,mask=J['mask']&~amask))
            Bd=render(cam,rt,dict(J,mask=J['mask']&~amask&~m37))
            for arm in ('baseline',) if resolution=='HR' else ('baseline','area_compensated'):
                if arm=='area_compensated':
                    op=J['opacities'].clone();op[asset_indices]=op[asset_indices]*factors[:,None]
                    assert torch.equal(op[asset_indices][~chart],J['opacities'][asset_indices][~chart])
                    assert torch.equal(op[~amask],J['opacities'][~amask])
                    Jr=dict(J,opacities=op)
                else:Jr=J
                Ar=select_rows(Jr,amask)
                assert row_hash(Ar,('opacities',))==nonopacity
                if arm=='baseline':assert row_hash(Ar)==before
                solo=render(cam,rt,Ar)
                joint=render(cam,rt,Jr)
                deletion=render(cam,rt,dict(Jr,mask=Jr['mask']&~m37))
                aux_error=float(np.abs(solo['asset_rgb']-solo['rgb']).max());assert aux_error<4e-6
                mono=float(np.maximum(joint['asset_alpha']-solo['asset_alpha'],0).max())
                assert mono<2e-5
                assert float(np.maximum(deletion['asset_alpha']-solo['asset_alpha'],0).max())<2e-5
                alpha_res=float(np.abs(joint['alpha']-(solo['alpha']+B['alpha']-solo['alpha']*B['alpha'])).max())
                alpha_res_d=float(np.abs(deletion['alpha']-(solo['alpha']+Bd['alpha']-solo['alpha']*Bd['alpha'])).max())
                if resolution=='HR':
                    yy,xx=np.where(solo['alpha'][0]>.01)
                    boxH=(max(0,(int(xx.min())-48)//4*4),max(0,(int(yy.min())-48)//4*4),
                          min(1920,((int(xx.max())+52)//4)*4),min(1280,((int(yy.max())+52)//4)*4))
                    boxL=tuple(v//4 for v in boxH)
                    ref_outputs={n:down_result(o) for n,o in (('solo',solo),('joint',joint),('delete',deletion))}
                    ref_outputs['env']=down(E)
                    mask=ref_outputs['solo']['alpha'][0]>.01
                    fixed_mask=mask;interior=binary_erosion(mask,iterations=2);edge=mask&~interior
                    if raw==106:
                        replay=render(cam,rt,dict(Ar,opacities=Ar['opacities']*hr_factors[:,None]))
                        record['HR_compensated_identity_replay_exact']={k:bool(np.array_equal(replay[k],solo[k])) for k in ('rgb','alpha','asset_rgb','asset_alpha','depth_sum')}
                        assert all(record['HR_compensated_identity_replay_exact'].values())
                        del replay
                box=boxH if resolution=='HR' else boxL
                outdir=per/(resolution+'_'+arm);outdir.mkdir()
                for name,o in (('solo',solo),('joint_keep',joint),('joint_delete',deletion)):
                    np.savez_compressed(outdir/(name+'_crop_float.npz'),**{k:crop(o[k],box) for k in ('rgb','alpha','asset_rgb','asset_alpha','depth_sum','group_alpha')},crop_xyxy=np.array(box))
                board([crop(solo['rgb'],box),crop(joint['rgb']+(1-joint['alpha'])*E,box),crop(deletion['rgb']+(1-deletion['alpha'])*E,box)],
                      [resolution+' '+arm+' solo/black',resolution+' '+arm+' joint/a37',resolution+' '+arm+' joint/no-a37'],outdir/'same_resolution_fair.png')
                arm_record=dict(aux_asset_RGB_vs_solo_native_RGB_maxabs=aux_error,asset_alpha_joint_exceeds_solo_max=mono,
                    alpha_total_product_identity_maxabs_keep=alpha_res,alpha_total_product_identity_maxabs_delete=alpha_res_d,
                    group_conservation_maxabs=max(o['group_conservation_maxabs'] for o in (solo,joint,deletion)),
                    native_a6_rows_input=0,effective_a37_rows_delete=0,asset_rows=len(asset.xyz),
                    unchanged_geometry_and_colours=True,opacity_change='chart-only row-copy factor' if arm!='baseline' else 'none')
                if resolution=='LR':
                    arm_record['vs_area_downsample_HR']={n:{region:compare(o,ref_outputs[n],m) for region,m in (('object',fixed_mask),('interior',interior),('edge',edge))}
                        for n,o in (('solo',solo),('joint',joint),('delete',deletion))}
                    arm_record['dark_opaque_diagnostics']={n:dark(o,fixed_mask) for n,o in (('solo',solo),('joint',joint),('delete',deletion))}
                    board([crop(solo['rgb'],box),crop(ref_outputs['solo']['rgb'],box),crop(joint['asset_rgb'],box),crop(ref_outputs['joint']['asset_rgb'],box)],
                          [arm+' direct480 solo','area1920->480 solo',arm+' direct480 joint A RGB','area1920->480 joint A RGB'],outdir/'sampling_comparison.png')
                    if arm=='baseline':
                        base_solo=solo;base_joint=joint;base_env=E
                    else:
                        board([crop(base_solo['rgb'],box),crop(solo['rgb'],box),crop(ref_outputs['solo']['rgb'],box),
                               crop(base_joint['rgb']+(1-base_joint['alpha'])*base_env,box),crop(joint['rgb']+(1-joint['alpha'])*E,box)],
                              ['base480 solo','area-comp480 solo','area1920->480 solo','base480 same-cam joint','area-comp480 same-cam joint'],per/'base_vs_comp_fair480.png')
                record['arms'][resolution+'_'+arm]=arm_record
                if resolution=='HR':del solo,joint,deletion
                elif arm=='area_compensated':del solo,joint,deletion,base_solo,base_joint,base_env
                del Ar
            del B,Bd,E
        record['crop_HR_xyxy']=boxH;record['crop_LR_xyxy']=boxL
        assert row_hash(A)==before
        assert rt.modelunchanged() and asset.assert_geometry_unchanged()
        write(per/'receipt.json',record);records.append(record)
        print(json.dumps(dict(stage='scale_view_complete',raw=raw,render_calls=CALLS,seconds=time.time()-start)),flush=True)
        del d,scene,J,A,factors,ref_outputs
        rt._data.clear();torch.cuda.empty_cache()
    assert hashes=={str(p):sha(p) for p in evidence}
    binding=dict(files=hashes,active_extension_matches_deps_build_byte_exact=True,native_filter_variance_px2=FLOOR,
        opacity_area_compensation_in_native=False,prefiltered=False,feature_channels=24,
        feature_RGB_is_auxiliary_and_does_not_change_RGB_alpha=True,
        pipe_neg_fov=bool(rt.pipe.neg_fov),pipe_compute_cov3D_python=bool(rt.pipe.compute_cov3D_python),
        native_background=cpu(rt.n.bg).tolist(),SH_degree=int(rt.n.static.active_sh_degree))
    result=dict(status='PASS_THREE_VIEW_FROZEN_ASSET_SCALE_PROBE',raws=list(RAWS),asset=str(ASSET),asset_sha256=EXPECTED,
        binding=binding,views=records,render_calls=CALLS,seconds=time.time()-start,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
        opacity_control=dict(formula='k_R=sqrt(det(G_R)/det(G_R+0.3I))/sqrt(det(G_1920)/det(G_1920+0.3I))',
          chart_only=True,HR_factor_strictly_one=True,reference='1920x1280 at the same view; not full Mip-Splatting',
          methods_reference='https://niujinshuchong.github.io/mip-splatting/',
          official_code='https://github.com/autonomousvision/mip-splatting/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu'),
        comparison_target='Area-downsampled HR actual-render RGBpremul/alpha; not hidden appearance ground truth',
        pixel_center_mapping='x_L=(x_H+0.5)/4-0.5; exact 4x4 area means of premultiplied RGB and alpha separately',
        model_weight_updates=0,geometry_updates=0,colour_updates=0,source_opacity_updates=0,shared_kernel_edits=0,
        automatic_adoption=False,geometry_accuracy_claim=False,HELD_views_are_internal_diagnostics=True)
    write(DEST/'receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('views','binding')},indent=2),flush=True)


if __name__=='__main__':main()
