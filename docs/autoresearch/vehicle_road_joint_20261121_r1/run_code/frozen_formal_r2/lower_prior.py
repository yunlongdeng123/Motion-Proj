"""有尺寸与可见轮廓约束的皮卡下半形状先验；非实测隐藏车体。"""
import numpy as np
import torch
import torch.nn.functional as F

def install(e,m):
    original=e.initial;half=np.asarray(e.targets[110]['box']['dimensions'])/2
    inside=(np.abs(original['xyz'])<=half+np.array([.15,.15,.10])).all(1)
    observed_support=np.zeros(len(original['xyz']),bool)
    for raw in m.TRAIN:
        p=m.ten(original['xyz'])@e.poses[raw][:3,:3].T+e.poses[raw][:3,3];uv,z=m.project(p,e.cameras[raw]);ij=uv.long();valid=(z>0)&(ij[:,0]>=0)&(ij[:,0]<1152)&(ij[:,1]>=0)&(ij[:,1]<768)
        idx=torch.nonzero(valid).reshape(-1);v=torch.zeros(len(p),device='cuda',dtype=torch.bool);v[idx]=e.targets[raw]['mask'][ij[idx,1],ij[idx,0]];observed_support|=m.cpu(v).numpy()
    # Keep every inherited Gaussian that actually contributes through the unified
    # rasterizer to a true visible a20 pixel, including centers outside the mask.
    weights=torch.ones(len(original['xyz']),device='cuda',requires_grad=True)
    previous_rows=e.vehicle.rows
    def tagged_rows(pose,cam):
        rows=previous_rows(pose,cam);f=rows['features'].clone();f[:,2]=weights
        return dict(rows,features=f)
    e.vehicle.rows=tagged_rows;contribution=np.zeros(len(original['xyz']),np.float64)
    for raw in m.TRAIN:
        out=e.render(raw);score=out['vehicle_alpha'][e.targets[raw]['mask']].sum();g=torch.autograd.grad(score,weights)[0]
        contribution+=m.cpu(g).numpy()
    e.vehicle.rows=previous_rows
    observed_support=contribution>1e-8
    selected=inside|observed_support
    kept={k:v[selected].copy() for k,v in original.items()};visible=observed_support[selected]
    colors=[]
    for raw in m.TRAIN:
        a=m.cpu(e.targets[raw]['rgb']).numpy()[:,m.cpu(e.targets[raw]['mask']).numpy()].T
        luminance=a.mean(1);colors.append(a[luminance<np.quantile(luminance,.65)])
    paint=np.median(np.concatenate(colors),axis=0).clip(.02,.15).astype(np.float32)
    hidden=(kept['xyz'][:,2]<-.15)&~visible
    kept['coefficients'][hidden]=0;kept['coefficients'][hidden,0]=(paint-.5)/.28209479177387814
    width=float(min(half[1]*.88,np.quantile(np.abs(kept['xyz'][visible,1]),.95))) if visible.sum()>50 else float(half[1]*.88)
    width=max(.7,width);chunks=[];counts=[]
    def panel(points,q,scale,color,name):
        points=np.asarray(points,np.float32);allowed=np.ones(len(points),bool)
        # A prior may occupy original a9 pixels. It may not overwrite observed a20,
        # any retained vehicle, or points beneath the annotated vehicle base.
        for raw in m.FRAMES:
            xyz=m.ten(points)@e.poses[raw][:3,:3].T+e.poses[raw][:3,3];uv,z=m.project(xyz,e.cameras[raw]);ij=uv.long();valid=(z>0)&(ij[:,0]>=0)&(ij[:,0]<1152)&(ij[:,1]>=0)&(ij[:,1]<768)
            block=(e.targets[raw]['labels']>0)&(e.targets[raw]['labels']!=m.DELETE)
            block=F.max_pool2d(block.float()[None,None],7,1,3)[0,0]>0
            idx=torch.nonzero(valid).reshape(-1);blocked=torch.zeros(len(points),device='cuda',dtype=torch.bool);blocked[idx]=block[ij[idx,1],ij[idx,0]]
            allowed&=~m.cpu(blocked).numpy()
            a9=e.targets[raw]['labels']==m.DELETE
            min_depth=float(z[valid].min()) if valid.any() else 1.
            radius=int(np.ceil(4.5*max(scale)*e.scale*max(e.cameras[raw].fx,e.cameras[raw].fy)/max(min_depth,1e-6)))+2
            kernel=2*radius+1
            a9_interior=F.avg_pool2d(a9.float()[None,None],kernel,1,radius)[0,0]>.999
            hidden=torch.zeros(len(points),device='cuda',dtype=torch.bool);hidden[idx]=a9_interior[ij[idx,1],ij[idx,0]]
            allowed&=m.cpu(hidden).numpy()
        allowed&=points[:,2]>=-half[2]+.01
        points=points[allowed];n=len(points);coeff=np.zeros((n,16,3),np.float32);coeff[:,0]=(np.asarray(color)-.5)/.28209479177387814
        chunks.append(dict(xyz=points,scales=np.tile(scale,(n,1)).astype(np.float32),rotation=np.tile(q,(n,1)).astype(np.float32),coefficients=coeff,opacity=np.full(n,.92,np.float32),source_row=np.full(n,-1,np.int64),source_branch=np.full(n,3,np.int8),source_raw=np.full(n,-1,np.int16)));counts.append(dict(component=name,candidates=len(allowed),accepted=n))
    ys=np.arange(-width,width+.001,.055);zs=np.arange(-.64,-.10+.001,.055)
    panel([[-half[0]+.06,y,z] for y in ys for z in zs],[.70710678,0,.70710678,0],[.047,.047,.014],paint,'rear_lower_body')
    xs=np.arange(-half[0]+.06,half[0]-.15,.07);zs2=np.arange(-.57,-.15,.06)
    for side in (-1,1):panel([[x,side*width,z] for x in xs for z in zs2],[.70710678,side*.70710678,0,0],[.05,.05,.014],paint,'lower_side_body')
    panel([[-half[0]-.02,y,z] for y in ys for z in [-.66,-.62,-.58]],[.70710678,0,.70710678,0],[.045,.035,.018],paint*.7,'rear_bumper')
    radius=.34;wheel_z=-half[2]+radius+.012
    for xcenter in (-half[0]*.65,half[0]*.61):
        for side in (-1,1):
            points=[]
            for rad in np.arange(0,radius+.001,.035):
                for angle in np.linspace(0,2*np.pi,max(8,int(2*np.pi*rad/.035)),endpoint=False):points.append([xcenter+rad*np.cos(angle),side*width,wheel_z+rad*np.sin(angle)])
            panel(points,[.70710678,side*.70710678,0,0],[.03,.03,.03],[.012,.014,.018],'tire_shape_prior')
    out={k:np.concatenate([kept[k]]+[a[k] for a in chunks]) for k in kept}
    np.savez_compressed(m.ROOT/'initial_vehicle_prior.npz',**out);e.initial=out;e.vehicle=m.Vehicle(out).cuda();e.original_state=m.cpu(e.vehicle.state_dict())
    supported=m.ten(np.r_[observed_support[selected],np.zeros(sum(len(a['xyz']) for a in chunks),bool)],torch.bool)
    # Re-evaluate true visible contribution after appending the prior. Any prior
    # footprint that reaches a true observation is trainable rather than frozen.
    weights2=torch.ones(len(out['xyz']),device='cuda',requires_grad=True);previous_rows2=e.vehicle.rows
    def tagged_final(pose,cam):
        rows=previous_rows2(pose,cam);f=rows['features'].clone();f[:,2]=weights2
        return dict(rows,features=f)
    e.vehicle.rows=tagged_final;prior_visible_mass=0.;prior_visible_rows=torch.zeros(len(out['xyz']),device='cuda',dtype=torch.bool)
    for raw in m.TRAIN:
        out_render=e.render(raw);g=torch.autograd.grad(out_render['vehicle_alpha'][e.targets[raw]['mask']].sum(),weights2)[0];supported|=g>1e-8;prior_visible_rows|=g>1e-8;prior_visible_mass+=float(g[len(kept['xyz']):].sum())
    e.vehicle.rows=previous_rows2
    e.vehicle_fit_visible=supported.detach();e.vehicle.opacity_logits.register_hook(lambda g:g*e.vehicle_fit_visible);e.vehicle.log_scale_delta.register_hook(lambda g:g*e.vehicle_fit_visible[:,None])
    m.write(m.ROOT/'lower_prior.json',dict(schema='a20_box_conditioned_lower_shape_prior.v1',prior='lower rear/side panels, bumper and four tire disks; engineering shape prior, not observed completion',box_lwh=(half*2).tolist(),body_half_width_from_visible_support_m=width,paint_rgb_from_observed_dark_a20=paint.tolist(),native_rows_before=len(original['xyz']),native_inside_box=int(inside.sum()),native_outside_box_observed_support_retained=int((~inside&observed_support).sum()),native_outside_box_unobserved_retired=int((~selected).sum()),visible_support_method='actual common-renderer feature VJP over a20 true visible masks raw86/110; contribution>1e-8',native_hidden_colors_reinitialized=int(hidden.sum()),new_prior_gaussians=sum(len(a['xyz']) for a in chunks),components=counts,new_prior_opacity=.92,prior_local_geometry_frozen=True,hidden_prior_opacity_and_scale_frozen=True,opacity_scale_trainable_visible_rows=int(supported.sum()),prior_centers_inside_eroded_a9_occluder_mask_all5=True,prior_rejected_if_overlapping_observed_vehicle_masks=True,prior_bottom_above_box_base=True,new_prior_color_source='median real visible dark truck pixels, excludes 35% brightest; no road teacher or diffusion',true_hidden_shape_and_RGB_GT=False,total_rows=len(out['xyz'])))
    receipt=m.read(m.ROOT/'lower_prior.json');receipt.update(prior_occluder_mask_erosion='per-component/camera 4.5 projected sigma +2 pixels',prior_visible_contribution_mass_pixels=prior_visible_mass,prior_rows_with_real_visible_contribution=int(prior_visible_rows[len(kept['xyz']):].sum()),support_feature_VJP_calls=4,opacity_scale_trainable_visible_rows=int(supported.sum()));m.write(m.ROOT/'lower_prior.json',receipt)
    print('LOWER_PRIOR',int(inside.sum()),sum(len(a['xyz']) for a in chunks),paint.tolist(),flush=True)
