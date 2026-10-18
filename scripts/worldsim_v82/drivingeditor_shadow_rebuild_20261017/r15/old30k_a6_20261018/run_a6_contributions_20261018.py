"""Exact full-scene alpha*T attribution for the same a6 observed body pixels."""
from run_a6_direct_20261018 import *

def main():
    dest=OUT/'a6_contributors'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir();torch.set_num_threads(2);start=time.monotonic()
    slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text())
    saved=torch.load(CK,map_location='cuda',weights_only=False)
    args,scene,static,instances,opt,env,exclusion=load_models(spec,dest,restore=saved)
    for group in static.optimizer.param_groups:
        for p in group['params']:p.requires_grad_(False)
    instances.requires_grad_(False);env.requires_grad_(False)
    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.raster import cuda_raster,Camera
    from shadow_rebuild.math3d import linear_to_srgb,srgb_to_linear
    data=Dataset(R15/'data','cuda',True)
    current=torch.load(R15/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    half=current['state']['field.actor_box_half'][6].cuda()
    old_bundle=np.load(Path(spec['prepared'])/'instances.npz',allow_pickle=False)
    n64=old_bundle['normalization'];s=float(old_bundle['world_scale'])
    rot=torch.tensor(n64[:3,:3]/s,device='cuda',dtype=torch.float32)
    o64=np.loadtxt(Path(spec['source_scene'])/'pose/0008089.txt')[:3,3]
    shift=torch.tensor(n64[:3,:3]@o64+n64[:3,3],device='cuda',dtype=torch.float32)
    factor=(quaternion_to_matrix(static.get_rotation)*static.get_scaling[:,None,:])
    factor_phys=(rot.T[None]@factor)/s;cov_phys=factor_phys@factor_phys.transpose(-1,-2)
    records=[]
    for i,camera in enumerate(scene.getTrainCameras()):
        raw=int(Path(camera.image_name).stem[-3:])
        if raw not in [105,113,137]:continue
        frame=data.frame(i);body=frame.masks[...,6]&frame.mask_valid[...,6]
        sf=torch.ones(len(static.get_xyz),1,device='cuda',requires_grad=True)
        df=torch.ones(len(instances.local_xyz),1,device='cuda',requires_grad=True)
        p=render(camera,static,instances,args,torch.zeros(3,device='cuda'),env_map=env,
            other=sf,dynamic_other=df,instance_ids=[],render_separate=False)
        cs,cd=torch.autograd.grad((p['feature'][0]*body).sum(),(sf,df))
        cs=cs[:,0].detach();cd=cd[:,0].detach()
        total=float(cs.sum()+cd.sum());static_mass=float(cs.sum());dynamic_mass=float(cd.sum())
        assert abs(total-float(p['alpha'][0][body].sum()))<1e-3
        with torch.no_grad():
            xyz_old=static.get_xyz_SHM(camera.timestamp)
            xyz_phys=((xyz_old-shift)@rot)/s;pose=frame.poses[6]
            local=(xyz_phys-pose[:3,3])@pose[:3,:3]
            core=(local.abs()<=half).all(-1);halo=(local.abs()<=half+.75).all(-1)
            values,order=torch.sort(cs,descending=True)
            cumulative=values.cumsum(0)
            cut=int(torch.searchsorted(cumulative,cumulative[-1]*.98))+1
            selected=torch.zeros_like(core);selected[order[:cut]]=True
            rgb,alpha=raster_old_subset(camera,static,args,selected)
            png(dest/f'raw{raw}_top98_rgb.png',rgb);png(dest/f'raw{raw}_top98_alpha.png',alpha)
            # Both camera-center and projected a6-center agreement are checked.
            center_old=(rot@pose[:3,3])*s+shift
            clip=torch.cat((center_old,center_old.new_ones(1)))@camera.full_proj_transform
            pixel_old=((clip[:2]/clip[3]+1)*clip.new_tensor([camera.image_width,camera.image_height])-1)/2
            cv=(pose[:3,3]-frame.camera.center)@frame.camera.pose[:3,:3]
            uvw=frame.camera.K@cv;pixel_new=uvw[:2]/uvw[2]
            uv_error=float((pixel_old-pixel_new).abs().max())
            # Original 3DGS uses pixel-center (-.5) conversion.
            assert uv_error<1.1,(pixel_old,pixel_new)
            scales=static.get_scaling.max(-1).values/s
            row={'raw':raw,'body_pixels':int(body.sum()),'static_mass':static_mass,'dynamic_mass':dynamic_mass,
                'total_alpha_mass':total,'static_fraction':static_mass/total,
                'static_alpha_mass_fraction_center_inside_box':float(cs[core].sum())/static_mass,
                'static_alpha_mass_fraction_center_inside_halo075m':float(cs[halo].sum())/static_mass,
                'static_alpha_mass_fraction_max_principal_scale_gt1m':float(cs[scales>1].sum())/static_mass,
                'top98_static_point_count':cut,'top98_static_mass_fraction':float(cs[selected].sum())/static_mass,
                'top98_max_principal_scale_p50_p95_p99_m':torch.quantile(scales[selected],scales.new_tensor([.5,.95,.99])).tolist(),
                'old_projected_a6_center_xy':pixel_old.tolist(),'r15_projected_a6_center_xy':pixel_new.tolist(),
                'projection_max_error_pixels_including_halfpixel':uv_error,
                'selection_is_body_pixel_contribution_not_a6_owner':True}
            np.savez_compressed(dest/f'raw{raw}_top98_rows.npz',rows=order[:cut].cpu().numpy(),
                contribution=cs[selected].cpu().numpy(),local_xyz=local[selected].cpu().numpy())
            old_rgb=prepare_static(camera,static,args)['colors_precomp']
            opacity=static.get_opacity*static.get_marginal_t(camera.timestamp)
            for label,eye_local in [('front',[9,0,3]),('rear',[-9,0,3]),('side',[0,-9,3])]:
                eye=pose[:3,:3]@pose.new_tensor(eye_local)+pose[:3,3]
                forward=torch.nn.functional.normalize(pose[:3,3]-eye,dim=0)
                right=torch.nn.functional.normalize(torch.cross(forward,pose[:3,2],dim=0),dim=0)
                down=torch.cross(forward,right,dim=0)
                c2w=torch.eye(4,device='cuda');c2w[:3,:3]=torch.stack((right,down,forward),-1);c2w[:3,3]=eye
                view=Camera([[400.,0.,240.],[0.,400.,160.],[0.,0.,1.]],c2w.cpu().numpy(),320,480,'cuda')
                f=torch.cat((torch.ones_like(opacity[selected]),srgb_to_linear(old_rgb[selected].clamp(0,1))),-1)
                r=cuda_raster(view,xyz_phys[selected],cov_phys[selected],opacity[selected],f,10)
                png(dest/f'raw{raw}_{label}_top98_rgb.png',linear_to_srgb(r[...,1:4]))
                png(dest/f'raw{raw}_{label}_top98_alpha.png',r[...,0])
            records.append(row);print(json.dumps(row),flush=True)
        del p,sf,df,cs,cd
    result={'schema':'a6_exact_old_fullscene_contributions_v1','status':'COMPLETE','actor':6,
        'track_id':data.actor_ids[6],'parameter_updates':0,'records':records,
        'selection':'smallest static point set accounting for >=98% full-scene alpha*T in unchanged a6 body pixels',
        'semantics':'visible contributors are not certified ownership; may include road, occluders and background',
        'novel_view_color_limit':'original-view SH colors held fixed; alpha/geometry use shared physical cameras',
        'seconds':time.monotonic()-start}
    write(dest/'contributions.json',result);slot.close()

if __name__=='__main__':main()
