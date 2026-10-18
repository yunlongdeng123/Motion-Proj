"""Same physical a6 in old baseline and R15. OBB selections are diagnostics only."""
from run_fragment_audit_20261018 import *
from renderer import prepare_static, _screenspace, _rasterize, _payload
from rigid_model import quaternion_to_matrix

def raster_old_subset(camera,static,config,mask):
    d=prepare_static(camera,static,config,mask=mask)
    # Explicitly subset the same native Gaussian rows; no mask-image multiplication.
    keep=d['mask'];d={k:v[keep] if torch.is_tensor(v) else v for k,v in d.items()}
    d['means2D']=_screenspace(d['means3D'])
    settings=RasterizationSettings(int(camera.image_height),int(camera.image_width),
        math.tan(-.5 if getattr(config,'neg_fov',False) else camera.FoVx*.5),
        math.tan(-.5 if getattr(config,'neg_fov',False) else camera.FoVy*.5),
        torch.zeros(3,device='cuda'),1.,camera.world_view_transform,camera.full_proj_transform,
        int(static.active_sh_degree),camera.camera_center,False,False)
    from diff_gaussian_rasterization import GaussianRasterizer
    r=GaussianRasterizer(raster_settings=settings)
    result=_rasterize(r,d,_payload(d,camera,d['means3D'].new_zeros(len(d['means3D'])),
        torch.full((len(d['means3D']),),-1,device='cuda',dtype=torch.long),[]),settings,10)
    return result[1],result[2][1]

def main():
    dest=OUT/'a6_direct'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir()
    slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    torch.set_num_threads(2);start=time.monotonic()
    spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text())
    saved=torch.load(CK,map_location='cuda',weights_only=False)
    args,scene,static,instances,opt,env,exclusion=load_models(spec,dest,restore=saved)
    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import cuda_raster,Camera
    from shadow_rebuild.math3d import linear_to_srgb,srgb_to_linear
    current=torch.load(R15/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    raw=current['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);model=SceneModel(seed_from_state(current['state']),cfg).cuda()
    model.load_state_dict(current['state']);model.requires_grad_(False)
    assert data.actor_ids[6]=='7MxwK9zTRZ9g5lc_ZC0z6A'
    half=model.field.actor_box_half[6]
    old_bundle=np.load(Path(spec['prepared'])/'instances.npz',allow_pickle=False)
    N=torch.tensor(old_bundle['normalization'],device='cuda',dtype=torch.float32)
    s=float(old_bundle['world_scale']);rot=N[:3,:3]/s
    origin=torch.tensor(np.loadtxt(Path(spec['source_scene'])/'pose/0008089.txt')[:3,3],device='cuda',dtype=torch.float32)
    # Avoid subtracting large absolute world coordinates in FP32. R15 translation
    # removes world_origin only; old normalization is a similarity of absolute world.
    n64=old_bundle['normalization'];o64=np.loadtxt(Path(spec['source_scene'])/'pose/0008089.txt')[:3,3]
    physical_to_old_shift=torch.tensor(n64[:3,:3]@o64+n64[:3,3],device='cuda',dtype=torch.float32)
    covrot=quaternion_to_matrix(static.get_rotation)
    factor=(covrot*static.get_scaling[:,None,:])
    factor_phys=(rot.T[None]@factor)/s
    cov_phys=factor_phys@factor_phys.transpose(-1,-2)
    records=[];errors=[];views=[]
    old_centers=static.get_xyz
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i,camera in enumerate(scene.getTrainCameras()):
            frame=data.frame(i);fid=frame.frame_id;raw=int(fid[-3:]);folder=dest/fid;folder.mkdir()
            assert fid==Path(camera.image_name).stem
            center_expected=(rot@frame.camera.center)*s+physical_to_old_shift
            camera_error=float(torch.linalg.vector_norm(center_expected-camera.camera_center)/s)
            errors.append(camera_error)
            assert camera_error<.002,(fid,camera_error)
            xyz_old=static.get_xyz_SHM(camera.timestamp)
            xyz_phys=((xyz_old-physical_to_old_shift)@rot)/s
            pose=frame.poses[6]
            local=(xyz_phys-pose[:3,3])@pose[:3,:3]
            local_cov=pose[:3,:3].T[None]@cov_phys@pose[:3,:3][None]
            local_std=local_cov.diagonal(dim1=-2,dim2=-1).clamp_min(0).sqrt()
            active=static.get_marginal_t(camera.timestamp)[:,0]>.05
            masks={'center_box':(local.abs()<=half).all(-1)&active,
                'center_halo075m':(local.abs()<=half+.75).all(-1)&active,
                'support3sigma':(local.abs()<=half+3*local_std).all(-1)&active}
            row={'raw':raw,'body_pixels':int(frame.masks[...,6].sum()),
                'selection_is_not_owner_ground_truth':True,'old_selection_counts':{k:int(v.sum()) for k,v in masks.items()}}
            for label,m in masks.items():
                rgb,alpha=raster_old_subset(camera,static,args,m)
                png(folder/(label+'_rgb.png'),rgb);png(folder/(label+'_alpha.png'),alpha)
                # Native observed view, exact original camera/covariance/SH path.
                body=frame.masks[...,6];row[label+'_alpha_in_body_mean']=float(alpha[body].mean()) if body.any() else None
            if raw in [105,113,137]:
                g=model.field.evaluate(frame.poses,frame.valid,frame.camera.center)
                take=g['owner']==6
                chosen=masks['center_halo075m']
                old_rgb=prepare_static(camera,static,args)['colors_precomp']
                opacity=(static.get_opacity*static.get_marginal_t(camera.timestamp))
                # Shared physical camera and rasterizer inspect 3D support. Old
                # colors are original-view SH held fixed; alpha is view-correct.
                for label,eye_local in [('front',[9,0,3]),('rear',[-9,0,3]),('side',[0,-9,3])]:
                    eye=pose[:3,:3]@pose.new_tensor(eye_local)+pose[:3,3]
                    forward=torch.nn.functional.normalize(pose[:3,3]-eye,dim=0)
                    right=torch.nn.functional.normalize(torch.cross(forward,pose[:3,2],dim=0),dim=0)
                    down=torch.cross(forward,right,dim=0)
                    c2w=torch.eye(4,device='cuda');c2w[:3,:3]=torch.stack((right,down,forward),-1);c2w[:3,3]=eye
                    view=Camera([[400.,0.,240.],[0.,400.,160.],[0.,0.,1.]],c2w.cpu().numpy(),320,480,'cuda')
                    for group,xyz,cov,op,rgbv in [
                        ('old_halo075m',xyz_phys[chosen],cov_phys[chosen],opacity[chosen],srgb_to_linear(old_rgb[chosen].clamp(0,1))),
                        ('r15_owner6',g['xyz'][take],g['cov'][take],g['opacity'][take],g['appearance'][take])]:
                        f=torch.cat((torch.ones_like(op),rgbv),-1)
                        r=cuda_raster(view,xyz,cov,op,f,10)
                        png(dest/f'raw{raw}_{label}_{group}_rgb.png',linear_to_srgb(r[...,1:4]))
                        png(dest/f'raw{raw}_{label}_{group}_alpha.png',r[...,0])
                    views.append({'raw':raw,'view':label,'old_selection':'center_halo075m','current_selection':'all_owner6',
                        'old_selection_is_not_certified_a6_asset':True,'old_original_view_SH_held_fixed':True})
            records.append(row)
            if i%7==0:print(json.dumps({'a6_direct_frames':i+1,'seconds':time.monotonic()-start}),flush=True)
    result={'schema':'a6_same_track_old30k_vs_r15_diagnostic_v1','status':'COMPLETE',
        'track_id':data.actor_ids[6],'actor':6,'frames':49,'parameter_updates':0,
        'old_independent_owner6':False,'old_branch':'static_PVG_with_time_and_velocity_freedom',
        'old_checkpoint':snap(CK),'current_checkpoint':snap(R15/'paused_checkpoint_004000_20261017/checkpoint.pt'),
        'box_half_m':half.tolist(),'old_world_scale':s,'max_camera_center_alignment_error_m':max(errors),
        'selection_methods':{'center_box':'native Gaussian centers inside same physical a6 OBB',
            'center_halo075m':'same OBB plus 0.75m on every face; no height or opacity cleanup',
            'support3sigma':'Gaussian 3sigma axis support intersects same a6 OBB; includes surrounding geometry'},
        'records':records,'novel_views':views,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'seconds':time.monotonic()-start,'limits':['old 3D subsets are extraction diagnostics, not existing a6 ownership',
          'box intersections may include road or adjacent occluder; absence after selection is not asset-cleanliness proof',
          'same49 observed views, no paired edited GT; 4k vs 30k is not matched training budget']}
    write(dest/'a6_direct.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','novel_views']},ensure_ascii=False),flush=True)
    slot.close()

if __name__=='__main__':main()
