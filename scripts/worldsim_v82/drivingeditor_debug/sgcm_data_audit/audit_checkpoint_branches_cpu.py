"""CPU per-Gaussian geometry/appearance diagnostics; no rasterization or training."""
from audit_cpu import *

meta=json.loads((OUT/'sgcm_runtime_geometry.json').read_text())
transform=np.array(meta['world_to_model'])
firstpose=np.loadtxt(DATA/'pose/0017000.txt')
world_to_anchor=np.linalg.inv(firstpose)
model_to_anchor=world_to_anchor@np.linalg.inv(transform)
plane=np.array(meta['ground_fit']['plane_anchor'])
model_scale=meta['meter_to_model_scale']
shutils=load_module(SRC/'utils/sh_utils.py','frozen_sh_diagnostic')
initial_dynamic,_=read_ply(DATA/'points3d_dynamic_0.ply')
initial_dynamic_anchor=sg.transform_points(initial_dynamic,world_to_anchor)
initial_tree=cKDTree(initial_dynamic_anchor)
frame='0017010'
timestamp=-.28
K,tr,ego=calibration(frame)
world_to_camera=tr@np.linalg.inv(ego)
model_to_camera=world_to_camera@np.linalg.inv(transform)
camera_center_model=sg.transform_points((ego@np.linalg.inv(tr))[:3,3][None],transform)[0]
mask=np.asarray(Image.open(DATA/'dynamic_masks/dynamic_mask_0'/f'{frame}.png'))>0
if mask.ndim==3:mask=mask[...,0]
H,W=mask.shape


def branch_stats(path,branch):
    cp,it=torch.load(path,map_location='cpu')
    xyz=cp[1].detach().numpy()
    sigma=np.exp(cp[8].detach().numpy()[:,0])
    time=cp[7].detach().numpy()[:,0]
    velocity=cp[9].detach().numpy()
    marginal=np.exp(-.5*((timestamp-time)/sigma)**2)
    a=2*np.pi/cp[16]
    means=xyz+velocity*np.sin((timestamp-time)[:,None]*a)/a
    anchor=sg.transform_points(means,model_to_anchor)
    ground_signed=anchor@plane[:3]+plane[3]
    camera,uv,pix,inside=project(means,K,model_to_camera,W,H)
    live=inside&(marginal>.05)
    outside_mask=np.zeros(len(xyz),bool)
    inds=np.flatnonzero(inside)
    outside_mask[inds]=~mask[pix[inds,1],pix[inds,0]]
    roadproxy=live&(pix[:,1]>=H*.55)&(np.abs(ground_signed)<=.5)
    opacity=torch.sigmoid(cp[6]).detach().numpy()[:,0]
    scales=np.exp(cp[4].detach().numpy())/model_scale
    quat=cp[5].detach().numpy()
    sh_rgb=np.zeros((len(xyz),3),np.float32)
    for begin in range(0,len(xyz),262144):
        end=min(begin+262144,len(xyz))
        sh=torch.cat([cp[2][begin:end],cp[3][begin:end]],dim=1).detach().numpy().transpose(0,2,1)
        dirs=means[begin:end]-camera_center_model
        dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-15)
        sh_rgb[begin:end]=np.maximum(shutils.eval_sh(int(cp[0]),sh,dirs)+.5,0)
    color_spread=np.ptp(sh_rgb,axis=1)
    sum_moment=np.zeros(len(xyz),np.float64)
    for state in cp[14]['state'].values():
        mom=state['exp_avg'].detach().numpy()
        sum_moment+=np.abs(mom).reshape(len(xyz),-1).sum(axis=1)
    near_init,_=initial_tree.query(anchor,k=1,distance_upper_bound=.5,workers=1)
    # Identical coordinates expose exact repeated copies, without guessing lineage.
    coordinates,counts=np.unique(xyz,axis=0,return_counts=True)
    largest=np.argsort(counts)[-5:][::-1]
    groups=[]
    for uid in largest:
        ids=np.flatnonzero(np.all(xyz==coordinates[uid],axis=1))
        groups.append({'members':len(ids),'xyz_model':coordinates[uid].tolist(),
            'ground_center_distance_m':float(abs(sg.transform_points(coordinates[uid][None],model_to_anchor)[0]@plane[:3]+plane[3])),
            'features_DC_spread_max':float(np.ptp(cp[2].detach().numpy()[ids],axis=0).max()),
            'features_rest_spread_max':float(np.ptp(cp[3].detach().numpy()[ids],axis=0).max()),
            'scaling_log_spread_max':float(np.ptp(cp[4].detach().numpy()[ids],axis=0).max()),
            'velocity_spread_max':float(np.ptp(velocity[ids],axis=0).max()),
            'all_Adam_first_moments_zero_count':int((sum_moment[ids]==0).sum()),
            'projected_visible_at_raw10':int(live[ids].sum())})
    subsets={}
    for name,selected in [('all',np.ones(len(xyz),bool)),('projected_centers_live',live),
                           ('projected_lower_image_near_ground',roadproxy),
                           ('projected_live_outside_actor_mask',live&outside_mask)]:
        subsets[name]={'count':int(selected.sum()),
          'effective_opacity_quantiles':qsummary((opacity*marginal)[selected]),
          'max_spatial_sigma_quantiles_m':qsummary(scales.max(axis=1)[selected]),
          'max_color_quantiles':qsummary(sh_rgb.max(axis=1)[selected]),
          'color_channel_spread_quantiles':qsummary(color_spread[selected]),
          'color_any_channel_gt_1':int(((sh_rgb>1).any(axis=1)&selected).sum()),
          'color_channel_spread_gt_1':int(((color_spread>1)&selected).sum()),
          'all_Adam_first_moments_zero':int(((sum_moment==0)&selected).sum()),
          'distance_to_ground_quantiles_m':qsummary(np.abs(ground_signed[selected])),
          'centers_beyond_05m_of_initial_dynamic':int((~np.isfinite(near_init)&selected).sum())}
    result={'path':str(path),'branch':branch,'iteration':it,'points':len(xyz),
       'query_frame':frame,'query_model_time':timestamp,'active_SH_degree':int(cp[0]),
       'unique_exact_xyz':len(coordinates),'duplicated_xyz_extra_copies':int((counts-1).sum()),
       'largest_exact_xyz_group_size':int(counts.max()),'largest_exact_xyz_groups':groups,
       'quaternion_norm_quantiles':qsummary(np.linalg.norm(quat,axis=1)),
       'subsets':subsets,'cuda_initialized':torch.cuda.is_initialized(),
       'limits':'Projection of Gaussian centers and per-Gaussian SH colors is not rasterization or a pixel contribution attribution. Lower-image near-plane subset is a diagnostic proxy, not a semantic road label.'}
    print(json.dumps({'branch':branch,'iteration':it,'points':len(xyz),'largest_group':int(counts.max()),
                      'road_proxy':subsets['projected_lower_image_near_ground']},ensure_ascii=False),flush=True)
    save(f'branch_{branch}_{it}_cpu.json',result)
    del cp
    gc.collect()
    return result


if __name__=='__main__':
    paths=[('static',BASE/'runs/scene0017_000_048_paper_50000_seed0/chkpnt10000.pth'),
           ('static',RUN/'chkpnt20000.pth'),('dynamic',RUN/'dynamic_chkpnt20000.pth'),
           ('static',RUN/'chkpnt30000.pth'),('dynamic',RUN/'dynamic_chkpnt30000.pth')]
    results=[branch_stats(path,branch) for branch,path in paths]
    save('checkpoint_branch_summary.json',results)
    assert not torch.cuda.is_initialized()
    print('BRANCH_CPU_AUDIT_COMPLETE',flush=True)
