"""真实joint足迹资格；禁止迁入分量改变五帧保留车辆。"""
from native import *
from asset_io import build_asset
from seam_domain import seam_rows
from gaussian_patch_transport import covariance_from_sigma_quaternion
from scipy.spatial import cKDTree

def changed_pixel_footprints(ctx,raw,new_ids,opacity,scales,changed):
    """捕捉alpha early-stop跳过颜色累加的分量：只在实际变化K像素做保守足迹筛选。"""
    if not len(changed):return np.zeros(len(new_ids),bool)
    active_pos=np.searchsorted(ctx.active_world_rows,new_ids)
    sigma=(ctx.base_packet(raw)['scales'] if scales is None else scales)[active_pos].detach().cpu().numpy()/float(ctx.road['meter_to_model'])
    op=(ctx.base_packet(raw)['opacities'].reshape(-1) if opacity is None else opacity)[active_pos].detach().cpu().numpy()
    covariance=covariance_from_sigma_quaternion(sigma,ctx.road['gaussian_rotation_world_wxyz'][new_ids])
    camera=load(C/'road_common1152_20261121/maps'/f'{raw:03d}.npz');T=camera['world_to_camera'];K=camera['camera_K']
    world=ctx.provenance['world_xyz'][new_ids];xyz=world@T[:3,:3].T+T[:3,3]
    z=np.maximum(xyz[:,2],1e-6);projected=xyz@K.T;uv=projected[:,:2]/z[:,None]-.5
    J=np.zeros((len(z),2,3));J[:,0,0]=K[0,0]/z;J[:,0,2]=-K[0,0]*xyz[:,0]/z**2
    J[:,1,1]=K[1,1]/z;J[:,1,2]=-K[1,1]*xyz[:,1]/z**2;J=J@T[:3,:3]
    screen=J@covariance@J.transpose(0,2,1)+np.eye(2)*.3
    radius=4*np.sqrt(np.linalg.eigvalsh(screen)[:,-1])+2
    eligible=(xyz[:,2]>.1)&(op>=1/255)&np.isfinite(radius)&(radius<1e5)
    bad=np.zeros(len(z),bool);indices=np.flatnonzero(eligible)
    neighbours=cKDTree(changed[:,::-1]).query_ball_point(uv[indices],radius[indices])
    bad[indices]=np.fromiter((bool(x) for x in neighbours),bool,len(indices));return bad

def main():
    start=time.monotonic();n=len(load(PARENT/'road.npz')['active'])
    a=load(ROOT/'proposal/transferred_rows.npz');M=len(a['world_xyz']);edge_all=seam_rows(ROOT)['new_seam_mask']
    checkpoint=ROOT/'qualification/K_rejected_rows.npz'
    rejected=load(checkpoint)['reject_new_mask'] if checkpoint.exists() else np.zeros(M,bool)
    # 持久资产绑定绝对路径；旧失败现场移入工程归档后按相同数组在新路径重新pin。
    current=ROOT/'qualification'/f'peeling_resume_{int(rejected.sum()):06d}'/'asset'
    if not current.exists():current=build_asset(current,reject_new_mask=rejected)
    assert np.array_equal(load(current/'road.npz')['active'][n:],~rejected)
    history=[];totalF=0;totalB=0
    # 新块会互相遮挡；剔除前层后必须再次认证新显露的后层，不能只做一次VJP。
    for iteration in range(M+1):
        ctx=open_context(current);active=ctx.active_world_rows;new=active>=n;new_ids=active[new]
        pos=tensor(np.flatnonzero(new),torch.long);edgepos=pos[tensor(edge_all[new_ids-n],torch.bool)]
        colors=ctx.base_rgb.detach().clone().requires_grad_(True);found=np.zeros(M,bool);records=[];residual=[]
        try:
            packet=ctx.base_packet(FRAMES[0]);base_op=packet['opacities'].detach().reshape(-1);base_scale=packet['scales'].detach()
            states=[('initial',None,None)]
            for name,opvalue,factor in [('seam_parameter_max',.999,1.15),('seam_parameter_min',.0001,.9)]:
                op=base_op.index_copy(0,edgepos,torch.full((len(edgepos),),opvalue,device='cuda'))
                sigma=base_scale[edgepos].clone();sigma[:,:2]*=factor
                states.append((name,op,base_scale.index_copy(0,edgepos,sigma)))
            for state,op,scale in states:
                for raw in FRAMES:
                    out=render(ctx,raw,colors,op,scale,joint=True)
                    grad=torch.autograd.grad(out['rgb'][:,tensor(ctx.K_masks[raw],torch.bool)].sum(),colors)[0].abs().amax(1).cpu().numpy()
                    bad=grad[new]>0;found[new_ids[bad]-n]=True
                    parent=load(ROOT/'parent/rendered'/f'{raw:03d}.npz');K=ctx.K_masks[raw]
                    records.append(dict(state=state,raw=raw,new_K_contributing_rows=int(bad.sum()),
                        RGB_max=float(np.abs(out['rgb'].detach().cpu().numpy()-parent['joint_rgb'])[:,K].max()),
                        alpha_max=float(np.abs(out['alpha'].detach().cpu().numpy()-parent['joint_alpha'])[K].max())))
                    difference=np.max(np.abs(out['rgb'].detach().cpu().numpy()-parent['joint_rgb']),axis=0)>0
                    difference|=np.abs(out['alpha'].detach().cpu().numpy()-parent['joint_alpha'])>0
                    residual.append((raw,op,scale,np.argwhere(difference&K)))
            if not found.any() and any(len(x[3]) for x in residual):
                for raw,op,scale,changed in residual:
                    bad=changed_pixel_footprints(ctx,raw,new_ids,op,scale,changed)
                    found[new_ids[bad]-n]=True
                records.append(dict(state='actual_changed_K_pixel_footprint_veto',raw=None,new_K_contributing_rows=int(found.sum()),RGB_max=0.,alpha_max=0.))
            totalF+=15;totalB+=15
            history.append(dict(iteration=iteration,new_rejected=int((found&~rejected).sum()),records=records))
            write(ROOT/'qualification/peeling_progress.json',dict(history=history,nativeF=totalF,nativeB=totalB,optimizer_updates=0))
        finally:close()
        if not found.any():
            assert all(x['RGB_max']==0 and x['alpha_max']==0 for x in records),'无颜色足迹但K不一致，保留现场检查'
            break
        assert (found&~rejected).any(),'资格迭代没有进展'
        rejected|=found;np.savez_compressed(checkpoint,reject_new_mask=rejected)
        current=build_asset(ROOT/f'qualification/peeling_active_reject_{int(rejected.sum()):06d}/asset',reject_new_mask=rejected)
    else:raise RuntimeError('资格迭代没有收敛')
    np.savez_compressed(ROOT/'qualification/K_rejected_rows.npz',reject_new_mask=rejected)
    final=build_asset(ROOT/'transfer/asset',reject_new_mask=rejected);ctx=open_context(final);checks=[]
    try:
        with torch.no_grad():
            for raw in FRAMES:
                out=render(ctx,raw,joint=True);parent=load(ROOT/'parent/rendered'/f'{raw:03d}.npz');K=ctx.K_masks[raw]
                rgb=float(np.abs(out['rgb'].cpu().numpy()-parent['joint_rgb'])[:,K].max())
                alpha=float(np.abs(out['alpha'].cpu().numpy()-parent['joint_alpha'])[K].max())
                checks.append(dict(raw=raw,RGB_max=rgb,alpha_max=alpha));assert rgb==0 and alpha==0,checks[-1]
        export(ctx,ROOT/'transfer/rendered')
        ctx.validate_public_reload(ctx.base_rgb,final,ROOT/'transfer/public_reload.json')
        write(ROOT/'transfer/receipt.json',dict(complete=True,optimizer_updates=0,new_diffusion_calls=0,checks=checks,
            asset=str(final),rejected_new_rows=int(rejected.sum()),native_gate_F=totalF+25,native_gate_B=totalB,qualification_rounds=len(history),seconds=time.monotonic()-start,
            source_RGB_preserved=True,source_covariance_pushed_forward=True,source_block_retained=True))
        print(json.dumps(read(ROOT/'transfer/receipt.json')),flush=True)
    finally:close()
if __name__=='__main__':main()
