"""Authorized read-only synthetic ideal control: thirty native forwards, zero updates."""
from pathlib import Path
import json,sys,time,traceback
import numpy as np
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
O=Q/'gpu_20261119/ideal_read_probe'
sys.path.insert(0,str(Q/'gpu_20261119/native/code'))
from run_native_qualification import ctx_load,write,sha

def main():
    if (O/'native_started.json').exists():raise FileExistsError('Preserve previous native run')
    root_auth=Q/'gpu_20261119/authorization.json';auth=json.loads(root_auth.read_text())
    if auth.get('GPU_authorized') is not True or auth['lease']['native_ctx_owner']!='stage_qa':raise RuntimeError('No root authorization')
    lease=O/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner='stage_qa',root_authorization=str(root_auth),
        authorization_sha256=sha(root_auth),purpose='Root explicitly requested synthetic 12/24cm native read probe',max_native_forward_calls=30,optimizer_updates=0))
    start=time.time();write(O/'native_started.json',dict(time=start,optimizer_updates=0,max_native_forward_calls=30))
    ctx=None
    try:
        import torch
        torch.set_num_threads(8);ctx=ctx_load(lease)
        with np.load(O/'avg_rows.npz') as z:rgb=z['rgb'];mass=z['mass'];local=z['active_local_indices'];active=z['active_world_rows']
        if not np.array_equal(active,ctx.active_world_rows):raise RuntimeError('Canonical road identities changed')
        output=O/'rendered';output.mkdir(exist_ok=False);receipts=[]
        with torch.no_grad():
            colors=[];use=torch.as_tensor(local[mass>1e-12],device=ctx.base_rgb.device)
            for i in range(3):
                c=ctx.base_rgb.clone();c[use]=torch.as_tensor(rgb[i,mass>1e-12],device=c.device);colors.append(c)
            for raw in (86,110,134,179,198):
                road=[];joint=[];alpha=[];jointalpha=[]
                for c in colors:
                    if ctx.native_calls+2>30:raise RuntimeError('Native probe cap exceeded')
                    r=ctx.render_road(raw,c);j=ctx.render_joint(raw,c)
                    road.append(r['rgb'].cpu().numpy());joint.append(j['rgb'].cpu().numpy())
                    alpha.append(r['alpha'][0].cpu().numpy());jointalpha.append(j['alpha'][0].cpu().numpy())
                ae=max(float(np.max(np.abs(alpha[0]-a)))for a in alpha)
                jae=max(float(np.max(np.abs(jointalpha[0]-a)))for a in jointalpha)
                K=ctx.K_masks[raw];ke=max(float(np.max(np.abs(joint[0][:,K]-j[:,K])))for j in joint)
                if ae!=0 or jae!=0 or ke!=0:raise RuntimeError('Ideal probe changes fixed alpha or joint K')
                np.savez_compressed(output/f'{raw:03d}.npz',avg_road_rgb=np.stack(road),avg_joint_rgb=np.stack(joint),
                    road_alpha=alpha[0],joint_alpha=jointalpha[0],K=K)
                receipts.append(dict(raw=raw,alpha_max=ae,joint_alpha_max=jae,joint_K_RGB_max=ke,native_calls=6))
                print('IDEAL_NATIVE_READY',raw,ctx.native_calls,flush=True)
        ctx.assert_frozen()
        write(O/'native_completed.json',dict(completed=True,rows=receipts,native_forward_calls=ctx.native_calls,
            native_backward_calls=0,optimizer_updates=0,seconds=time.time()-start,geometry_opacity_scales_and_ledger_unchanged=True,
            scope=89887,source='Independent continuous world function sampled on fixed real 2cm source chart',
            no_claim='Synthetic read/filter control only; not actual teacher fit or production asset; no capacity upper-bound certificate'))
    except Exception:
        write(O/'native_failed.json',dict(error=traceback.format_exc(),native_calls=0 if ctx is None else ctx.native_calls,optimizer_updates=0));raise
    finally:
        if ctx is not None:ctx.close()
        a=json.loads(lease.read_text());a['active']=False;a['ended_time']=time.time();write(lease,a)

if __name__=='__main__':main()
