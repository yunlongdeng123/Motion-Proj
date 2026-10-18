"""Frozen repaired4K true-owner 49-frame independent/delete evaluation."""
from pathlib import Path
import sys,time,json
Q=Path(__file__).resolve().parent
sys.path.insert(0,str(Q/'implementation'))
TASK=Q.parent.parent.parent/'identity_completion_20261018'
sys.path.insert(0,str(TASK/'code'))
from identity_common import *
from render_identity_matrix import save_png
def run_r15(args,out):
    import torch
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import render_buffers,cuda_raster
    from shadow_rebuild.factor_shadow import factors_for_geometry
    ck=Q/'train/checkpoint_004000.pt';before=stat(ck)
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);model=SceneModel(seed_from_state(saved['state']),cfg).cuda()
    model.load_state_dict(saved['state']);model.set_stage('geometry');model.requires_grad_(False)
    assert cfg.sun_samples==5 and cfg.shadow_integrator=='factor_fp64' and saved['global_step']==4000
    records=[]
    with torch.no_grad():
        for i in range(len(data.rows)):
            frame=data.frame(i);fid=frame.frame_id;folder=out/fid;folder.mkdir()
            g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
            cfg.shadows=False;full_off=model.train_image(p,g)
            cfg.shadows=True;full_on=model.train_image(p,g)
            save_png(folder/'gt.png',frame.rgb,True);save_png(folder/'full_no_shadow.png',full_off,True)
            save_png(folder/'full.png',full_on,True)
            for actor in ACTORS:
                take=g['owner']==actor
                feat=torch.cat((torch.ones_like(g['opacity']),g['appearance']),-1)
                r=cuda_raster(frame.camera,g['xyz'][take],g['cov'][take],g['opacity'][take],feat[take],10)
                keep=~take;dg=type(g)(**{k:v[keep] for k,v in g.items()})
                assert not (dg['owner']==actor).any()
                for key in ['rows','xyz','cov','opacity','appearance']:assert torch.equal(dg[key],g[key][keep])
                # 删除的同一几何同时从主相机和太阳caster集合消失，不能复用父缓存。
                dg.shadow_factors=factors_for_geometry(model,frame,dg)
                assert not (dg.shadow_factors.owners==actor).any()
                dp=render_buffers(frame.camera,dg,len(data.actor_ids),'cuda',cfg.max_features,isolated=False)
                dp['sky']=p['sky'];dp['canonical']=dp['appearance']+(1-dp['alpha'])[...,None]*p['sky']
                assert dp['visible_labels'][...,actor].count_nonzero()==0
                cfg.shadows=False;delete_off=model.train_image(dp,dg)
                cfg.shadows=True;delete_on=model.train_image(dp,dg)
                save_png(folder/f'a{actor:02d}_isolated.png',r[...,1:4],True)
                save_png(folder/f'a{actor:02d}_alpha.png',r[...,0])
                save_png(folder/f'a{actor:02d}_visible.png',p['visible_labels'][...,actor])
                save_png(folder/f'a{actor:02d}_mask.png',frame.masks[...,actor].float())
                save_png(folder/f'a{actor:02d}_delete.png',delete_on,True)
                save_png(folder/f'a{actor:02d}_delete_no_shadow.png',delete_off,True)
                np.savez_compressed(folder/f'a{actor:02d}.npz',isolated_rgb=r[...,1:4].cpu().numpy(),
                    isolated_alpha=r[...,0].cpu().numpy(),visible_alpha=p['visible_labels'][...,actor].cpu().numpy(),
                    delete_rgb=delete_on.cpu().numpy(),delete_no_shadow_rgb=delete_off.cpu().numpy())
                records.append(dict(frame_id=fid,actor=actor,track_id=data.actor_ids[actor],
                    model_native_RGB_domain='linear RGB',delete_target_visible_alpha_exact_zero=True,
                    target_solar_casters_removed=True,other_geometry_bitwise_unchanged=True,
                    isolated_points=int(take.sum()),isolated_render_mask_multiplied=False,
                    known_body_pixels=int((frame.masks[...,actor]&frame.mask_valid[...,actor]).sum()),
                    known_negative_pixels=int((~frame.masks[...,actor]&frame.mask_valid[...,actor]).sum()),
                    visible_alpha_in_body=float(p['visible_labels'][...,actor][frame.masks[...,actor]&frame.mask_valid[...,actor]].sum()),
                    visible_alpha_in_known_negative=float(p['visible_labels'][...,actor][~frame.masks[...,actor]&frame.mask_valid[...,actor]].sum()),
                    rgb_mae_in_body=float((full_on-frame.rgb).abs()[frame.masks[...,actor]].mean()) if frame.masks[...,actor].any() else None))
                del dg,dp,r
            if i%7==0:print(json.dumps(dict(model='r15',frames=i+1)),flush=True)
    assert before==stat(ck)
    return dict(checkpoint=before,checkpoint_step=4000,actors=ACTORS,records=records,
        actual_ID_ownership=True,shadow_implementation='integrated validated FP64 factor kernel; freshly trained repair4K',
        physical_shadow_module=True,sun_samples=5)

def main():
    sys.path.insert(0,str(TASK/'infra'))
    from priority import require_gpu_slot
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    import torch
    torch.set_num_threads(2);out=Q/'renders_repair4k';out.mkdir(exist_ok=False)
    try:
        start=time.monotonic();r=run_r15(None,out)
        r.update(status='COMPLETE_REVIEW_PENDING',model='r15_repair4k',frames=49,parameter_updates=0,
            checkpoint_file_unchanged=True,seconds=time.monotonic()-start,admission=admission,
            human_verdict=None,paired_edited_GT_available=False)
        write(out/'render_receipt.json',r)
        saved=torch.load(Q/'train/checkpoint_004000.pt',map_location='cpu',weights_only=False)
        old=torch.load(R15/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
        from shadow_rebuild.data import Dataset
        data=Dataset(R15/'data','cpu',True)
        half=torch.tensor(read(R15/'box_half_sizes.json')['half_sizes_m']);stats={}
        for label,state in [('original4k',old['state']),('repair4k',saved['state'])]:
            records=[]
            for a in ACTORS:
                take=state['field.owner']==a;x=state['field.xyz'][take];status=state['field.identity_evidence_status'][take]
                seed=data.seed['xyz'][data.seed['owner']==a]
                records.append(dict(actor=a,initial_points=len(seed),points=len(x),growth=len(x)/len(seed),
                    outside_exact_box=int((x.abs()>half[a]).any(-1).sum()),
                    outside_padded_box=int((x.abs()>half[a]+.1).any(-1).sum()),
                    raw_status_counts={str(k):int((status==k).sum()) for k in [0,1,2]},
                    maximum_sigma_quantile=torch.quantile(state['field.log_scales'][take].exp().max(-1).values,torch.tensor([.5,.95,.99,1.])).tolist()))
            stats[label]=records
        write(Q/'geometry_comparison.json',stats)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),assets_preserved=True));raise
    finally:slot.close()

if __name__=='__main__':main()
