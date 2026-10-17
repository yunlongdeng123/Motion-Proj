"""R15-A37：只有实测参考的静态点外观拟合，固定几何/owner/opacity/光照。

这是局部强控制，不能把137点的参考误差写成整辆车残影正确率。
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from packed_shadow import install


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--audit',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();sys.path.insert(0,a.source)
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.math3d import srgb_to_linear,linear_to_srgb,quaternion_matrix,normalize
    from shadow_rebuild.legacy.sh_utils import eval_sh
    from shadow_rebuild.losses import pixel_valid
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    refs=dict(np.load(Path(a.audit)/'a37_static_reference.npz'));rows=torch.as_tensor(refs['rows'],device='cuda')
    result=dict(run_id='R15-A37-APPEARANCE-CONTROL',status='RUNNING',component_optimizer_updates=0,
        global_training_updates=0,reference_points=len(rows),frames=[],geometry_owner_opacity_light_frozen=True,
        scope='sparse reference-supported appearance only; not full Q01 global optimization pilot')
    def save():
        p=out/'appearance_control.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(p)
    if not len(rows):
        result.update(status='ABSTAIN_NO_REFERENCE');save();return
    try:
        base=Path(a.base);saved=torch.load(base/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
        raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
        data=Dataset(base/'data','cuda',True);assert saved['input_contract']==data.manifest
        model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
        model.set_stage('geometry');model.requires_grad_(False);install(model,dtype=torch.float64)
        field=model.field;assert bool((field.owner[rows]==-1).all())
        anchors={101,105,113,121,137};before={}
        def png(path,rgb):
            Image.fromarray((linear_to_srgb(rgb).clamp(0,1)*255).round().byte().cpu().numpy()).save(path)
        def render(frame,delete=False,static=False):
            edits={j:{'delete':True} for j in range(len(data.actor_ids))} if static else ({37:{'delete':True}} if delete else None)
            g,p=model.buffers(frame,len(data.actor_ids),'cuda',edits,False)
            old=cfg.shadows
            if delete or static:cfg.shadows=False
            image=model.train_image(p,g)
            cfg.shadows=old
            return image,p
        # 只测量；QUERY从不进入拟合目标和优化选择。
        with torch.no_grad():
            for i,row in enumerate(data.rows):
                f=data.frame(i);image,p=render(f);deleted,_=render(f,True)
                before[i]=(image.cpu().numpy(),deleted.cpu().numpy())
                raw_id=int(f.frame_id)%1000
                if raw_id in anchors:
                    png(out/f'raw{raw_id}_before_full.png',image);png(out/f'raw{raw_id}_before_delete37.png',deleted)
                    static,_=render(f,static=True);png(out/f'raw{raw_id}_before_static.png',static)
                del image,p,deleted
        albedo=torch.nn.Parameter(field.albedo_logit[rows].detach().clone())
        dc=torch.nn.Parameter(field.features_dc[rows].detach().clone())
        rest=torch.nn.Parameter(field.features_rest[rows].detach().clone())
        road=field.road[rows]
        normal=quaternion_matrix(field.quaternion[rows])[...,2]
        irradiance=model.light.ambient_logit.sigmoid()+(1-model.light.ambient_logit.sigmoid())*(normal*model.light.direction()).sum(-1).clamp_min(0)[:,None]
        direction=torch.as_tensor(refs['view_direction'],device='cuda')
        target=srgb_to_linear(torch.as_tensor(refs['target_srgb'],device='cuda'))
        weight=torch.as_tensor(refs['confidence'],device='cuda')[:,None]*torch.as_tensor(refs['observed'],device='cuda')
        def objective():
            sh=torch.cat((dc,rest),1).transpose(1,2)
            view=(sh[:,None].expand(-1,direction.shape[1],-1,-1).reshape(-1,3,16))
            color=srgb_to_linear((eval_sh(int(field.active_sh_degree),view,direction.reshape(-1,3))+.5).clamp_min(0)).reshape(len(rows),-1,3)
            predict=torch.where(road[:,None,None],(albedo.sigmoid()*irradiance)[:,None,:],color)
            return ((predict-target[:,None,:]).square().mean(-1)*weight).sum()/weight.sum().clamp_min(1e-8)
        optimizer=torch.optim.Adam([albedo,dc,rest],lr=cfg.appearance_lr)
        before_loss=float(objective().detach());history=[]
        for step in range(1,201):
            optimizer.zero_grad(set_to_none=True);loss=objective();loss.backward();optimizer.step()
            if step%50==0:history.append(dict(component_step=step,reference_MSE=float(objective().detach())))
        with torch.no_grad():
            field.albedo_logit[rows]=albedo;field.features_dc[rows]=dc;field.features_rest[rows]=rest
        after_loss=float(objective().detach())
        result.update(component_optimizer_updates=200,static_road_points=int(road.sum()),static_nonroad_points=int((~road).sum()),
            reference_MSE_before=before_loss,reference_MSE_after=after_loss,
            reference_MSE_reduction=1-after_loss/max(before_loss,1e-12),fit_history=history)
        np.savez_compressed(out/'appearance_patch.npz',rows=refs['rows'],albedo_logit=albedo.detach().cpu().numpy(),
                            features_dc=dc.detach().cpu().numpy(),features_rest=rest.detach().cpu().numpy())
        with torch.no_grad():
            for i,row in enumerate(data.rows):
                f=data.frame(i);image,p=render(f);deleted,_=render(f,True)
                b=torch.as_tensor(before[i][0],device='cuda');bd=torch.as_tensor(before[i][1],device='cuda')
                error=(image-f.rgb).abs();old_error=(b-f.rgb).abs();body=f.masks&pixel_valid(f)
                metrics=dict(frame_id=f.frame_id,split=f.split,
                    PSNR_before=float(-10*torch.log10((b-f.rgb).square().mean().clamp_min(1e-12))),
                    PSNR_after=float(-10*torch.log10((image-f.rgb).square().mean().clamp_min(1e-12))),
                    MAE_before=float(old_error.mean()),MAE_after=float(error.mean()),body=[])
                for actor in [5,6,12,18,25,26,37]:
                    mask=body[...,actor];denom=int(mask.sum())
                    metrics['body'].append(dict(actor=actor,pixels=denom,
                        MAE_before=float(old_error[mask].mean()) if denom else None,
                        MAE_after=float(error[mask].mean()) if denom else None))
                mask=body[...,37];delta=(deleted-bd).abs().mean(-1)
                metrics['delete37_changed_body_pixels_gt_1_over_255']=int(((delta>1/255)&mask).sum())
                metrics['body37_pixels']=int(mask.sum());metrics['delete37_body_mean_absolute_change']=float(delta[mask].mean()) if mask.any() else None
                result['frames'].append(metrics)
                raw_id=int(f.frame_id)%1000
                if raw_id in anchors:
                    png(out/f'raw{raw_id}_after_full.png',image);png(out/f'raw{raw_id}_after_delete37.png',deleted)
                    static,_=render(f,static=True);png(out/f'raw{raw_id}_after_static.png',static)
                if (i+1)%10==0:save()
        result.update(status='COMPLETE_SPARSE_CONTROL_ONLY',coverage_interpretation='137 certified reference points /71746 static delete candidates; not regional clean-road truth',
            global_a37_ghost_success='NOT_ESTABLISHED',geometry_owner_opacity_changed=False,original_checkpoint_modified=False,
            failure_ledger_refs=['V82-F06'],human_verdict=None)
        save();print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
    except BaseException as e:
        result.update(status='ERROR',error=repr(e));save();raise


if __name__=='__main__':main()
