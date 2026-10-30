"""Fixed-budget original-RGB-only local DC/opacity control, geometry frozen."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def refit(rows, selected, camera, pipe, helpers, degree, env, output, observation):
    if rows['shs'] is None or rows['colors_precomp'] is not None:
        raise ValueError('This control requires native SH representation')
    # The exact M1 selection plus all new split background children.
    indices=torch.where(selected)[0]
    geometry=np.load(Path(output)/'repaired_geometry_000008.npz')
    static_count=len(geometry['selected'])
    nchildren=len(geometry['split_parent_indices'])
    if nchildren:
        indices=torch.cat([indices,torch.arange(static_count-nchildren,static_count,device=selected.device)])
    dc=torch.nn.Parameter(rows['shs'][indices,0,:].clone())
    logit=torch.nn.Parameter(torch.logit(rows['opacities'][indices].clamp(1e-6,1-1e-6)))
    optimizer=torch.optim.Adam([dict(params=[dc],lr=.0025),dict(params=[logit],lr=.05)],eps=1e-8)
    target=torch.tensor(np.asarray(Image.open(observation).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255,device=dc.device)
    bg=torch.zeros(3,device=dc.device)
    losses=[]
    def materialize():
        result=dict(rows)
        shs=rows['shs'].clone()
        shs[:,0,:]=shs[:,0,:].index_copy(0,indices,dc)
        result['shs']=shs
        result['opacities']=rows['opacities'].index_copy(0,indices,logit.sigmoid())
        return result
    with torch.enable_grad():
        for step in range(200):
            optimizer.zero_grad(set_to_none=True)
            package=helpers.raster(camera,pipe,degree,materialize(),bg)
            prediction=package['rgb']+(1-package['alpha'])*env
            loss=(prediction-target).abs().mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite local refit')
            loss.backward(); optimizer.step()
            if step in (0,49,99,149,199): losses.append(dict(step=step+1,l1=float(loss.detach())))
    with torch.no_grad():
        before=materialize()
        p=helpers.raster(camera,pipe,degree,before,bg)
        before_rgb=p['rgb']+(1-p['alpha'])*env
        checkpoint=Path(output)/'local_refit_state.pt'
        torch.save(dict(indices=indices.cpu(),dc=dc.detach().cpu(),opacity_logit=logit.detach().cpu(),optimizer=optimizer.state_dict(),updates=200),checkpoint)
        saved=torch.load(checkpoint,map_location=dc.device,weights_only=False)
        dc.copy_(saved['dc']); logit.copy_(saved['opacity_logit'])
        optimizer.load_state_dict(saved['optimizer'])
        result=materialize()
        p=helpers.raster(camera,pipe,degree,result,bg)
        restored_rgb=p['rgb']+(1-p['alpha'])*env
        error=float((restored_rgb-before_rgb).abs().max())
        if error>2e-6: raise RuntimeError('Local refit save/reload render parity failed')
        arrays={k:geometry[k] for k in geometry.files}
        arrays['opacities']=result['opacities'][:static_count].cpu().numpy()
        arrays['refit_indices']=indices.cpu().numpy(); arrays['refit_dc']=dc.detach().cpu().numpy()
        np.savez_compressed(Path(output)/'repaired_geometry_000008.npz',**arrays)
    return result,dict(updates=200,optimizer='Adam eps1e-8',dc_lr=.0025,opacity_lr=.05,loss='full-frame original observation RGB L1, raw composite, no edited/heldout GT, no masks',trainable_rows=len(indices),trainable_parameters=4*len(indices),geometry_fixed=True,DC_only=True,opacity_parameterization='sigmoid logit',loss_trace=losses,save_reload_render_max_abs=error,checkpoint=str(checkpoint),model_scope='single-time local diagnostic, no joint/new model training')
