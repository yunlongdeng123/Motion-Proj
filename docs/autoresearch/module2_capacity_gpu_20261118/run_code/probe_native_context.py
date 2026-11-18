from pathlib import Path
import json
import numpy as np
from road_capacity_native import load_context,GPU_CAMPAIGN,CPU_CAMPAIGN,load_npz,write_json

def main():
    import torch
    out=GPU_CAMPAIGN/'check3/context_probe_r1';out.mkdir(parents=True,exist_ok=False)
    ctx=load_context(lease_file=GPU_CAMPAIGN/'gpu_lease.json',lease_owner='road_asset')
    scope=load_npz(GPU_CAMPAIGN/'check1/native_r1/qualified_scope.npz')
    assert np.array_equal(ctx.world_row_ids,scope['active_world_rows'])
    records=[]
    for raw in (86,110):
        data=load_npz(CPU_CAMPAIGN/f'supervision_v4_candidate_r1/{raw:03d}.npz')
        colors=ctx.base_rgb.detach().clone().requires_grad_(True)
        rendered=ctx.render_road(raw,colors);alpha=rendered['alpha'][0].detach()
        mask=torch.as_tensor(data['S_target_shadow_photo'],device=colors.device)
        target=torch.as_tensor(data['target_S_srgb'],device=colors.device)
        loss=((rendered['rgb']-alpha[None]*target).abs()*mask[None]).sum()/(3*mask.sum().clamp_min(1))
        gradient=torch.autograd.grad(loss,colors)[0]
        s_only=torch.as_tensor(scope['S_only_known']&scope['eligible_color'],device=colors.device)
        row=dict(raw=raw,loss=float(loss),S_photo_pixels=int(mask.sum()),gradient_finite=bool(torch.isfinite(gradient).all()),
            gradient_max=float(gradient.abs().max()),S_only_known_gradient_max=float(gradient[s_only].abs().max()),
            fixed_alpha_source_max=float((alpha-torch.as_tensor(data['fixed_initial_alpha'],device=alpha.device)).abs().max()))
        if not row['gradient_finite'] or row['gradient_max']==0:raise RuntimeError('actual S gradient invalid')
        records.append(row)
    eligible=torch.as_tensor(scope['eligible_color'],device=ctx.base_rgb.device)
    stresses=[]
    for value in (0.,1.):
        colors=ctx.base_rgb.clone();colors[eligible]=value
        stresses.append(dict(value=value,K=ctx.audit_K(colors)))
    ctx.assert_frozen()
    write_json(out/'completed.json',dict(completed=True,actual_optimizer_updates=0,actual_native=True,
        scope_rows=int(scope['eligible_color'].sum()),S_only_known_rows=int((scope['S_only_known']&scope['eligible_color']).sum()),
        rows=records,stress=stresses,native_calls=ctx.native_calls))
    print(json.dumps(dict(completed=True,path=str(out),gradient=records)),flush=True)
    ctx.close()

if __name__=='__main__':main()
