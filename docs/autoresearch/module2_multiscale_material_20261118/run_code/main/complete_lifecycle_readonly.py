"""工程update到360即止；最终全部physical query重构+权威field/native只读验证。"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from train_world_material import TrainingState,P
from field_native_io import save_field_asset,load_field_context
from road_chart_sampling import make_exact_sample_plan
from run_material_lifecycle import compare


def run(output,owner):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    s=TrainingState(P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',P/'stage0_native_r1',P/'execution_contract.json',owner,purpose='smoke')
    previous=json.loads((P/'lifecycle_600s_r2/receipt.json').read_text())
    saved=torch.load(P/'lifecycle_600s_r2/final_smoke_state.pt',map_location='cpu',weights_only=False)
    s.arm=saved['arm'];s.restore(saved)
    if compare(saved,s.snapshot())!=0:raise RuntimeError('最终smoke完整恢复不exact')
    m=s.field_metadata
    with np.load(m['geometry_kernel']) as f:g={k:f[k].copy() for k in ['active_world_rows','world_xyz','normal_world','sigma_m']}
    with np.load(m['A0_source']) as f:a={k:f[k].copy() for k in ['surface_world_xyz','grid_world_row','world_from_uv']}
    with np.load(Path(m['coverage_parent'])/'provenance.npz') as f:keys=f['world_keys'].copy()
    query=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],keys,g['active_world_rows'],g['normal_world'],g['sigma_m'],
        a['world_from_uv'],order=15,world_xyz=g['world_xyz'],sample_uv_path=output/'reconstructed_world_quad_UV_f64.npy')
    error=0.;valid_exact=True
    for begin in range(0,len(g['active_world_rows']),1024):
        end=begin+1024
        error=max(error,float(np.abs(query['sample_uv'][begin:end]-s.model.sample_uv[begin:end].cpu().numpy()).max()))
        valid_exact &= np.array_equal(query['geometric_sample_valid'][begin:end],s.model.geometric_sample_valid[begin:end].cpu().numpy())
    if error>1e-7 or not valid_exact:raise RuntimeError('最终全量worldΣquery与权威field不同')
    colors=save_field_asset(s,output/'final_smoke_field_asset',dict(role='final360update smoke authority; zero additional updates'))
    field=load_field_context(output/'final_smoke_field_asset',lease_file=P/'gpu_lease.json',lease_owner=owner)
    rows=[]
    with torch.no_grad():
        for raw in s.ctx.frames:
            expected=s.ctx.render_road(raw,colors);joint=s.ctx.render_joint(raw,colors)
            actual=field.render_road(raw);actualjoint=field.render_joint(raw);cache=field.base.render_road(raw)
            er=float((expected['rgb']-actual['rgb']).abs().max());ec=float((actual['rgb']-cache['rgb']).abs().max())
            with np.load(Path(s.contract['coverage_parent']).parent/f'{raw:03d}_native.npz') as f:parent=f['joint_rgb'];alpha=f['road_alpha']
            K=s.ctx.K_masks[raw];dk=np.abs(actualjoint['rgb'].cpu().numpy()-parent).max(0)
            if max(er,ec)>1e-6 or np.any(dk[K]!=0) or not np.array_equal(actual['alpha'][0].cpu().numpy(),alpha):
                raise RuntimeError('最终field重载native/缓存/K/alpha未过')
            np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=actual['rgb'].cpu().numpy(),road_alpha=alpha,
                joint_rgb=actualjoint['rgb'].cpu().numpy(),K=K)
            rows.append(dict(raw=raw,field_native_RGB_max=er,baked_cache_RGB_max=ec,joint_K_max=float(dk[K].max()),
                RGB8_max=int((torch.round(expected['rgb']*255)-torch.round(actual['rgb']*255)).abs().max())))
    extra=field.base.native_calls;field.close();s.assert_frozen();calls=s.ctx.native_calls+extra;s.ctx.close()
    receipt=dict(schema='multiscale_final_readonly_lifecycle.v1',GPU_executed=True,optimizer_updates=0,native_calls=calls,
        full_world_quad_points=int(query['sample_uv'].shape[0]*query['sample_uv'].shape[1]),world_quad_UV_max=error,
        geometric_valid_exact=bool(valid_exact),source_reconstruction=query['sampling_receipt'],
        field_requery=field.query_receipt,rows=rows,elapsed_seconds=time.monotonic()-start)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    verified=dict(previous,completed=True,source_smoke_receipt=str(P/'lifecycle_600s_r2/receipt.json'),
        final_readonly_receipt=str(output/'receipt.json'),read_only_completion_optimizer_updates=0,
        elapsed_seconds=previous['elapsed_seconds']+receipt['elapsed_seconds'],elapsed_update_smoke_seconds=previous['elapsed_seconds'],
        elapsed_readonly_seconds=receipt['elapsed_seconds'],total_native_calls=previous['total_native_calls']+calls,
        checks=previous['checks']+[dict(test='final_all34197300_worldquad_rebuild_and_native',receipt=str(output/'receipt.json'))])
    (output/'verified_smoke_receipt.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(dict(completed=True,updates=previous['optimizer_updates'],readonly_updates=0,total_elapsed=verified['elapsed_seconds'],readonly_native_calls=calls)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    a=p.parse_args();run(a.output,a.lease_owner)
