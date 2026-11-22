"""只读 CPU 重建原 checkpoint 选中行，审计冻结原生源包；不加载 native engine。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,time
import numpy as np
import torch
torch.set_num_threads(1)


def delta(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return dict(shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype,
                exact=bool(np.array_equal(a,b)),max_abs=float(abs(a.astype(float)-b.astype(float)).max()))


def main():
    started=time.monotonic();C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
    R=C/'road_source_fidelity_20261122_r2'
    A=C/'road_surface_residual_20261121_r2/appearance_fit/asset'
    manifest=json.loads((A/'persistent/scene.json').read_text())
    frames={int(v['key']):v for v in manifest['frames']}
    checkpoint=Path(manifest['model'])/f"chkpnt{manifest['iteration']}.pth"
    payload,step=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert len(payload)==18 and step==manifest['iteration']
    with np.load(C/'road_gaussian_transfer_20261122_r1/source_selection/candidates/patch_02.npz',allow_pickle=False) as z:
        source={k:z[k] for k in z.files}
    world_from_model=np.linalg.inv(source['model_from_world'])
    frame=source['source_tangent_frame_world'];center=source['source_center_world']
    rows=[]
    with torch.no_grad():
      for run,name in [('native_static_02_115','frozen_native_source.npz'),('native_static_contributors_02_115','frozen_contributor_source.npz')]:
        with np.load(R/run/name,allow_pickle=False) as z:packet={k:z[k] for k in z.files}
        ids=packet['native_static_row_ids'];ti=torch.as_tensor(ids)
        xyz,dc,rest,scaling,rotation,opacity,t,scaling_t,velocity=[payload[i][ti].detach() for i in range(1,10)]
        ref_time=float(frames[115]['model_time']);angular=2*np.pi/float(payload[16])
        means=xyz+velocity*torch.sin((ref_time-t)*angular)/angular
        marginal=torch.exp(-.5*(t-ref_time)**2/torch.exp(scaling_t)**2)
        expected=dict(means3D=means.numpy(),shs=torch.cat((dc,rest),dim=1).numpy(),
            scales=torch.exp(scaling).numpy(),rotations=torch.nn.functional.normalize(rotation,dim=1).numpy(),
            opacities=(torch.sigmoid(opacity)*marginal).numpy(),mask=(marginal[:,0]>.05).numpy())
        checks={k:delta(packet[k],v) for k,v in expected.items()}
        world=packet['means3D']@world_from_model[:3,:3].T+world_from_model[:3,3]
        local=(world-center)@frame[:3,:3]
        row=dict(run=run,path=str(R/run/name),rows=len(ids),native_checkpoint_rows=len(payload[1]),
            reference_raw=115,reference_model_time=ref_time,checkpoint_active_SH_degree=int(payload[0]),
            native_row_IDs_sorted_unique=bool(np.array_equal(ids,np.unique(ids))),
            checkpoint_reference_state_CPU_reconstruction=checks,
            world_from_saved_model_coordinates=delta(packet['world_xyz'],world),
            row_center_abs_height_relative_source_plane_quantiles_m=np.quantile(abs(local[:,2]),[.5,.9,.95,1]).tolist(),
            SH_coefficients_preserved=checks['shs']['exact'],source_fields=list(packet),
            prepared_depth_feature_is_reference_only=True)
        if 'reference_mass' in packet:
            receipt=json.loads((R/run/'receipt.json').read_text())
            native_rows=[]
            for e in receipt['rows']:
                raw=e['raw']
                with np.load(R/run/f'{raw:03d}'/'native_static_float.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
                score=f['score_mask'];d=f['frozen_rgb'][score]-f['live_rgb'][score]
                native_rows.append(dict(raw=raw,score_n=int(score.sum()),score_alpha_min=float(f['frozen_alpha'][score].min()),
                    frozen_vs_full_native_RGB_MAE=float(abs(d).mean()),frozen_vs_full_native_RGB_max=float(abs(d).max()),
                    frozen_vs_full_native_alpha_max=float(abs(f['frozen_alpha'][score]-f['live_alpha'][score]).max())))
                if raw==115:
                    total_mass=float(f['live_alpha'][score].astype(float).sum()/3)
                    source_mass=float(packet['reference_mass'].astype(float).sum())
                    row['reference_mass_audit']=dict(selected_mass=source_mass,total_mass_from_alpha=total_mass,
                        independently_retained_ratio=source_mass/total_mass,
                        receipt_retained_ratio=receipt['geometry']['retained_reference_mass'],
                        min_selected_row_mass=float(packet['reference_mass'].min()),
                        selected_mass_nonnegative=bool((packet['reference_mass']>=0).all()),
                        mass_within_true_source_plane_10cm=float(packet['reference_mass'][abs(local[:,2])<.1].sum()/packet['reference_mass'].sum()),
                        full_unselected_gradient_vector_saved=False)
            row['existing_native_output_CPU_reread']=native_rows
        rows.append(row)
    print(json.dumps(dict(schema='static_original_checkpoint_source_pack_CPU_audit.v1',rows=rows,
        checkpoint=str(checkpoint),checkpoint_loaded_CPU_mmap=True,engine_imported=False,
        new_native_calls=0,optimizer_updates=0,source_assets_modified=False,
        seconds=time.monotonic()-started),ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
