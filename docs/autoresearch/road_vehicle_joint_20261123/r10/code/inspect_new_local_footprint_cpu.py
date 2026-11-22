"""Read actual all-packet VJP to qualify the appended local-row footprint.

No render, parameter edit, or persistent gate is performed by this audit.
"""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import read,write,sha,bank,load_local_asset
from render_new_local_zero import pin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    asset=load_local_asset(a.asset,verify_parent_pins=True);m=read(a.measurement/'receipt.json')
    assert m['complete'] and m['nativeF']==m['nativeB']==5
    assert m['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    parent=read(asset.manifest['parent_contract']['path']);assert len(parent['actors'])==1
    actor=parent['actors'][0];assert sha(actor['asset'])==actor['sha256']
    with np.load(actor['asset'],allow_pickle=False) as z:nactor=len(z['xyz'])
    nlocal=len(asset.field['xyz']);regions=m['ROI_names'];allmass=[];rows=[]
    assert nlocal==175302 and nactor==419005
    for r in m['rows']:
        raw=r['raw'];path=a.measurement/f'{raw:03d}/raw_feature_alphaT_vjp.npz'
        assert sha(path)==r['raw_region_vjp']['sha256'];z=bank(path)
        assert z['ROI_names'].tolist()==regions
        nc=int(z['context_prefix_rows']);ntotal=int(z['all_packet_rows'])
        assert nc==r['all_context_rows'] and ntotal==r['all_packet_rows']
        assert ntotal==nc+nactor+nlocal,'Actual packet suffix not exactly canonical actor then local field'
        # JointNativeAdapter appends each full canonical actor packet to the
        # context prefix. LocalJointAdapter then appends the full local packet.
        start=nc+nactor;mass=np.stack([z['alphaT_'+k][start:] for k in regions],axis=1)
        assert mass.shape==(nlocal,len(regions)) and np.isfinite(mass).all() and (mass>=0).all()
        allmass.append(mass)
        per={name:dict(rows_positive=int((mass[:,j]>0).sum()),alphaT_mass=float(mass[:,j].astype(np.float64).sum())) for j,name in enumerate(regions)}
        forbidden=(mass[:,[regions.index(k) for k in ['outside_allowed_edit','actor_protected','K_protected']]]>0).any(1)
        rows.append(dict(raw=raw,actual_packet_rows=ntotal,context_prefix_rows=nc,canonical_actor_suffix_rows=nactor,
            local_suffix_start=start,local_suffix_end=ntotal,local_rows=nlocal,local_rows_touching_forbidden=int(forbidden.sum()),regions=per))
    mass=np.stack(allmass,axis=1)
    outside=mass[:,:,regions.index('outside_allowed_edit')];car=mass[:,:,regions.index('actor_protected')];K=mass[:,:,regions.index('K_protected')]
    inside=mass[:,:,regions.index('inside_allowed_edit')]
    safe=(outside==0).all(1)&(car==0).all(1)&(K==0).all(1)
    contributing=(inside>0).any(1);unsafe=~safe
    np.savez_compressed(a.output/'local_row_footprint.npz',local_row_id=np.arange(nlocal,dtype=np.int64),frames=np.asarray(m['frames']),
        ROI_names=np.asarray(regions),allview_mass=mass,all_five_outside_car_K_exact_zero=safe,
        any_inside_edit_contribution=contributing,proposed_retained_local_rows=safe,proposed_zero_local_rows=unsafe)
    code=Path(__file__).parent
    out=dict(schema='R10.new_local_actual_packet_footprint_CPU.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        measurement=pin(a.measurement/'receipt.json'),asset_manifest=pin(a.asset/'manifest.json'),actor_asset=pin(actor['asset']),
        packet_order_proof=dict(context_prefix_then_complete_canonical_actor_then_complete_local=True,
            native_adapter=pin(Path(asset.manifest['r8_code'])/'joint_native_adapter.py'),identity_adapter=pin(Path(asset.manifest['r8_code'])/'joint_identity_adapter.py'),
            local_adapter=pin(code/'local_joint_adapter.py'),actual_dimensions_verified_all_five=True),
        source_fields_unchanged=True,selection_uses_RGB=False,selection_uses_residual=False,current_alpha_not_used_for_masks=True,
        local_rows=nlocal,rows=rows,all_five_safe_local_rows=int(safe.sum()),any_forbidden_local_rows=int(unsafe.sum()),
        all_five_safe_and_inside_positive=int((safe&contributing).sum()),safe_but_all_five_invisible=int((safe&~contributing).sum()),
        unsafe_inside_alphaT_mass=float(inside[unsafe].astype(np.float64).sum()),safe_inside_alphaT_mass=float(inside[safe].astype(np.float64).sum()),
        local_outside_edit_alphaT_total=float(outside.astype(np.float64).sum()),local_car_alphaT_total=float(car.astype(np.float64).sum()),local_K_alphaT_total=float(K.astype(np.float64).sum()),
        sidecar=pin(a.output/'local_row_footprint.npz'),persistent_asset_created=False,child_render_authorized_by_this_audit=False,
        whole_quality_accepted=False,strict_zero_selection_not_relaxed=True)
    write(a.output/'receipt.json',out)
    print({k:out[k] for k in ['complete','local_rows','all_five_safe_local_rows','any_forbidden_local_rows','all_five_safe_and_inside_positive','local_outside_edit_alphaT_total','local_car_alphaT_total','local_K_alphaT_total']})

if __name__=='__main__':main()
