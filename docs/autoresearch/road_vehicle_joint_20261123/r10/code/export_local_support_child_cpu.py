"""Deactivate only local rows with actual protected/outside support.

The original asset is immutable. All numeric fields except the active mask
and all inherited actor/native/road parents retain identical array bytes.
"""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import bank,read,write,sha,load_local_asset,persist
from render_new_local_zero import pin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','audit','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();asset=load_local_asset(a.asset,verify_parent_pins=True)
    audit=read(a.audit/'receipt.json');assert audit['complete'] and audit['CPU_only']
    assert audit['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    z=bank(a.audit/'local_row_footprint.npz');assert audit['sidecar']['sha256']==sha(a.audit/'local_row_footprint.npz')
    names=z['ROI_names'].tolist();mass=z['allview_mass'];n=len(asset.field['xyz'])
    assert mass.shape==(n,5,len(names)) and np.isfinite(mass).all() and (mass>=0).all()
    safe=(mass[:,:,names.index('outside_allowed_edit')]==0).all(1)&(mass[:,:,names.index('actor_protected')]==0).all(1)&(mass[:,:,names.index('K_protected')]==0).all(1)
    assert np.array_equal(safe,z['all_five_outside_car_K_exact_zero'])
    field={k:v.copy() for k,v in asset.field.items()};field['active']=field['active']&safe
    assert int((~safe).sum())==audit['any_forbidden_local_rows']
    pruning=dict(schema='R10.active_only_actual_support_pruning.v1',complete=True,action='active_mask_intersection',
        parent_asset_manifest=pin(a.asset/'manifest.json'),measurement=audit['measurement'],audit=pin(a.audit/'receipt.json'),
        row_scope=pin(a.audit/'local_row_footprint.npz'),selection='all-five actual outsideEdit/car/K alphaT exactly zero',
        numerical_zero_tolerance=0,source_RGB_used=False,rows=n,active_before=int(asset.field['active'].sum()),active_after=int(field['active'].sum()),
        deactivated_rows=int((asset.field['active']&~field['active']).sum()),native_or_old_road_rows_edited=0,optimizer_updates=0,
        all_centres_RGB_scales_opacity_rotations_unchanged=True,quality_accepted=False)
    manifest=dict(asset.manifest,parent_local_manifest_sha256=sha(a.asset/'manifest.json'),support_pruning=pruning,
        quality_accepted=False,whole_scene_restoration_accepted=False)
    persist(a.output,field,asset.world_from_patch,asset.lineage,a.asset/'native_gate.npz',a.asset/'native_gate_proof.json',manifest)
    fresh=load_local_asset(a.output,verify_parent_pins=True)
    comparisons={k:bool(fresh.field[k].dtype==v.dtype and fresh.field[k].shape==v.shape and fresh.field[k].tobytes()==v.tobytes()) for k,v in asset.field.items() if k!='active'}
    assert all(comparisons.values()) and np.array_equal(fresh.field['active'],asset.field['active']&safe)
    for k in ['lineage.npz','pose.json','native_gate.npz','native_gate_proof.json']:assert sha(a.asset/k)==sha(a.output/k)
    write(a.output/'active_only_proof.json',dict(pruning,child_manifest=pin(a.output/'manifest.json'),fresh_loader_pass=True,
        unchanged_array_byte_checks=comparisons,all_inherited_geometry_source_and_parent_pins_exact=True,
        lineage_pose_native_gate_byteexact=True,CPU_only=True,nativeF=0,nativeB=0))
    print(dict(complete=True,asset=str(a.output),active_before=pruning['active_before'],active_after=pruning['active_after'],deactivated_rows=pruning['deactivated_rows'],fresh_loader_pass=True,nativeF=0,nativeB=0))

if __name__=='__main__':main()
