"""CPU export of 40392 world-road rows with one parent-level witness table.

The exact persistent native tuple and the selected 10098-row material override
are explicit inputs. A CPU-preview mode is visibly marked and cannot render.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from world_surface_sampling import read_surface,resample_surface,summarize,sha,array_hash


NATIVE_CAPTURE='NATIVE_REFINED_PERSISTENT_TUPLE_NO_FORWARD'
FIELDS=('render_model_xyz','render_scales_model','render_rotation_wxyz','render_opacity_logit','render_colours_srgb')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def export(parent_manifest,capture_path,material_path,expected_material_sha,output,relative_mip=False,allow_cpu_preview=False):
    parent_manifest=Path(parent_manifest).resolve();capture_path=Path(capture_path).resolve()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(parent_manifest.read_text(encoding='utf-8'));parent_asset=parent_manifest.parent/manifest['new_road']['file']
    assert sha(parent_asset)==manifest['new_road']['sha256']
    parent=load(parent_asset);record=load(capture_path)
    if material_path is None:
        material_path=parent_asset;expected_material_sha=sha(parent_asset)
        material={'colours_srgb':parent['reference_srgb']}
        material_selection='unchanged parent cellfit1000 material copied from canonical parent'
    else:
        material_path=Path(material_path).resolve();assert sha(material_path)==expected_material_sha
        material=load(material_path);material_selection='explicit material override'
    assert str(record['parent_manifest_sha256'])==sha(parent_manifest) and str(record['parent_asset_sha256'])==sha(parent_asset)
    assert int(record['factor'])==2
    native_capture=str(record['capture_origin'])==NATIVE_CAPTURE
    if not native_capture and not allow_cpu_preview:raise ValueError('Final export requires a real native tuple capture')
    if 'parent_asset_sha256' in material:assert str(material['parent_asset_sha256'])==sha(parent_asset)
    colours=material['colours_srgb'].astype(np.float32);assert colours.shape==(10098,3) and np.isfinite(colours).all()
    assert np.all((colours>=0)&(colours<=1))
    child=resample_surface(read_surface(parent_asset),2);ids=child['parent_indices'];n=len(ids);assert n==40392
    assert np.array_equal(ids,record['child_to_parent_ids'])
    expected_xyz=child['model_xyz'].astype(np.float32)
    expected_scale=(child['gaussian_sigma_m']*float(parent['meter_to_model'])).astype(np.float32)
    assert np.array_equal(record['render_model_xyz'],expected_xyz)
    assert np.array_equal(record['render_scales_model'],expected_scale)
    assert np.array_equal(record['render_rotation_wxyz'],record['parent_render_rotation_wxyz'][ids])
    assert np.array_equal(record['render_opacity_logit'],record['parent_render_opacity_logit'][ids])
    assert np.array_equal(record['parent_render_model_xyz'],parent['model_xyz'].astype(np.float32))
    assert np.array_equal(record['parent_render_scales_model'],(parent['gaussian_sigma_m']*float(parent['meter_to_model'])).astype(np.float32))
    # Confirm canonical AA's exact parent tuple equals the evaluated callback's inputs.
    from relative_mip import RelativeMip
    evaluated_parent=RelativeMip(parent_asset)
    parent_AA_checks={key:np.array_equal(record[f'parent_{key}'],value) for key,value in (
        ('render_model_xyz',evaluated_parent.parent_xyz),('render_scales_model',evaluated_parent.parent_scales),
        ('render_rotation_wxyz',evaluated_parent.parent_rotation))}
    assert all(parent_AA_checks.values()),parent_AA_checks
    assert str(record['captured_V44_binding_sha256'])==manifest['frozen_v44_binding_sha256']
    payload={key:value.copy() for key,value in child.items()}
    payload.update(active=np.ones(n,bool),child_to_parent_ids=ids.astype(np.int64),
        reference_srgb=colours[ids].copy(),material_parent_colours_srgb=colours.copy(),
        original_source_proxy_srgb=parent['original_source_proxy_srgb'][ids].copy(),
        source_valid_count=parent['source_valid_count'][ids].copy(),source_valid_count49=parent['source_valid_count49'][ids].copy(),
        original_appearance_provenance=parent['original_appearance_provenance'][ids].copy(),
        appearance_provenance=np.full(n,2,np.uint8),opacity=np.full(n,.99,np.float32),
        coordinate_anchor=parent['coordinate_anchor'].copy(),world_to_model=parent['world_to_model'].copy(),
        meter_to_model=parent['meter_to_model'].copy(),cell_m=np.array(float(parent['cell_m'])/2),
        parent_cell_m=parent['cell_m'].copy(),strip_lateral_bounds_m=parent['strip_lateral_bounds_m'].copy(),
        parent_manifest_sha256=np.array(sha(parent_manifest)),parent_asset_sha256=np.array(sha(parent_asset)),
        tuple_capture_sha256=np.array(sha(capture_path)),material_sha256=np.array(expected_material_sha),
        material_role=np.array('FITTED_VISIBLE_ROAD_MATERIAL_NOT_HIDDEN_RGB_TRUTH'),
        candidate_status=np.array('GEOMETRY_SUPPORT_RESEARCH_CANDIDATE'),capture_origin=record['capture_origin'].copy(),
        material_selection=np.array(material_selection))
    # Source/proxy/199-frame witnesses remain at 10098-parent cardinality exactly once.
    payload.update({f'parent_{key}':value.copy() for key,value in parent.items()})
    for key in FIELDS:
        payload[key]=record[key].copy();payload[f'parent_{key}']=record[f'parent_{key}'].copy()
    payload['render_colours_srgb']=colours[ids].copy()
    payload['render_tuple_geometry_authoritative']=np.array(True)
    target=output/'new_road_canonical_candidate.npz';np.savez_compressed(target,**payload)
    reloaded=load(target)
    checks={key:array_hash(value)==array_hash(reloaded[key]) for key,value in payload.items()}
    assert all(checks.values())
    assert reloaded['parent_source_rgb'].shape==(199,10098,3) and reloaded['parent_source_valid'].shape==(199,10098)
    assert not np.isnan(reloaded['parent_source_rgb'][reloaded['parent_source_valid']]).any()
    assert np.array_equal(reloaded['render_colours_srgb'],colours[ids])
    assert all(reloaded[key].dtype==np.float32 for key in FIELDS)
    sources={}
    for name in ('canonical_road_runtime.py','relative_mip.py','export_refined_canonical.py','verify_canonical_replay.py',
                 'capture_refined_tuple.py','world_surface_sampling.py'):
        path=Path(__file__).with_name(name)
        if not path.exists():raise FileNotFoundError('Missing replay delivery source: '+name)
        shutil.copy2(path,output/name);sources[name]=dict(file=name,sha256=sha(output/name))
    shutil.copy2(parent_manifest,output/'parent_scene_manifest.json')
    # Exact parent contract/dependency hashes include legacy/static/gates/env/trajectory and original road.
    policy=dict(relative_mip=bool(relative_mip),source_file='relative_mip.py',legacy_neg_fov=True,
        normalize_quaternion=False,covariance_floor_pixel2=.3,native_CUDA_modified=False,
        intrinsic_opacity_nominal=.99,asset_opacity_training_updates=0,
        interpretation='same-camera deterministic newroad amplitude; not a trained intrinsic asset parameter')
    dependencies=dict(manifest['runtime_dependencies'])
    source_path=Path(manifest['v44_contract_path']).parent.parent/'source/v44_runtime.py'
    if str(source_path) not in dependencies:raise ValueError('Frozen manifest must pin V44 runtime source')
    result=dict(schema='WORLD_ROAD_EXACT_TUPLE_RESEARCH_CANDIDATE_V1',status='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE',
        native_capture_verified=native_capture,new_road=dict(file=target.name,sha256=sha(target),rows=n,parent_rows=10098,
            always_enabled=True,intrinsic_opacity_nominal=.99,tuple_fields=list(FIELDS),witness_table_rows=10098,witness_frames=199),
        replay_sources=sources,v44_runtime_path=str(source_path),v44_contract_path=manifest['v44_contract_path'],
        frozen_v44_binding_sha256=manifest['frozen_v44_binding_sha256'],frozen_v44_binding=manifest['frozen_v44_binding'],
        runtime_dependencies=dependencies,renderer_policy=policy,
        lineage=dict(parent_manifest_file='parent_scene_manifest.json',parent_manifest_sha256=sha(parent_manifest),
            parent_asset_sha256=sha(parent_asset),capture_path=str(capture_path),capture_sha256=sha(capture_path),
            capture_source_sha256=str(record['refined_source_sha256']),material_path=str(material_path),material_sha256=expected_material_sha,
            material_selection=material_selection,upstream_material_sha256=manifest['lineage']['material_sha256']),
        shared_scene_state=manifest['shared_scene_state'],edit_operator='newroad always enabled; only physical a37 existence changes',
        known_unresolved=['Blue road seam and red wheel ordering require separate acceptance.',
            'Inferred hidden vehicle/road geometry conflict is not resolved by material fit or canonical export.',
            'Fitted road colours are not independently observed hidden-road RGB truth.'],
        exporter_sha256=sha(__file__),GPU_parity_receipt='parity/receipt.json',GPU_parity_state_at_export='NOT_RUN',adoption_claim=False)
    write(output/'scene_manifest.json',result)
    verification=dict(status='PASS_CPU_EXPORT_RELOAD',native_capture_verified=native_capture,rows=n,parent_rows=10098,
        exact_reload_checks=checks,tuple_hashes={key:array_hash(reloaded[key]) for key in FIELDS},
        evaluated_parent_relative_mip_geometry_exact_checks=parent_AA_checks,
        selected_material_equals_captured_parent_material=np.array_equal(payload['render_colours_srgb'],record['render_colours_srgb']),
        source_witness_table_single_copy=True,geometry=summarize(read_surface(parent_asset),child,2),
        original_parameter_updates=0,newroad_training_updates=0,GPU_used=False,adoption_claim=False)
    write(output/'cpu_export_verification.json',verification)
    write(output/'asset_interface.json',dict(arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in payload.items()},
        authoritative_render_tuple=list(FIELDS),child_to_parent='child_to_parent_ids into parent_* tables',
        source_rgb_role='parent_source_rgb: unchanged original qualified real FRONT observations; invalid samples remain unavailable NaN',
        material_role='render_colours_srgb: fitted persistent appearance, not independently observed hidden texture',
        native_capture_verified=native_capture))
    return dict(asset=str(target),sha256=sha(target),manifest=str(output/'scene_manifest.json'),manifest_sha256=sha(output/'scene_manifest.json'),
        native_capture_verified=native_capture,rows=n,all_reload_checks=all(checks.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-manifest',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--material',type=Path);p.add_argument('--expected-material-sha');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--relative-mip',action='store_true');p.add_argument('--allow-cpu-preview',action='store_true');args=p.parse_args()
    print(json.dumps(export(args.parent_manifest,args.capture,args.material,args.expected_material_sha,args.output,args.relative_mip,args.allow_cpu_preview)))


if __name__=='__main__':main()
