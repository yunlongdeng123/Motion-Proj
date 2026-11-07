"""CPU bake of a fitted shared parent table into an immutable canonical layer."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from shared_fit_common import OUT,load,read,write,sha,tree_hash
from fusion_colour_runtime import PARENT


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['direct','fusion'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();branch=OUT/args.variant;material_path=branch/'material_override.npz'
    m=load(material_path);fit=read(OUT/'fit_contract.json');base_manifest=read(PARENT/'scene_manifest.json')
    assert not fit['placeholder_engineering_only'] and int(m['formal_steps'])==400
    assert str(m['fit_contract_sha256'])==sha(OUT/'fit_contract.json') and str(m['variant'])==args.variant
    source_path=PARENT/base_manifest['new_road']['file'];assert sha(source_path)==str(m['asset_sha256'])
    a=load(source_path);parent=m['colours_srgb'];ids=a['child_to_parent_ids']
    assert parent.shape==(10098,3) and parent.dtype==np.float32 and np.isfinite(parent).all() and parent.min()>=0 and parent.max()<=1
    assert np.array_equal(parent[~m['eligible']],m['initial_colours_srgb'][~m['eligible']])
    payload={key:value.copy() for key,value in a.items()}
    payload['render_colours_srgb']=parent[ids].copy();payload['reference_srgb']=parent[ids].copy()
    payload['material_parent_colours_srgb']=parent.copy();payload['parent_render_colours_srgb']=parent.copy()
    payload['initial_parent_colours_srgb']=m['initial_colours_srgb'].copy()
    payload['writeback_eligible_parents']=m['eligible'].copy();payload['writeback_eligible_parent_ids']=m['eligible_parent_ids'].copy()
    payload['material_sha256']=np.array(sha(material_path))
    payload['material_selection']=np.array(args.variant+'_190SOURCE_SHARED_ATLAS_TARGET_WRITEBACK')
    payload['material_target_role']=np.array(fit['target_roles'][args.variant])
    payload['material_role']=np.array('FITTED_ATLAS_TARGET_MATERIAL_NOT_OBSERVED_HIDDEN_RGB')
    payload['writeback_fit_contract_sha256']=np.array(sha(OUT/'fit_contract.json'))
    payload['writeback_target_manifest_sha256']=m['target_manifest_sha256'].copy()
    changed_allowed={'render_colours_srgb','reference_srgb','material_parent_colours_srgb','parent_render_colours_srgb',
        'material_sha256','material_selection','material_role'}
    unchanged={key:tree_hash(value)==tree_hash(payload[key]) for key,value in a.items() if key not in changed_allowed}
    assert all(unchanged.values())
    args.output.mkdir(parents=True,exist_ok=False);target=args.output/'new_road_canonical_candidate.npz'
    np.savez_compressed(target,**payload);reloaded=load(target)
    reload_checks={key:tree_hash(value)==tree_hash(reloaded[key]) for key,value in payload.items()};assert all(reload_checks.values())
    source={}
    for path in (PARENT/'canonical_road_runtime.py',PARENT/'relative_mip.py',Path(__file__),Path(__file__).with_name('verify_baked_material.py')):
        shutil.copy2(path,args.output/path.name);source[path.name]=dict(file=path.name,sha256=sha(args.output/path.name))
    for path,name in ((PARENT/'scene_manifest.json','baseline_scene_manifest.json'),(material_path,'material_override.npz'),
        (OUT/'fit_contract.json','fit_contract.json'),(Path(fit['target_manifest']),'target_manifest.json')):shutil.copy2(path,args.output/name)
    manifest=dict(base_manifest)
    manifest.update(schema='GEOMETRY_SUPPORT_CANONICAL_WITH_SHARED_ATLAS_MATERIAL_V1',human_verdict=None,
        new_road=dict(base_manifest['new_road'],sha256=sha(target),material_variant=args.variant,
            qualified_parent_rows=int(m['eligible'].sum()),frozen_parent_rows=int((~m['eligible']).sum())),
        replay_sources=source,lineage=dict(baseline_manifest_file='baseline_scene_manifest.json',baseline_manifest_sha256=sha(PARENT/'scene_manifest.json'),
            baseline_asset_sha256=sha(source_path),new_material_file='material_override.npz',new_material_sha256=sha(material_path),
            fit_contract_file='fit_contract.json',fit_contract_sha256=sha(OUT/'fit_contract.json'),
            target_manifest_file='target_manifest.json',target_manifest_sha256=fit['target_manifest_sha256'],
            target_role=fit['target_roles'][args.variant],source_information_matched_between_variants=True),
        material_writeback=dict(formal_steps=400,shared_parent_RGB=True,child_inheritance=4,optimizer='Adam .01 projected[0,1]',
            geometry_updates=0,opacity_updates=0,original_parameter_updates=0,
            old_parent_reference_table_role='parent_reference_srgb/parent_rgb retain prior material provenance; current parent material is material_parent_colours_srgb and parent_render_colours_srgb',
            new_target_source_pool='190 real nonHELD source views; parent material/proxy was previously conditioned on199'),
        known_unresolved=list(base_manifest['known_unresolved'])+[
            'Only2053 parent colours are eligible for current BUILD supervision;8045 preserve baseline bytes.',
            'Fusion target is a constructed shared atlas target, not independently observed hidden RGB.',
            'Full49 comparison and independent human verdict determine quality; bake/replay establishes reproducibility only.'],
        GPU_parity_state_at_export='NOT_RUN',GPU_parity_receipt='parity/receipt.json',adoption_claim=False)
    write(args.output/'scene_manifest.json',manifest)
    write(args.output/'cpu_bake_verification.json',dict(status='PASS_CPU_MATERIAL_BAKE',variant=args.variant,
        material_sha256=sha(material_path),asset_sha256=sha(target),scene_manifest_sha256=sha(args.output/'scene_manifest.json'),
        unchanged_baseline_fields=unchanged,all_payload_reload_checks=reload_checks,
        parent_colour_exact=np.array_equal(reloaded['material_parent_colours_srgb'],parent),
        child_colour_exact=np.array_equal(reloaded['render_colours_srgb'],parent[ids]),
        unqualified_colour_byte_exact=True,original_parameter_updates=0,GPU_forwards=0))
    print(json.dumps(dict(variant=args.variant,asset=str(target),asset_sha256=sha(target),manifest=str(args.output/'scene_manifest.json'),
        manifest_sha256=sha(args.output/'scene_manifest.json'),unchanged_field_count=len(unchanged))))


if __name__=='__main__':main()
