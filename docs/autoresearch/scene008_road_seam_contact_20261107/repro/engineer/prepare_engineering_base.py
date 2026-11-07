"""Read-only parent snapshot and CPU engineering checks; never imports CUDA."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import numpy as np
from world_surface_sampling import read_surface,resample_surface,summarize,array_hash,sha

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PARENT=CAM/'scene008_road_r0_v44_20261107_r1'
RUN=CAM/'scene008_road_seam_contact_20261107_r1'


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=PARENT/'candidates/baked_cellfit_strip1000_candidate_v1/scene_manifest.json')
    p.add_argument('--output',type=Path,default=RUN/'engineer')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'cpu_base_receipt.json').exists():raise FileExistsError('Preserve existing CPU engineering snapshot')
    manifest=json.loads(args.manifest.read_text());parent_asset=args.manifest.parent/manifest['new_road']['file']
    assert sha(parent_asset)==manifest['new_road']['sha256']
    inputs=args.output/'inputs';inputs.mkdir(exist_ok=False)
    for source,name in [(parent_asset,manifest['new_road']['file']),(args.manifest,'scene_manifest.json'),
        (args.manifest.parent/'baked_road_runtime.py','baked_road_runtime.py')]:
        shutil.copy2(source,inputs/name);assert sha(source)==sha(inputs/name)
    snapshot=args.output/'source_snapshot';snapshot.mkdir(exist_ok=False)
    sources=[PARENT/'source/r0_runtime.py',PARENT/'source/corridor_runtime.py',
        CAM/'scene008_v44_rgb_asset_20261107_r1/source/v44_runtime.py',args.manifest.parent/'baked_road_runtime.py']
    source_hashes={}
    for source in sources:
        dest=snapshot/source.name;shutil.copy2(source,dest);assert sha(source)==sha(dest)
        source_hashes[str(source)]=sha(source)
    parent=read_surface(inputs/manifest['new_road']['file']);checks={};previews={}
    immutable={k:array_hash(v) for k,v in parent.items()}
    for factor in (1,2,4):
        child=resample_surface(parent,factor);summary=summarize(parent,child,factor)
        assert summary['material_copy_exact'] and summary['opacity_logit_copy_exact'] and summary['source_identity_copy_exact']
        assert abs(summary['intrinsic_plane_kernel_area_ratio']-1)<1e-12
        if factor==1:
            checks['n1_geometry_material_opacity_arrays_bitwise']=all(np.array_equal(child[k],parent[k])
                for k in ('local_xyz','world_xyz','model_xyz','normal_local','normal_world','gaussian_sigma_m','reference_srgb','opacity','opacity_logit'))
        summary['CPU_geometry_payload_bytes']=sum(v.nbytes for v in child.values())
        summary['GPU_execution_count']=0
        previews[str(factor)]=summary
    assert checks['n1_geometry_material_opacity_arrays_bitwise']
    checks['parent_arrays_unchanged']=all(array_hash(parent[k])==h for k,h in immutable.items())
    checks['existing_ABI_preserved']=dict(features=24,original_groups_alpha=[2,6],original_groups_RGB=[6,18],
        newroad_alpha=18,newroad_RGB=[19,22],newroad_centre_depth_m=22,asset_centre_depth_m=23)
    code={}
    for name in ('world_surface_sampling.py','refined_road_runtime.py','prepare_engineering_base.py'):
        path=args.output/name;ast.parse(path.read_text());code[name]=sha(path)
    plan=dict(status='CPU_ENGINEERING_DRAFT_NO_GPU_NO_TRAINING',parent_archive_HEAD='851703e7',
        parent_manifest=str(args.manifest),parent_manifest_sha256=sha(args.manifest),
        parent_canonical_sha256=sha(parent_asset),local_manifest=str(inputs/'scene_manifest.json'),
        source_hashes=source_hashes,engineering_source_sha256=code,checks=checks,CPU_geometry_previews=previews,
        interface=dict(class_name='RefinedRoadRuntime',constructor='(manifest_path,factor=2,verify_dependencies=False,opacity_fn=None,opacity_policy_id=None)',
            properties=['rt','parent(BakedRoadRuntime)','base(parent R0Runtime)'],
            methods=['scene(raw,delete=False)','extension_rows(raw,opacity_fn=None)','render_scene(raw,delete=False)','receipt()']),
        evaluation_scopes=dict(blue_seam='separate road appearance/continuity assessment',red_tire='separate tire/contact/occlusion assessment'),
        opacity='copy parent nominal .99; intrinsic plane kernel area preserved, not screen alpha',
        normal_sigma='same parent surface; all kernel axes /factor, no centre height bias',
        AA='not implemented; optional functional opacity hook reserved for reviewed formula only',
        rollback='factor1 directly uses original parent scene rows; no cached asset is overwritten',
        required_next_checks=['fresh process, one wrapper per factor','n1 GPU exact parent parity','single n2 five-key paired pilot',
            'original24channel conservation and raw oldroad subtraction residuals','no automatic n4 escalation'],
        time_cost='one native shared sort per scene; same24channel ABI. Actual timing required; .3I filter floor prevents an alpha-preservation claim',
        old_model_updates=0,newroad_training_updates=0,GPU_forwards=0,shutdown_performed=False)
    write(args.output/'cpu_base_receipt.json',plan)
    print(json.dumps(dict(status=plan['status'],manifest=plan['local_manifest'],parent_canonical_sha256=plan['parent_canonical_sha256'],
        n1_bitwise=checks['n1_geometry_material_opacity_arrays_bitwise'],rows={k:v['child_rows'] for k,v in previews.items()})),flush=True)


if __name__=='__main__':main()
