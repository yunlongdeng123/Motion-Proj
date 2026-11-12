"""精简归档：保存图块、固定坐标、合法观测与mask，原始51张RGB以路径溯源。"""
from pathlib import Path
import argparse,json,tarfile


def package(dataset,code,output):
    root=Path(dataset).resolve();code=Path(code).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    active_manifest='dataset_fineguard.json' if (root/'dataset_fineguard.json').is_file() else 'dataset.json'
    m=json.loads((root/active_manifest).read_text());sources=json.loads((root/m.get('sources','sources.json')).read_text())['sources']
    names=[active_manifest,m.get('sources','sources.json'),'train.json','eval.json','audit_receipt.json','selection_receipt.json',
           'reference_selection_audit.json','contact_sheet.png','contact_train.png','contact_eval.png']
    if (root/'continuous_source_guard_receipt.json').is_file():names.append('continuous_source_guard_receipt.json')
    for directory in ['samples','candidates',('all_frame_continuous_query_masks' if active_manifest=='dataset_fineguard.json' else 'all_frame_query_masks')]:
        names += [str(p.relative_to(root)) for p in (root/directory).rglob('*') if p.is_file()]
    names += [s['observations'] for s in sources]
    names += [s['feature_allowed_mask'] for s in sources]
    for name in ['world_xyz.npy','local_xyz.npy','geometry_valid.npy','measured_geometry_support.npy',
                 'measured_distance_m.npy','supervision_valid.npy','feature_exclude_world_mask.npy','exclude_world_mask.npy','eval_world_mask.npy','observed_valid.npy']:
        names.append('atlas/'+name)
    index=dict(schema='road_material_bundle.v1',dataset_directory=root.name,train=len(m['train']),eval=len(m['eval']),candidates=len(m['candidates']),
        original_rgb_copied=False,original_source_rgb_paths=[s['image_path'] for s in sources],
        feature_inputs_omitted=True,rebuild_feature_inputs='python code/materialize_feature_inputs.py --dataset DATASET_DIRECTORY',
        source_allowed_masks_included=True,source_camera_and_projection_weights_included=True,
        supervision_atlas_rgb_omitted=True,atlas_input_rgb_omitted=True,
        omitted_manifest_fields=['atlas.input_rgb','atlas.supervision_only_rgb'],
        reason='Training and retrieval use explicit sample/candidate images; original RGB and full base atlas remain preserved remotely.',
        geometry='fixed measured plane with exact dense world coordinates included',data_revision=m['data_revision'])
    index_path=output.parent/'dataset_bundle_index.json';index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    script_names=['build_road_dataset.py','select_clear_revision.py','audit_road_dataset.py','audit_reference_selection.py',
                  'world_feature_field.py','materialize_feature_inputs.py','repair_continuous_source_masks.py','package_road_dataset.py','scene065_config_v2_clean.json','README.md']
    with tarfile.open(output,'w:gz',compresslevel=3,dereference=True) as tar:
        for name in sorted(set(names)):
            path=root/name
            if not path.is_file():raise FileNotFoundError(path)
            tar.add(path,arcname=root.name+'/'+name,recursive=False)
        if active_manifest!='dataset.json':
            tar.add(root/active_manifest,arcname=root.name+'/dataset.json',recursive=False)
            tar.add(root/'dataset.json',arcname=root.name+'/dataset_pre_continuous_guard.json',recursive=False)
        tar.add(index_path,arcname='dataset_bundle_index.json')
        for name in script_names:tar.add(code/name,arcname='code/'+name,recursive=False)
    print(json.dumps(dict(status='PACKAGED_COMPACT_REPRODUCIBLE_ROAD_DATA',path=str(output),bytes=output.stat().st_size,
        data_files=len(set(names)),train=len(m['train']),eval=len(m['eval']),candidates=len(m['candidates']),full_rgb_copied=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);p.add_argument('--code',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();package(a.dataset,a.code,a.output)
