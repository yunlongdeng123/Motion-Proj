"""按实际PNG逐字节相等复用重复文件，减少传输；不哈希、不改像素。"""
from pathlib import Path
import json
import tarfile

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013/post30k_r3')
OUT=BASE.parent/'full30k_export_r3/actual_full30k_transport_light.tar.gz'
if OUT.exists():raise FileExistsError('保留旧包')
evaluation=json.loads((BASE/'evaluation/scene_evaluation_manifest.json').read_text())
summary=json.loads((BASE/'edit_summary.json').read_text())
assert summary['engineering_contract']['passed'] and summary['engineering_contract']['verified_FP32_pairs']==294
copies={}
for name,operation in summary['operations'].items():
    actor=operation['actor_index']
    for row in operation['frames']:
        index=row['frame'];frame=evaluation['streams']['observed']['frames'][index]
        prefix='edits/'+name+'/%03d_%s_'%(index,row['frame_id'])
        candidates={'gt':frame['images']['gt'],'before':frame['images']['composite'],
                    'static':frame['images']['static_rgb'],'after_static':frame['images']['static_rgb'],
                    'before_vehicles':frame['images']['all_vehicles_raw_rgb'],
                    'before_target_visible':frame['instances'][actor]['images']['visible_alpha'],
                    'after':frame['images']['composite'],
                    'after_vehicles':frame['images']['all_vehicles_raw_rgb'],
                    'after_target_visible':frame['instances'][actor]['images']['visible_alpha']}
        for field,source in candidates.items():
            relative=prefix+field+'.png';reference='evaluation/'+source
            if (BASE/relative).read_bytes()==(BASE/reference).read_bytes():copies[relative]=reference
files=list(BASE.rglob('*.png'))+list(BASE.rglob('*.json'))
assert sum(path.suffix=='.png' for path in files)==3969
payload=json.dumps({'schema':'exact_PNG_copy_transport_v1','original_PNG_count':3969,'copies':copies,
                    'comparison':'whole_actual_PNG_bytes_equal; no_hashes; no_generated_pixels'},indent=2).encode()
with tarfile.open(OUT,'w:gz',compresslevel=1) as package:
    for path in sorted(files):
        name=path.relative_to(BASE).as_posix()
        if name not in copies:package.add(path,arcname=name)
    import io
    info=tarfile.TarInfo('exact_PNG_copy_transport.json');info.size=len(payload)
    package.addfile(info,io.BytesIO(payload))
print(json.dumps({'bytes':OUT.stat().st_size,'exact_PNG_copies':len(copies),'original_PNGs':3969,'path':str(OUT)}))
