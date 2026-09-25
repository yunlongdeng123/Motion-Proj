"""检查实际文件分母、固定法线样例、每张已生成SAM图的ID合同。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from generate_sam_prior import dynamic_objects

root = Path('/root/autodl-tmp/data/v76_vadgs')
result = {'task_id': 'VADGS-MATCHED-PRIORS-20260926', 'seed': 0,
          'failure_ledger_refs': ['V76-F01','V76-F02'], 'failure_ledger_delta': 'V76-F02', 'scenes': {}}
for name in ['scene_0230','scene_0255']:
    scene = root/name
    expected = {f'{f:03d}_{c}' for f in range(61) for c in range(6)}
    counts = {}
    for kind in ['depth_v2','normal_img','sam_masks','sam_bkgd_masks']:
        found = {p.stem for p in (scene/kind).glob('*.png') if not p.name.endswith('.tmp.png')}
        counts[kind] = {'present_expected': len(found & expected), 'missing': len(expected-found)}
    normals = {}
    for frame in (0,20,60):
        for camera in (0,5):
            stem = f'{frame:03d}_{camera}'
            path = scene/'normal_img'/f'{stem}.png'
            if not path.is_file():
                continue
            array = np.asarray(Image.open(path))
            assert array.shape == (900,1600,3) and array.dtype == np.uint8
            norm = np.linalg.norm(array.astype(np.float32)/255*2-1,axis=-1)
            normals[stem] = {'norm_min': float(norm.min()), 'norm_max': float(norm.max())}
            assert norm.min() > .97 and norm.max() < 1.03
    actors = dynamic_objects(scene)
    sam_rows = []
    sam_source = scene/'sam_masks'
    quarantined = (scene/'sam_prior_evidence/DYNAMIC_IDENTITY_BLOCKED.json').is_file()
    if quarantined:
        sam_source = scene/'sam_prior_evidence/rejected_box_only_sam_masks_20260926'
    for path in sorted(sam_source.glob('*.png')):
        if path.stem not in expected:
            continue
        frame = int(path.stem.split('_')[0])
        array = np.asarray(Image.open(path))
        assert array.shape == (900,1600,3) and array.dtype == np.uint8
        unique = set(int(x) for x in np.unique(array)) - {255}
        allowed = {key for key, frames in actors.items() if frame in frames}
        assert unique <= allowed, (path,unique-allowed)
        background = np.asarray(Image.open(scene/'sam_bkgd_masks'/path.name))
        assert background.shape == (900,1600) and background.dtype == np.uint8
        assert 1 not in background and background.max() <= 254
        sam_rows.append({'name': path.stem, 'ids': sorted(unique), 'valid_ids': True,
                         'dynamic_pixels': int(np.any(array != 255,axis=-1).sum()),
                         'background_regions': int(len(np.unique(background))-int(np.any(background==0)))})
    result['scenes'][name] = {'expected_views':366, 'counts': counts, 'normal_samples':normals,
        'sam_encoding_checks': sam_rows, 'dynamic_ids': sorted(actors),
        'dynamic_identity_gate': 'failed' if quarantined else 'not_checked',
        'encoding_check_source':str(sam_source),
        'training_input_complete': not quarantined and all(count['missing']==0 for count in counts.values())}
(root/'matched_priors_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{'counts':v['counts'], 'training_input_complete':v['training_input_complete']} for k,v in result['scenes'].items()}))
