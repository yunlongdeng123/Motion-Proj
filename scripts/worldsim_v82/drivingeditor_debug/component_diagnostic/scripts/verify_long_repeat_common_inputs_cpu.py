"""Recheck original saved common49 inputs after B2; preserve evidence limits."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012')
B1 = C / 'continuation_diagnostic/original30k_to32000'
B2 = C / 'continuation_diagnostic/original30k_to32000_noise_reference'
COMMON = C / 'sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json'
COMMON_SHA = '1b7161a2b30a08fb7fd8354a71b257cf99ad39bef16732a5f24a367174c24d50'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(COMMON) == COMMON_SHA
    common = json.loads(COMMON.read_text())
    assert common['status'] == 'PASS' and common['all_original_input_sha_match']
    assert sha(Path(common['input_manifest'])) == common['input_manifest_sha256']
    source = Path(common['data_source'])
    plans = [json.loads((out / 'diagnostic_plan.json').read_text()) for out in (B1, B2)]
    assert plans[0]['repository'] == plans[1]['repository']
    assert plans[0]['training_config']['source_path'] == plans[1]['training_config']['source_path'] == str(source)
    for out in (B1, B2):
        status = json.loads((out / 'status.json').read_text())
        assert status['status'] == 'complete' and status['actual_exit_code'] == 0
    actual = {relative: sha(source / relative) for relative in common['file_sha256']}
    assert actual == common['file_sha256'] and len(actual) == common['full_input_file_count'] == 467
    assert not (source / 'dynamic_masks/shadow_mask_0').exists()
    masks = []
    required = []
    for index in range(49):
        name = f'0017{index:03}'
        relatives = (f'calib/{name}.txt', f'pose/{name}.txt', f'image/image_0/{name}.png',
                     f'sky/sky_0/{name}.png', f'dynamic_masks/dynamic_mask_0/{name}.png')
        for relative in relatives:
            assert relative in actual
            required.append(relative)
        for kind, relative in zip(('sky', 'actor'), relatives[-2:]):
            with Image.open(source / relative) as im:
                pixels = np.asarray(im.convert('L'))
                masks.append({'frame_id': f'image_0/{name}', 'kind': kind,
                              'source_relative_path': relative, 'encoded_sha256': actual[relative],
                              'decoded_original_L_uint8_shape': list(pixels.shape),
                              'decoded_original_L_uint8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
    result = {'schema_version': 1, 'status': 'PASS_common49_original_input_SHA_after_B2',
              'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only': True, 'GPU_or_training_started': False,
              'input_manifest_sha256': common['input_manifest_sha256'], 'common_independent_receipt_sha256': COMMON_SHA,
              'actual467_file_SHA_matches_original': True, 'actual245_required_common49_SHA_match': True,
              'new_shadow_mask_absent': True, 'same_original_source_path_and_repository': True,
              'common49_required_relative_paths': required, 'actual_input_SHA': actual,
              'common49_original_decoded_mask_receipts': masks,
              'evidence_limit': 'Actual original encoded and decoded mask inputs are unchanged, using the same pinned frozen loader. Runtime evaluation region counts and GT arrays are separately paired. NPZ does not contain runtime masks or an independent actualScene pose snapshot; no such pixel/pose snapshot equality is claimed.',
              'whole_debug_task_complete': False}
    path = B2 / 'receipts/common49_inputs_after_B2_cpu.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path)}))


if __name__ == '__main__':
    main()
