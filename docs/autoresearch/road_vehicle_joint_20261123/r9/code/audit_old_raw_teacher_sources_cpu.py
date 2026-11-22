"""Bind two previously reviewed Brown-domain raw teachers. CPU/read-only sources."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
EXPECTED = {
    86: ['eb813f4d2a9ab4035413dd4fe929656e8f3841f7e3d243d9ed9a85597eee802c', 'f07f8ecb7c4c9b9c8a39802eb8a15af2efc9046a062849cdc354d1800c6cb5e6', '015ba06692287865077074427b07fbcba414f27b9a0fae61751a921878bc2474'],
    110: ['fb3dc7b2da3308a8b74c2c93dd5e66f2882a8a77d13b031806eedafa18d8be2e', '750d12ab399eb2b9d06610068ebf5d5253034044bf2b13cbdd595a25cb140390', '5d130b552227a317d4def48493cdc8a64189a39664bd8251e7139b2ec04f6e2f'],
}


def pin(path, image=False):
    p = Path(path)
    d = {'path': p.as_posix(), 'bytes': p.stat().st_size,
         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    if image:
        with Image.open(p) as im:
            d.update(size_wh=list(im.size), mode=im.mode)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=C/'old_raw_teacher_joint_20261123_r9')
    ap.add_argument('--local-workspace', default='C:/Users/dengyunlong/Documents/Codex/2026-11-22/065-file-c-users-dengyunlong-documents-2')
    a = ap.parse_args()
    start = time.time()
    old = C/'road_2d_teacher_prompt_20261120_r1'
    plan_path = old/'prepared_r1/plan.json'
    plan = json.loads(plan_path.read_text())
    photo_root = Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    sources = []
    for raw, worker in [(86, 0), (110, 1)]:
        case = next(x for x in plan['cases'] if x['id'] == f'scene065_raw{raw:03d}')
        run = old/'formal_r1'/f'worker{worker}'/f'scene065_raw{raw:03d}__full1152_T0__legacy__seed101'
        rgb = {}
        for i, name in enumerate(['raw_model', 'raw_native', 'final_native']):
            rgb[name] = pin(run/f'{name}.png', True)
            assert rgb[name]['sha256'] == EXPECTED[raw][i], (raw, name, 'historical SHA differs')
        assert rgb['raw_model']['size_wh'] == [1152, 768]
        assert rgb['raw_native']['size_wh'] == rgb['final_native']['size_wh'] == [1920, 1280]
        model = np.asarray(Image.open(run/'raw_model.png').convert('RGB'))
        native = np.asarray(Image.open(run/'raw_native.png').convert('RGB'))
        final = np.asarray(Image.open(run/'final_native.png').convert('RGB'))
        prepared_obs = np.asarray(Image.open(case['observation']['path']).convert('RGB'))
        original_path = photo_root/'image/image_0'/f'0065{raw:03d}.png'
        original = np.asarray(Image.open(original_path).convert('RGB'))
        obs_path = photo_root/'observations'/f'{raw:03d}.json'
        obs = json.loads(obs_path.read_text())
        roles_path = Path(case['roles_native']['path'])
        with np.load(roles_path, allow_pickle=False) as role_npz:
            roles = {k: role_npz[k] for k in role_npz.files}
        role_stats = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                          'true_count': int(np.count_nonzero(v))} for k, v in roles.items() if v.ndim == 2}
        assert np.array_equal(prepared_obs, original), (raw, 'prepared observation differs from source photo')
        assert np.array_equal(native, np.asarray(Image.fromarray(model).resize((1920,1280), Image.Resampling.BILINEAR)))
        assert np.array_equal(final[~roles['W_output'].astype(bool)], original[~roles['W_output'].astype(bool)])
        assert np.array_equal(final[roles['K'].astype(bool)], original[roles['K'].astype(bool)])
        role_ref = pin(roles_path)
        role_ref.update(coordinate_domain='native Brown photo grid, 1920x1280; old output policy only',
                        arrays=role_stats, used_as_R9_geometric_visibility=False,
                        used_as_R9_permission=False)
        rgb['raw_model'].update(coordinate_domain='full-frame Brown distorted photo domain resized to 1152x768',
                                colour_encoding='sRGB encoded RGB uint8',
                                supervision_role='only permitted teacher RGB source; qualify surface independently',
                                local_path=f'{a.local_workspace}/work/old_raw_teacher_joint_20261123_r9/audit/source_inputs/{raw:03d}/raw_model.png')
        for name in ['raw_native', 'final_native']:
            rgb[name].update(coordinate_domain='native Brown distorted photo grid 1920x1280',
                             supervision_role='audit-only; not R9 teacher RGB')
        r8 = C/'road_vehicle_joint_dggt_20261123_r8/audit/mask_coordinate_audit/rectified_masks'/f'{raw:03d}'
        cache = C/'road_shared_appearance_20261122_r7/projection_cache_v1'/f'{raw:03d}'/'view_cache.npz'
        sources.append(dict(raw=raw, seed=101, prompt_id='legacy',
            prompt_text=next(x['text'] for x in plan['configuration']['prompts'] if x['id']=='legacy'),
            selected_before_R9_fusion=True, selection_policy='two explicitly accepted views; fixed seed101; no best-of-N',
            actual_model_wh=[1152,768], native_wh=[1920,1280], crop_native_xyxy_half_open=[0,0,1920,1280],
            timestamp_micros=obs['timestamp_micros'], **rgb,
            source_photo=pin(original_path, True), original_observation_json=pin(obs_path),
            prepared_observation=pin(case['observation']['path'], True),
            prepared_observation_decoded_exact_source_photo=True,
            raw_native_exact_PIL_BILINEAR_resize=True, final_outside_old_W_and_inside_old_K_exact_photo=True,
            old_native_roles=role_ref, old_model_roles={**pin(case['view']['roles_model']['path']),
                'coordinate_domain':'Brown full-frame 1152x768; full support resampled old C/S/W/K policy',
                'used_as_R9_visibility_or_permission':False},
            old_model_inputs={k:pin(case['view'][k]['path'], k!='construction') for k in ['I_geo','H_input','C_mat','construction']},
            camera_intrinsic_Brown=obs['camera_intrinsic_Brown'], camera_size=obs['camera_size'],
            camera_pose_world=obs['camera_pose_world'], camera_extrinsic=obs['camera_extrinsic'],
            projection_cache={**pin(cache), 'role':'independent geometric support and Brown-to-pinhole preview map; old requested_C/S/K not policy'},
            R8_pinhole_protection_reference={'path':(r8/'rectified_masks.npz').as_posix(),
                'exists_at_manifest_build':(r8/'rectified_masks.npz').exists(),
                'local_path':f'{a.local_workspace}/work/road_vehicle_joint_dggt_20261123_r8/audit/mask_coordinate_audit/rectified_masks/{raw:03d}/rectified_masks.npz',
                'role':'new pinhole protected mask; independently bind before using'},
            actual_road_semantic_and_visual_qualification='pending separate R9 audit; source SHA binding is not road qualification',
            composition_receipt=pin(run/'composition.json')))
    out = dict(schema='R9.old_raw_teacher_sources.v1', complete=True,
        scope='source identity, coordinate domains, and deterministic historical resize/composite replay',
        sources=sources, teacher_view_count=2, seed_count_per_view=1,
        legacy_plan=pin(plan_path), legacy_model=plan['model'], checkpoint=plan['checkpoint'],
        inference=plan['configuration']['inference'],
        sampling_contract={
            'world_to_source':'actual world XYZ -> independently bound world_to_camera -> Brown distortion -> native pixel-index UV -> model pixel-index UV',
            'native_pixel_index_to_model':'(native_uv + 0.5) * (1152/1920,768/1280) - 0.5',
            'RGB_interpolation':'sRGB EOTF per original raw_model texel BEFORE bilinear interpolation; do not interpolate encoded sRGB',
            'raw_native_not_resampled_for_supervision':True,
            'geometric_visibility':'R7 atlas_uv_map four-neighbour support plus same-world consistency certificate; no old C/S/W reuse',
            'new_protection':'R8 rectified pinhole protected mask; separate metadata; not old Brown K',
            'world_scope':'R7 write_domain is only initial world-material scope; no parent RGB supervision',
            'pinhole_preview':'separate warp using cached photo_source_brown_uv, not proof raw_model was generated pinhole'},
        CPU_only=True, new_GPU_forward=0, new_diffusion_calls=0, training_updates=0,
        elapsed_seconds=time.time()-start)
    audit=a.output_root/'audit'; audit.mkdir(parents=True,exist_ok=True)
    target=audit/'raw_teacher_sources.json'
    assert not target.exists(), f'will not overwrite {target}'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'manifest':pin(target),'source_raw_model_sha256':{x['raw']:x['raw_model']['sha256'] for x in sources}}))


if __name__=='__main__':
    main()
