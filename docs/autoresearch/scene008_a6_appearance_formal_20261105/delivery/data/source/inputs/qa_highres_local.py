"""Verify retrieved HR preview/sample files and the unchanged512 mapping locally."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE/'hr_qa/inputs_hr'
LOW=HERE/'collected'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'file_manifest.json').read_text())
    checked=0;operators_remote=[]
    for f in manifest['files']:
        path=ROOT/f['path']
        if path.name in ('design_A_normalized.npz','design_W_premultiplied.npz'):
            operators_remote.append(f['path']);continue
        assert path.exists(),path
        assert sha(path)==f['sha256'],path
        checked+=1
    frames=[]
    for raw in (106,114,129):
        per=ROOT/f'raw{raw:03d}';low=LOW/f'raw{raw:03d}'
        receipt=json.loads((per/'receipt.json').read_text())
        real=np.array(Image.open(per/'real_rgb_hr.png'))
        known_rgb=np.array(Image.open(per/'known_rgb_hr.png'))
        assert real.shape==known_rgb.shape==(1280,1920,3)
        assert sha(per/'real_rgb_hr.png')==receipt['source_rgb_sha256']
        with np.load(per/'raster_hr.npz') as r:
            known=r['known_mask_conservative'];visible=r['visible_mask_lifted'];support=r['object_mask'];unknown=r['inpaint_mask']
            assert np.array_equal(support,r['alpha']>.1)
            assert np.array_equal(unknown,support&~visible)
            assert not np.any(known&~visible)
            assert np.array_equal(known_rgb[known],real[known])
            assert not np.any(known_rgb[~known])
        with np.load(per/'operator_samples.npz') as s,np.load(per/'transform.npz') as t,np.load(low/'transform.npz') as old:
            pixels=s['pixels'];xy=s['pixel_xy'];generated=s['generated_xy']
            assert np.array_equal(pixels,np.flatnonzero(support))
            assert np.array_equal(xy,np.column_stack((pixels%1920,pixels//1920)))
            lowxy=(xy+.5)/4-.5
            expected=(np.column_stack((lowxy,np.ones(len(xy))))@old['original_to_generated_pixel_center'].T)[:,:2]
            mapping_error=float(np.max(np.abs(expected-generated)))
            assert mapping_error<1e-10
            assert np.array_equal(s['real_rgb_uint8'],real.reshape(-1,3)[pixels])
            assert np.array_equal(s['real_rgb'],real.reshape(-1,3)[pixels].astype(np.float32)/255)
            assert np.array_equal(s['fit_alpha_gt_09'],s['alpha']>.9)
            assert np.array_equal(s['fit_observed'],s['fit_alpha_gt_09']&known.ravel()[pixels])
            assert np.array_equal(s['fit_generated'],s['fit_alpha_gt_09']&s['generated_inpaint_nearest'])
            affine_error=float(np.max(np.abs(t['highres_to_generated_pixel_center']@t['generated_to_highres_pixel_center']-np.eye(3))))
        assert sha(low/'input_512.png')==receipt['same_original_proposal_input_sha256']
        assert sha(low/'inpaint_mask_512.png')==receipt['same_original_proposal_mask_sha256']
        frames.append(dict(raw=raw,status='PASS',operator_shape=receipt['shape'],
            samples=len(pixels),map_hr_low_512_maxabs=mapping_error,affine_roundtrip_maxabs=affine_error,
            known_hr_rgb_exact=True,known_mask_is_conservative=True,proposal512_unchanged=True,
            source_HR_rgb_sha256_verified=True))
    report=dict(status='PASS',local_files_sha256_verified=checked,
        six_sparse_operators_retained_remote=operators_remote,frames=frames)
    (HERE/'hr_qa/local_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
