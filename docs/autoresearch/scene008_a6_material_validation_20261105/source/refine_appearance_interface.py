"""Make model inputs unit-explicit and each pseudo case physically isolated."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent.parent
SOURCE=RUN/'appearance_inputs';DEST=RUN/'appearance_inputs_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    DEST.mkdir(exist_ok=False);base=json.loads((SOURCE/'contract.json').read_text());manifests={};changes=[]
    shutil.copyfile(SOURCE/'geometry_only.npz',DEST/'geometry_only.npz')
    for case,records in base['cases'].items():
        folder=DEST/case;folder.mkdir();newrecords=[]
        for row in records:
            raw=row['raw'];original=SOURCE/row['npz'];a=dict(np.load(original))
            linear=a['model_from_local'][:3,:3].astype(np.float64);scale=float(np.linalg.norm(linear,axis=0).mean())
            assert abs(scale-.1)<1e-7 and np.allclose(linear.T@linear,np.eye(3)*scale**2,atol=2e-8)
            model_camera=a['model_to_camera'].astype(np.float64)@a['model_from_local'].astype(np.float64)
            metric_camera=model_camera.copy();metric_camera[:3,:4]/=scale
            assert np.allclose(metric_camera[:3,:3].T@metric_camera[:3,:3],np.eye(3),atol=2e-6)
            K=a['camera_K'].copy();K[0,2]-=a['crop_xyxy'][0];K[1,2]-=a['crop_xyxy'][1]
            a.update(meter_to_model=np.asarray(scale),camera_from_local_m=metric_camera,camera_K_crop=K,
                full_image_hw=np.asarray([320,480]),RGB_unknown_mask=~a['RGB_known_mask'])
            dest=folder/f'{raw:03d}.npz';np.savez_compressed(dest,**a)
            b=dict(np.load(dest));old=dict(np.load(original))
            assert all(np.array_equal(b[k],old[k]) for k in old)
            assert not np.any(b['known_RGB_srgb'][b['RGB_unknown_mask']])
            shutil.copyfile(SOURCE/case/f'{raw:03d}_preview.png',folder/f'{raw:03d}_preview.png')
            newrecords.append(dict(raw=raw,known_pixels=row['known_pixels'],npz=f'{raw:03d}.npz',sha256=sha(dest)))
            changes.append(dict(case=case,raw=raw,source_sha256=sha(original),v2_sha256=sha(dest),original_arrays_bitwise_preserved=True))
        own=dict(schema='vehicle_appearance_input.v2',case=case,observed_camera_views=40,camera_stream='Same40 FRONT frames; not160 independent camera views',
            records=newrecords,known_pixels=base['known_pixels'][case],raw_ids=base['raw_ids'],
            authoritative_RGB='float32 sRGB known_RGB_srgb +RGB_known_mask. Unknown has placeholder0, never black/background target.',
            RGBA='Preview only:8bit quantized RGB, alpha0 meansunknown. Model adapters must consume an explicit mask.',
            geometry='geometry_only.npz is a fixed full M2/PVD prior, object-local metres; no fitted RGB/SH or hidden texture.',
            camera='camera_K uses full480x320 coordinates; camera_K_crop uses crop coordinates; image pixels follow native calibration convention.',
            pose='camera_from_local_m maps local metre XYZ directly into camera metre XYZ. model_from_local/model_to_camera preserve native model-coordinate insertion; meter_to_model explicitly supplied.',
            depth='prior_expected_depth_m is expected prior camera-z metres, not a measurement or sensor ground truth.',
            input_isolation='This standalone case archive contains only this case40views and common geometry. No full sibling or other pseudo-case RGB is admitted.',
            forbidden=['HELD RGB','Masked RGB','Full-fit M2 material','Other case/full sibling RGB','A37/road RGB'],
            source_contract_sha256=sha(SOURCE/'contract.json'),geometry_sha256=sha(DEST/'geometry_only.npz'))
        write(folder/'case_contract.json',own);manifests[case]=dict(contract_sha256=sha(folder/'case_contract.json'),views=40,known_pixels=own['known_pixels'])
    write(DEST/'contract.json',dict(status='PASS_UNIT_EXPLICIT_CASE_ISOLATED_INTERFACE',case_manifests=manifests,
        source=base['source'],source_contract_sha256=sha(SOURCE/'contract.json'),changes=changes,
        original_RGB_and_mask_and_geometry_bitwise_unchanged=True,
        hidden_colour_labels_exported=False,fitted_material_exported=False,model_training_steps=0))
    print(json.dumps(dict(status='PASS',cases=manifests,views=160,unique_camera_views=40)))
if __name__=='__main__':main()
