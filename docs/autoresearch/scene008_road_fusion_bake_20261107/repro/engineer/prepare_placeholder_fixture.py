"""One authorized temporary raw-RGB lifecycle fixture; no scientific fitting."""
from pathlib import Path
import shutil
import numpy as np
from shared_fit_common import RUN,OUT,BUILD,load,read,write,sha
from fusion_colour_runtime import PARENT


def main():
    target=RUN/'engineer/smoke_engineering_placeholder';target.mkdir(exist_ok=False)
    for name in ('qualified_parents.npz','shared_contract.json','supervision_masks.npz'):shutil.copy2(OUT/name,target/name)
    masks=load(OUT/'supervision_masks.npz');manifest=read(PARENT/'scene_manifest.json');contract=read(manifest['v44_contract_path'])
    observations=Path(contract['paths']['observations']);folder=target/'targets';folder.mkdir();inputs={}
    for raw in BUILD:
        src=observations/f'{raw:03d}.npz';data=load(src);rgb=data['rgb'].astype(np.float32)
        if rgb.max()>1:rgb/=255
        mask=masks[f'raw{raw:03d}'].astype(bool);protect=data['protect'].astype(bool)&~data['visible'].astype(bool).any(0)&~mask
        np.savez_compressed(folder/f'raw{raw:03d}.npz',direct_rgb=rgb,fusion_rgb=rgb,observation_rgb=rgb,
            fit_mask=mask,fusion_replace_mask=np.zeros_like(mask),protect_mask=protect,protect_rgb=rgb)
        inputs[str(src)]=sha(src)
    write(target/'placeholder_target_manifest.json',dict(fusion_ready=False,status='ENGINEERING_RAW_RGB_FIXTURE_NOT_SCIENTIFIC_TARGET',
        temporary=True,formal_forbidden=True,observations_sha256=inputs,source_sha256=sha(__file__)))
    print(target)


if __name__=='__main__':main()
