"""Copy only fixed shared eligibility/masks into the independent C output."""
from pathlib import Path
import shutil
from shared_fit_common import RUN,OUT,sha,write


def main():
    target=RUN/'engineer/material_fit_edgeblend_v1';target.mkdir(exist_ok=False)
    copied={}
    for name in ('qualified_parents.npz','shared_contract.json','supervision_masks.npz'):
        shutil.copy2(OUT/name,target/name);assert sha(target/name)==sha(OUT/name);copied[name]=sha(target/name)
    write(target/'reused_qualification.json',dict(source=str(OUT),copied_sha256=copied,new_GPU_qualification_forwards=0,
        original_AB_sources_and_outputs_modified=False,initial_material='same untouched canonical cellfit1000 parent colours'))
    print(target)


if __name__=='__main__':main()
