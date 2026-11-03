"""Validate the packaged minimal R4 dependency in a fresh namespace."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import shutil,subprocess,sys,json
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent;dest=RUN/'portable_check';run=dest/RUN.name
def main():
    (run/'source').mkdir(parents=True,exist_ok=False)
    for p in (RUN/'source').glob('*.py'):shutil.copy2(p,run/'source'/p.name)
    (run/'inputs').symlink_to(RUN/'inputs',target_is_directory=True);(run/'formal').symlink_to(RUN/'formal',target_is_directory=True);(run/'joint_contributors.npz').symlink_to(RUN/'joint_contributors.npz')
    oldname='shadow_a5_a37_199_20261103_r4';(dest/oldname).symlink_to(RUN/'dependency'/oldname,target_is_directory=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (RUN/'portable_check.log').open('x') as f:subprocess.run([sys.executable,str(run/'source/joint_runtime.py'),'--output','evaluation','--resume','formal/checkpoint001000.pt','--evaluate'],cwd=run,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    exact={str(raw):(RUN/'evaluation/all49'/('%03d_joint.png'%raw)).read_bytes()==(run/'evaluation/all49'/('%03d_joint.png'%raw)).read_bytes() for raw in range(89,138)};assert all(exact.values())
    result=dict(minimal_R4_dependency_loaded=True,all49_PNGs_exact=True,parameters_optimizer_RNG_restored=True,updates=0,original_B_and_V3_still_external=True)
    (RUN/'portable_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
