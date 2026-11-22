"""Copy the immutable selected original A20 r3 asset into the R10 payload without re-exporting it."""
from pathlib import Path
import argparse,hashlib,json,shutil

EXPECTED='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_joint_reconstruction_20261122_r1/execution_r3/post_geometry_ablation/assets/geometry_ablation/actor.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert sha(SOURCE)==EXPECTED;target=a.root/'provenance/original_A20_r3/actor.npz';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==EXPECTED
    else:shutil.copyfile(SOURCE,target)
    assert sha(target)==sha(SOURCE)==EXPECTED
    r=dict(schema='R10.original_selected_A20_r3_exact_backup.v1',complete=True,decision='COPY_VERIFIED',source_path=str(SOURCE),source_sha256=EXPECTED,
        archive_relative_path=a.root.name+'/provenance/original_A20_r3/actor.npz',run_relative_path='provenance/original_A20_r3/actor.npz',
        copied_bytes=target.stat().st_size,copy_sha256=sha(target),copy_method='raw file copy; no NPZ load/export or field recomputation',
        runtime_parent_contracts_unchanged=True,original_actor_file_unchanged=True,selected_original_vehicle_not_the_historical_400step_DC_variant=True,
        restore_path_map=[dict(original_path=str(SOURCE),archive_member=a.root.name+'/provenance/original_A20_r3/actor.npz',sha256=EXPECTED)],
        original_native_and_DGGT_large_checkpoints_still_external=True,GPU_calls=0,updates=0)
    (target.parent/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
