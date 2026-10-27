"""Lossless relocation of owned completed smoke checkpoints, original paths kept.

No checkpoint is discarded. Only the first already-completed scene is eligible;
active/formal checkpoints and all evaluation data remain untouched.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import datetime,gc,hashlib,json,shutil
from pathlib import Path
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
TARGET=Path('/root/persistent_world_checkpoint_spillover_20261027')
CONTEXT='14705303724557273004_3105_000_3125_000'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    scene=ROOT/'native_scene_training_v1'/CONTEXT
    assert json.loads((scene/'formal30k/returned.json').read_text())['iteration']==30000
    assert json.loads((scene/'formal30k_agent_acceptance.json').read_text())['formal_native_updates']==30000
    files=sorted([*(scene/'smoke_fresh').glob('*.pth'),*(scene/'smoke_restore').glob('*.pth')]);assert files
    for path in files:
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(scene.resolve())
        assert path.parent.name in ['smoke_fresh','smoke_restore']
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(str(path.parent) in command for path in files),'An active command still uses this smoke run'
    TARGET.mkdir(exist_ok=False)
    assert TARGET.resolve()==TARGET and TARGET.stat().st_dev!=ROOT.stat().st_dev
    total=sum(path.stat().st_size for path in files);assert shutil.disk_usage(TARGET).free-total>=2*2**30
    folder=ROOT/'checkpoint_storage_relocation_v1';folder.mkdir(exist_ok=False)
    receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='copying_verified_owned_smokes',source_scene=CONTEXT,target_root=str(TARGET),expected_bytes=total,source_free_before=shutil.disk_usage(ROOT).free,target_free_before=shutil.disk_usage(TARGET).free,files=[],checkpoint_content_discarded=False,formal_or_active_checkpoints_moved=False)
    def write():
        p=folder/'receipt.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(receipt,indent=2));q.replace(p)
    write()
    for source in files:
        target=TARGET/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        sha=digest(source);original=source.stat();temporary=target.with_suffix('.copying')
        shutil.copy2(source,temporary)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        assert temporary.stat().st_size==original.st_size and digest(temporary)==sha
        temporary.replace(target)
        link=source.with_suffix('.relocating_link');assert not link.exists();link.symlink_to(target);os.replace(link,source)
        assert source.is_symlink() and source.resolve()==target and source.stat().st_mtime_ns==original.st_mtime_ns
        value=torch.load(str(source),map_location='cpu',mmap=True);assert value is not None;del value;gc.collect()
        receipt['files'].append(dict(original_path=str(source),physical_path=str(target),bytes=original.st_size,sha256=sha,original_path_cpu_readable=True));write()
    receipt.update(phase='complete_content_preserved_original_paths_readable',source_free_after=shutil.disk_usage(ROOT).free,target_free_after=shutil.disk_usage(TARGET).free,actual_relocated_bytes=total)
    write();print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))

if __name__=='__main__':main()
