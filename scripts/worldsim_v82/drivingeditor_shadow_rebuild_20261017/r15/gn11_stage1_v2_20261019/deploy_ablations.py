"""仅在新目录安装三个消融；核对主实验源码保持原字节。"""
from pathlib import Path,PurePosixPath
import argparse,ast,datetime,hashlib,json,shutil,zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main(C,archive):
    group=C/'ablations'
    if group.exists():raise FileExistsError('new isolated ablation directory required')
    main_manifest=read(C/'source_bundle/candidate_manifest.json')['files']
    before={rel:sha(C/'engine'/rel) for rel in main_manifest}
    if any(before[rel]!=entry['candidate_sha256'] for rel,entry in main_manifest.items()):
        raise RuntimeError('main source differs before installation')
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            p=PurePosixPath(entry.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in entry.filename:raise ValueError('unsafe source archive')
        if sum(e.file_size for e in z.infolist())>5e6:raise ValueError('source archive unexpectedly large')
        group.mkdir();prepared=group/'prepared_source';z.extractall(prepared)
    if read(prepared/'cpu_contract_receipt.json')['status']!='PASSED_CPU_ABLATION_SOURCE_AND_SLOT_CONTRACTS':
        raise RuntimeError('CPU source contracts not passed')
    source=group/'source_bundle';source.mkdir()
    for name in ['run_ablation.py','ablation_support.py','run_road_ablation_group.py']:
        ast.parse((prepared/name).read_text());shutil.copyfile(prepared/name,source/name)
    for name in ['plan.json','cpu_contract_receipt.json']:shutil.copyfile(prepared/name,group/name)
    for name in ['audit_v2_checkpoint.py','evaluate_v2_full49.py']:
        text=(C/'source_bundle'/name).read_text()
        if name.startswith('evaluate_'):
            old="lock=(C.parent/'gpu_slot.lock').open('a+')"
            assert text.count(old)==1
            text=text.replace(old,"lock=Path('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock').open('a+')")
        ast.parse(text);(source/name).write_text(text)
    arms={}
    for name in ['real_only','prior_only','ownership_only']:
        root=group/name;root.mkdir();engine=root/'engine';engine.mkdir();ownsource=root/'source_bundle';ownsource.mkdir()
        for original in (C/'engine').iterdir():
            if original.name in ['__pycache__','Tracking-Anything-with-DEVA','scripts','tests']:continue
            if original.is_dir():shutil.copytree(original,engine/original.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            else:shutil.copyfile(original,engine/original.name)
        overlay=prepared/name/'candidate'
        for p in overlay.rglob('*'):
            if p.is_file():
                dest=engine/p.relative_to(overlay);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest=read(prepared/name/'candidate_manifest.json')
        if any(sha(engine/rel)!=item['candidate_sha256'] for rel,item in manifest['files'].items()):
            raise RuntimeError('installed arm differs: '+name)
        shutil.copyfile(prepared/name/'candidate_manifest.json',ownsource/'candidate_manifest.json')
        arms[name]=dict(engine=str(engine),road_weights=manifest['road_weights'],trained_updates=0)
    after={rel:sha(C/'engine'/rel) for rel in main_manifest}
    if before!=after:raise RuntimeError('main source was modified')
    result=dict(status='THREE_ISOLATED_ARMS_INSTALLED_NOT_YET_STARTED',arms=arms,
        main_source_unchanged=True,max_concurrent_training_including_main=2,
        archive_sha256=sha(archive),installed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (group/'deployment_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();main(a.campaign,a.archive)
