"""打包新实验现场，保留选定PVD权重；不递归解引用外部模型链接。"""
import hashlib
import json
from pathlib import Path
import tarfile
import time

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
ARCHIVE=Path('/root/autodl-tmp/codex_tmp/scene008_a37_C2_20261103_run.tar.gz')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    assert not ARCHIVE.exists(),ARCHIVE
    assert (RUN/'git_delivery_receipt.json').exists()
    assert (RUN/'report/report_verification.json').exists()
    assert json.loads((RUN/'formal/receipt.json').read_text())['step']==1000
    entries=[];links=[];excluded=[]
    for p in sorted(RUN.rglob('*')):
        rel=p.relative_to(RUN)
        if any(x in rel.parts for x in ('.git','__pycache__')) or p.name.endswith(('.tar.gz','.zip','.pyc','.aria2')):
            excluded.append(str(rel));continue
        if p.is_symlink():links.append(dict(path=str(rel),target=str(p.readlink())))
        elif p.is_file():entries.append(dict(path=str(rel),bytes=p.stat().st_size))
    manifest=dict(schema='C2 full new-run snapshot',regular_files=len(entries),regular_bytes=sum(x['bytes'] for x in entries),
        files=entries,external_links_preserved=links,excluded_redundant_archives_or_caches=excluded,
        selected_PVD_pretrained_weight_included=True,Omni_14GB_weights='retained remote model cache; failed proposals and pinned download receipts included',
        parent_dependencies=['scene008_a37_only_B10k_20261103_r1/formal10k','scene008_a37_only_B10k_road_20261103_r3/inputs and formal',
            'original native DrivingEditor CUDA environment and coordinate metadata'],
        local_parent_backups='Existing verified a37_B10k_replay.tar.gz and a37_B10k_road_replay.tar.gz in 2026-11-03/bug-a5-1k-mae-b-a5-2 outputs')
    (RUN/'snapshot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    started=time.monotonic();ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    def filt(info):
        rel=Path(info.name).relative_to('run')
        if any(x in rel.parts for x in ('.git','__pycache__')) or info.name.endswith(('.tar.gz','.zip','.pyc','.aria2')):return None
        return info
    with tarfile.open(ARCHIVE,'w:gz',compresslevel=1,dereference=False) as archive:
        archive.add(RUN,arcname='run',filter=filt)
    result=dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=digest(ARCHIVE),
        regular_files=len(entries)+1,seconds=time.monotonic()-started,selected_PVD_weight_included=True)
    receipt=ARCHIVE.with_suffix(ARCHIVE.suffix+'.receipt.json')
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
