"""Small reproducible Git bundle; raw floats stay in the verified raw archive."""
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
work = Path(__file__).parent
design = root / 'outputs/V43_Ownership_20261027'
report = design / 'gpu_v43/editable_ownership_20261030'
stage = work / 'publication'
code = stage / 'scripts/worldsim_v82/v43_ownership_20261028/editable_ownership_20261030'
docs = stage / 'docs/autoresearch/v43_ownership_20261028/editable_ownership_20261030'
code.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
for p in work.glob('*.py'):
    shutil.copy2(p, code / p.name)
shutil.copy2(work/'README.md',code/'README.md')
for p in [work / 'authorization.json', work / 'split_math_validation.json']:
    shutil.copy2(p, code / p.name)
for name in ['build_architecture.py','build_v43.py']:
    shutil.copy2(root / 'work/v43_ownership_20261027' / name, code / name)
for p in report.rglob('*'):
    if p.is_file() and p.suffix in {'.html','.png','.json','.md'}:
        out = docs / p.relative_to(report); out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
for name in ['architecture.svg','architecture.png','V43_DESIGN.md','ownership_record_contract.json','direction_state.json','design_revision_validation.json']:
    shutil.copy2(design / name, docs / name)
locations = set()
for p in docs.rglob('*'):
    if p.suffix not in {'.html','.md'}: continue
    content = p.read_text(encoding='utf8')
    if p.suffix == '.html':
        content = content.replace('../../architecture.svg','architecture.svg')
        def html_link(m):
            url=m.group(2)
            if url.startswith(('http:','https:','#','data:')): return m.group(0)
            if (p.parent / url.split('#')[0]).exists(): return m.group(0)
            if url in ['../r2_recovery_20261029/index.html','../unknown_white_20261030/index.html']: return m.group(0)
            locations.add(url)
            return m.group(1)+'LOCAL_WORKSPACE_REFERENCES.md'+m.group(3)
        content=re.sub(r'((?:href|src)=")([^"]+)(")',html_link,content)
        # JS-generated float links are explicitly routed to raw source locator in this Git copy.
        content=content.replace('a.href=`probe_anchors_r1/${p}/${f}`;', 'a.href=f.endsWith(".npz")?"LOCAL_WORKSPACE_REFERENCES.md":`probe_anchors_r1/${p}/${f}`;')
    else:
        content=content.replace('../../architecture.svg','architecture.svg')
        def md_link(m):
            url=m.group(2)
            if url.startswith(('http:','https:','#')) or (p.parent / url.split('#')[0]).exists():return m.group(0)
            locations.add(url); return m.group(1)+'LOCAL_WORKSPACE_REFERENCES.md'+m.group(3)
        content=re.sub(r'(\]\()([^\)]+)(\))',md_link,content)
    p.write_text(content,encoding='utf8')
archive=work/'raw_delivery.tar.gz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)=='9fdba0899d3c4d8ee06029b9bf950ad92add454890f1e6859f976c02788db837'
receipt=dict(raw_archive_sha256=sha(archive),raw_archive_bytes=archive.stat().st_size,
    local_archive=str(archive),remote_archive='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/editable_ownership_20261030_raw.tar.gz',
    child_refit_updates_total=200,classifier_training_updates=0,original_model_training_updates=0,raw_transfer_verified=True)
(docs/'raw_archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
(report/'raw_archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
(docs/'LOCAL_WORKSPACE_REFERENCES.md').write_text('# 原始产物定位\n\n本Git副本保存原生PNG、配置、摘要、设计和可复现源码；完整浮点、优化状态和旧失败现场另存已校验归档。原工作台及本地报告保持完整数据链接。\n\n'+json.dumps(receipt,ensure_ascii=False,indent=2)+'\n\n原工作台：'+str(root/'outputs/GS_Scene_Diagnostics_20261027/index.html')+'\n\n原设计：'+str(design/'V43_DESIGN.md')+'\n\n本地报告：'+str(report/'index.html')+'\n\n非本包历史/浮点引用：\n\n'+'\n'.join('- '+x for x in sorted(locations)),encoding='utf8')
files=[]
for p in stage.rglob('*'):
    if p.is_file():
        if p.suffix in {'.py','.json','.md','.html','.svg'}:
            data=p.read_bytes().replace(b'\r\n',b'\n')
            data=b'\n'.join(line.rstrip(b' \t') for line in data.split(b'\n'))
            p.write_bytes(data)
        files.append(dict(path=p.relative_to(stage).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
(work/'publication_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf8')
with tarfile.open(work/'publication.tar.gz','w:gz') as tar:
    for f in files: tar.add(stage/f['path'],arcname=f['path'])
print(json.dumps(dict(files=len(files),bytes=sum(f['bytes'] for f in files),raw_archive_verified=True)))
