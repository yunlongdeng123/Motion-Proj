import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'codex_tmp/horizon_waymo_inventory_20261031_r1'
OUT.mkdir(parents=True,exist_ok=False)
TARGETS=[4,20,29,35,65,67,79,81,86,113,140]
REGEX=re.compile(r'(?:segment-)?(\d+_\d+_\d+_\d+_\d+)(?:_with_camera_labels)?')

def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def rg_files(roots,patterns):
    args=['rg','--files']+[str(p) for p in roots if p.exists()]
    for pattern in patterns:args+=['-g',pattern]
    result=subprocess.run(args,text=True,capture_output=True)
    return result.stdout.splitlines(),dict(command=args,returncode=result.returncode,stderr=result.stderr)

started=time.monotonic()
roots=[ROOT/'external',ROOT/'third_party',ROOT/'street_crafter',ROOT/'dggt',ROOT/'datasets']
paths,search=rg_files(roots,['*waymo*list*.txt','*waymo*list*.csv','*waymo*list*.json','*waymo*example*.txt','*waymo*split*.txt'])
for p in (ROOT/'waymo_train_list.txt',ROOT/'waymo_val_list.txt'):
    if p.is_file():paths.append(str(p))
sources=[]
for name in sorted(set(paths)):
    path=Path(name);text=path.read_text(errors='replace');lines=text.splitlines()
    entries=[]
    for number,line in enumerate(lines,1):
        match=REGEX.search(line)
        if match:entries.append(dict(source_line=number,raw=line,context_id=match.group(1)))
    digest=hashlib.sha256(text.encode()).hexdigest()
    source=dict(path=name,sha256=digest,line_count=len(lines),context_entry_count=len(entries),
        ordering='file_order_not_resorted',zero_based_candidate_rows=[dict(scene_number=f'{i:03}',index=i,
            **entries[i]) for i in TARGETS if i<len(entries)],
        horizonforge_identity='unverified_candidate_mapping_only')
    sources.append(source)
    (OUT/('list_'+digest[:12]+'.txt')).write_text(text)
write('list_sources_and_mapping.json',dict(target_numbers=[f'{i:03}' for i in TARGETS],sources=sources,search=search))

inventory_roots=[ROOT/'datasets',ROOT/'dggt/data',ROOT/'drivingeditor_repro/data',ROOT/'street_crafter',
    ROOT/'streetcrafter_work',ROOT/'retained',ROOT/'data/downloads']
raw,raw_search=rg_files(inventory_roots,['*.tfrecord','*.tfrecords','*.parquet'])
raw_files=[]
for name in raw:
    path=Path(name);match=REGEX.search(name)
    raw_files.append(dict(path=name,bytes=path.stat().st_size,context_id=match.group(1) if match else None,
        suffix=path.suffix))
write('raw_file_inventory.json',dict(search=raw_search,files=raw_files,
    note='filename/stat inventory only; payloads not read; non-Waymo parquet may be present'))

entries=[]
for base in [ROOT/'datasets',ROOT/'datasets/waymo_v1_4_3',ROOT/'data',ROOT/'dggt',ROOT/'dggt/data',
             ROOT/'drivingeditor_repro',ROOT/'street_crafter',ROOT/'streetcrafter_work',ROOT/'retained']:
    children=[]
    if base.exists():
        for p in sorted(base.iterdir()):
            children.append(dict(name=p.name,is_dir=p.is_dir(),is_symlink=p.is_symlink(),
                resolves_to=str(p.resolve()) if p.is_symlink() else None))
    entries.append(dict(path=str(base),exists=base.exists(),children=children))
write('root_layout.json',entries)
print(json.dumps(dict(output=str(OUT),list_sources=[dict(path=s['path'],count=s['context_entry_count'],hash=s['sha256'][:12]) for s in sources],
    raw_files=len(raw_files),waymo_named_raw_files=sum(r['context_id'] is not None for r in raw_files),wall_seconds=time.monotonic()-started),ensure_ascii=False))
