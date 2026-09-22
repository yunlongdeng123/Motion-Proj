import collections, gzip, json, os, pathlib, re, stat, time

BASE = pathlib.Path('/root/autodl-tmp')
RUNS = BASE / 'runs'
REPO = BASE / 'motion_proj'
OUT = BASE / 'cleanup_manifests/20260922-old-runs-retire'
OUT.mkdir(parents=True, exist_ok=True)
TEXT = {'.json','.jsonl','.yaml','.yml','.csv','.tsv','.txt','.md','.log','.py','.sh','.toml','.ini','.cfg','.conf','.xml','.html','.css','.js','.sql','.sqlite','.sqlite3','.db','.ipynb','.env','.rst','.tex','.bib','.patch','.diff','.out','.err','.exitcode','.code','.status'}
roots = sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink() and p.name not in {'worldsim_v75','_workers'})
stats = {}
symlinks = []
large_text = []
with gzip.open(OUT/'inventory.jsonl.gz','wt') as inv:
    for root in roots:
        row={'files':0,'allocated':0,'logical':0,'text_allocated':0,'by_suffix':{},'largest':[]}
        for directory, dirs, files in os.walk(root,followlinks=False):
            for name in dirs+files:
                p=pathlib.Path(directory)/name
                s=p.lstat()
                if stat.S_ISLNK(s.st_mode):
                    symlinks.append({'path':str(p),'target':os.readlink(p),'resolved':str(p.resolve())})
                    continue
                if not stat.S_ISREG(s.st_mode):continue
                suffix=p.suffix.lower()
                item={'path':str(p),'root':root.name,'size':s.st_size,'allocated':s.st_blocks*512,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'suffix':suffix,'text':suffix in TEXT}
                inv.write(json.dumps(item)+'\n')
                row['files']+=1; row['allocated']+=item['allocated']; row['logical']+=s.st_size
                part=row['by_suffix'].setdefault(suffix,{'files':0,'allocated':0})
                part['files']+=1; part['allocated']+=item['allocated']
                if item['text']:
                    row['text_allocated']+=item['allocated']
                    if s.st_size>16*1024**2:large_text.append({'path':str(p),'size':s.st_size})
                row['largest'].append({'path':str(p),'size':s.st_size})
                row['largest']=sorted(row['largest'],key=lambda x:x['size'],reverse=True)[:12]
        stats[root.name]=row
        print(root.name,row['files'],round(row['allocated']/1024**3,3),flush=True)

# Preserve a complete metadata snapshot of current V7.5, without touching contents.
with gzip.open(OUT/'v75_before.jsonl.gz','wt') as f:
    for directory,dirs,files in os.walk(RUNS/'worldsim_v75',followlinks=False):
        for name in dirs+files:
            p=pathlib.Path(directory)/name;s=p.lstat()
            if not(stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode)):continue
            f.write(json.dumps({'path':str(p),'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':s.st_mode,'target':os.readlink(p) if p.is_symlink() else None})+'\n')

# Search current V7.5 code, protocols, run metadata and symlinks for old-run references.
rx=re.compile(r'(?:/root/autodl-tmp/)?runs/([A-Za-z0-9_.-]+)(?:/[^\s\"\x27<>;,\)\]\}]+)?')
refs=[];links=[]
scan_roots=[RUNS/'worldsim_v75',REPO/'scripts/worldsim_v75',REPO/'docs/v75',REPO/'docs/autoresearch/worldsim_v75']
for root in scan_roots:
    if not root.exists():continue
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=pathlib.Path(directory)/name
            if p.is_symlink():
                resolved=str(p.resolve())
                if resolved.startswith(str(RUNS)+'/') and not resolved.startswith(str(RUNS/'worldsim_v75')+'/'):
                    links.append({'source':str(p),'target':resolved})
            if p.is_symlink() or not p.is_file() or p.suffix.lower() not in TEXT or p.stat().st_size>32*1024**2:continue
            try:
                for no,line in enumerate(p.open(errors='replace'),1):
                    for m in rx.finditer(line):
                        if m.group(1)!='worldsim_v75':refs.append({'source':str(p),'line':no,'target':m.group(0)})
            except OSError:pass
procs=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        comm=(p/'comm').read_text().strip()
        cwd=os.readlink(p/'cwd')
        if cwd.startswith(str(RUNS)) or comm in {'python','python3','torchrun','aria2c'}:
            procs.append({'pid':p.name,'name':comm,'cwd':cwd})
    except OSError:pass
disk=os.statvfs(BASE)
result={'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'roots':stats,'large_text':large_text,'current_refs':refs,'current_old_symlinks':links,'processes':procs,'free_bytes':disk.f_bavail*disk.f_frsize,'root_symlink_count':len(symlinks)}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with gzip.open(OUT/'symlinks.json.gz','wt') as f:json.dump(symlinks,f)
print(json.dumps({'done':True,'root_count':len(stats),'current_refs':len(refs),'current_old_symlinks':links,'processes':procs,'large_text_count':len(large_text)},ensure_ascii=False),flush=True)
