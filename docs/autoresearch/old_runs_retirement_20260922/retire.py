import collections, gzip, json, os, pathlib, re, stat, sys, time

BASE=pathlib.Path('/root/autodl-tmp')
RUNS=BASE/'runs'
OUT=BASE/'cleanup_manifests/20260922-old-runs-retire'
AUDIT=json.loads((OUT/'audit.json').read_text())
ROOTS={RUNS/name for name in AUDIT['roots']}
VISUAL={'.svg','.pdf','.gif','.mp4','.webm'}
REPORT=re.compile(r'(?:figure|fig\d|panel|report|review|badcase|preview|architecture|comparison|contact_sheet|storyboard)',re.I)

def load_inventory():
    with gzip.open(OUT/'inventory.jsonl.gz','rt') as f:
        for line in f:yield json.loads(line)

def validate_path(p):
    rel=p.relative_to(RUNS)
    assert len(rel.parts)>=2 and RUNS/rel.parts[0] in ROOTS
    assert rel.parts[0] not in {'worldsim_v75','_workers'}
    assert not p.is_symlink() and p.resolve()==p
    assert p.is_file()

def same(d):
    p=pathlib.Path(d['path']);validate_path(p);s=p.stat()
    assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(d['inode'],d['device'],d['size'],d['mtime_ns']),str(p)

def free():
    s=os.statvfs(BASE);return s.f_bavail*s.f_frsize

def current_check():
    n=0;bad=[]
    with gzip.open(OUT/'v75_before.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line);p=pathlib.Path(d['path'])
            try:
                s=p.lstat();ok=(s.st_ino,s.st_size,s.st_mtime_ns,s.st_mode)==(d['inode'],d['size'],d['mtime_ns'],d['mode'])
                if d['target'] is not None:ok=ok and os.readlink(p)==d['target']
            except OSError:ok=False
            n+=1
            if not ok:bad.append(str(p))
    return {'checked':n,'changed_or_missing':bad}

def active_handles():
    hits=[]
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        for sub in ['cwd','fd']:
            try:items=list((proc/sub).iterdir()) if sub=='fd' else [proc/sub]
            except OSError:continue
            for item in items:
                try:target=os.readlink(item)
                except OSError:continue
                if any(target==str(r) or target.startswith(str(r)+'/') for r in ROOTS):hits.append({'pid':proc.name,'handle':str(item),'target':target})
    return hits

def plan():
    protected=[]
    for line in (OUT/'outside_symlinks.tsv').read_text().splitlines():
        source,_,_=line.partition('\t')
        p=pathlib.Path(source);dest=p.resolve()
        if any(dest==r or r in dest.parents for r in ROOTS):
            protected.append({'source':source,'target':str(dest),'is_dir':dest.is_dir(),'exists':dest.exists()})
    assert AUDIT['current_refs']==[] and AUDIT['current_old_symlinks']==[]
    live=active_handles();assert not live,live
    totals={'delete':{'files':0,'allocated':0},'keep':{'files':0,'allocated':0}}
    groups={};reasons=collections.Counter();inodes={};outside_shared={}
    with gzip.open(OUT/'plan.jsonl.gz','wt') as f:
        for d in load_inventory():
            p=pathlib.Path(d['path']);why=None
            for link in protected:
                t=pathlib.Path(link['target'])
                if link['exists'] and (p==t or link['is_dir'] and t in p.parents):why='outside_symlink_dependency';break
            if not why and d['text']:why='metadata_config_metrics_log_or_source'
            if not why and d['suffix'] in VISUAL:why='light_report_visual'
            if not why and d['suffix'] in {'.png','.jpg','.jpeg','.webp','.tar','.gz','.zip'} and REPORT.search(str(p)):why='report_or_review_asset'
            if not why and d['suffix'] in {'.tar','.gz'} and ('source' in p.name.lower() or d['size']<1024**2):why='source_or_small_archive'
            if not why and not d['suffix'] and d['size']<=1024**2:why='small_extensionless_record'
            action='keep' if why else 'delete'
            d.update(action=action,reason=why or 'retired_large_research_artifact')
            f.write(json.dumps(d)+'\n')
            totals[action]['files']+=1;totals[action]['allocated']+=d['allocated']
            group=groups.setdefault(d['root'],{'delete_files':0,'delete_allocated':0,'keep_files':0,'keep_allocated':0})
            group[action+'_files']+=1;group[action+'_allocated']+=d['allocated']
            reasons[d['reason']]+=1
            key=(d['device'],d['inode'])
            x=inodes.setdefault(key,{'delete':0,'keep':0,'nlink':d['nlink'],'allocated':d['allocated']})
            x[action]+=1
    reclaim=sum(x['allocated'] for x in inodes.values() if x['delete']==x['nlink'] and not x['keep'])
    result={'authorization':'User explicitly authorized removing old runs with reports or low relevance to current V7.5; supersedes historical artifact retention rules for these retired outputs.','base_head':'8b5eafee','protected_symlinks':protected,'totals_logical_including_shared_links':totals,'groups':groups,'reasons':dict(reasons),'estimated_reclaim_unique_file_bytes':reclaim,'active_handles':live,'v75_check':current_check(),'free_bytes_before':free(),'failure_ledger_delta':'none','model_calls':0}
    (OUT/'plan_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'groups','protected_symlinks'}},ensure_ascii=False),flush=True)
    print('outside dependency links',len(protected),flush=True)
    for link in protected:print(json.dumps(link),flush=True)

def execute():
    assert not (OUT/'result.json').exists(),'already executed'
    summary=json.loads((OUT/'plan_summary.json').read_text())
    before=current_check();assert not before['changed_or_missing']
    live=active_handles();assert not live,live
    # Complete a fresh pre-delete path/stat validation, then unlink only listed files.
    with gzip.open(OUT/'plan.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line)
            if d['action']=='delete':same(d)
    baseline=free();counts=collections.Counter();removed=0
    with gzip.open(OUT/'deleted.jsonl.gz','wt') as receipt:
        with gzip.open(OUT/'plan.jsonl.gz','rt') as f:
            for line in f:
                d=json.loads(line)
                if d['action']!='delete':continue
                same(d);pathlib.Path(d['path']).unlink()
                receipt.write(json.dumps({'path':d['path'],'size':d['size'],'inode':d['inode']})+'\n')
                counts[d['root']]+=1;removed+=1
                if removed%10000==0:print('deleted',removed,flush=True)
    # Remove only empty descendant directories; never follow links or remove version roots.
    pruned=0
    for root in ROOTS:
        for directory,dirs,files in os.walk(root,topdown=False,followlinks=False):
            p=pathlib.Path(directory)
            if p==root or p.is_symlink():continue
            try:p.rmdir();pruned+=1
            except OSError:pass
        (root/'ARTIFACTS_RETIRED.md').write_text('# 历史大产物已退役\n\n2026-09-22 按用户授权清理。保留的配置、指标、日志、源码及报告图不是完整备份；旧路径中的大型数组、检查点和渲染已移除，不能直接恢复旧队列。\n\n完整逐文件清单：/root/autodl-tmp/cleanup_manifests/20260922-old-runs-retire/plan.jsonl.gz\n报告：/root/autodl-tmp/motion_proj/docs/autoresearch/old_runs_retirement_20260922/README.md\n恢复需要按原配置/代码/输入在新 run 中重建；训练检查点需要重训，不能承诺逐字节复现。既有科学结论与失败记录保留。\n',encoding='utf-8')
    os.sync()
    verified=current_check();missing_keep=[];remaining_delete=[];kept=0
    with gzip.open(OUT/'plan.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line);p=pathlib.Path(d['path'])
            if d['action']=='delete':
                if p.exists():remaining_delete.append(str(p))
            else:
                kept+=1
                try:same(d)
                except (OSError,AssertionError):missing_keep.append(str(p))
    result={'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'deleted_files':removed,'by_root':dict(counts),'empty_directories_removed':pruned,'free_bytes_before':baseline,'free_bytes_after':free(),'actual_free_increase_bytes':free()-baseline,'kept_files_checked':kept,'kept_changed_or_missing':missing_keep,'remaining_delete':remaining_delete,'v75_verification':verified,'model_calls':0,'failure_ledger_delta':'none','human_verdict':None}
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert not missing_keep and not remaining_delete and not verified['changed_or_missing']

{'plan':plan,'execute':execute}[sys.argv[1]]()
