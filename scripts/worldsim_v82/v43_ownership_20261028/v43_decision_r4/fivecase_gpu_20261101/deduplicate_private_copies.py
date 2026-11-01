"""任务内重复只读副本逐字节核对后共享inode；不碰原数据、模型、日志、gauge。"""
from pathlib import Path
import filecmp,os,json,time
R=Path(__file__).resolve().parent
C={'008':R/'gpu/waymo-train-008_a37/audit49/restore/scene_loader/query_scene',
   '212':R/'gpu_r2/waymo-train-212_a0/audit49/restore/scene_loader/query_scene',
   '564':R/'gpu_564_replay/waymo-train-564_a0/audit49/restore/scene_loader/query_scene'}
records=[];start=time.monotonic()
for directory in R.glob('gpu*/**/query_scene'):
    relative=directory.resolve().relative_to(R.resolve())
    if directory.name!='query_scene' or directory.parent.name!='scene_loader':raise ValueError('目录边界')
    # 目录链上有场景短号，缓存只指向本任务已存在的另一份独立副本。
    number=next((n for n in C if any(('waymo-train-'+n+'_') in p for p in relative.parts)),None)
    if number is None:continue
    cache=C[number]
    if directory.resolve()==cache.resolve():continue
    count=freed=0
    for f in directory.rglob('*'):
        if not f.is_file() or f.is_symlink() or f.name=='transform_matrix.txt':continue
        prior=cache/f.relative_to(directory)
        if not prior.is_file() or prior.is_symlink() or os.path.samefile(f,prior):continue
        if f.stat().st_size==prior.stat().st_size and filecmp.cmp(f,prior,shallow=False):
            temp=f.with_name(f.name+'.task_private_same_bytes_link')
            if temp.exists():raise FileExistsError(temp)
            size=f.stat().st_size;os.link(prior,temp);os.replace(temp,f);count+=1;freed+=size
    records.append(dict(directory=str(directory),same_byte_files=count,logical_duplicate_bytes_replaced=freed))
    print(directory,count,freed,flush=True)
(R/'private_copy_deduplication.json').write_text(json.dumps(dict(records=records,wall_seconds=time.monotonic()-start,
    original_sources_or_models_touched=False,all_private_file_bytes_preserved=True,mutable_gauges_excluded=True,
    directories_deleted=False),indent=2))
