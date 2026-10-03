"""有界串行提取额外输入；只写新的r47缓存，不清理历史资产。"""
from common import *
import tarfile, shutil, time, fcntl
from collections import defaultdict
from PIL import Image
import numpy as np


def main():
    lock = open(O/'extract.lock','a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    files = read(O/'extra_input_plan.json')['files']; root = O/'extra_inputs'
    groups = defaultdict(set)
    for f in files:
        if not (root/f['filename']).is_file():
            for shard in f['shards']: groups[shard].add(f['filename'])
    state = {'stage':'extracting','completed':0,'total':len(files),'pid':os.getpid()}
    for shard, names in sorted(groups.items()):
        names = {n for n in names if not (root/n).is_file()}
        if not names: continue
        state.update(shard=shard, remaining_in_shard=len(names)); dump(O/'extract_state.json',state)
        start = time.monotonic()
        archive = Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{shard}_blobs.tgz'
        with tarfile.open(archive,'r|gz',bufsize=1024*1024) as tar:
            for member in tar:
                name = member.name.removeprefix('./')
                if name not in names: continue
                assert member.isfile() and 0<member.size<10000000
                out = root/name; out.parent.mkdir(parents=True,exist_ok=True)
                part = out.with_suffix('.partial')
                with tar.extractfile(member) as inp, part.open('wb') as dst: shutil.copyfileobj(inp,dst)
                assert part.stat().st_size==member.size
                part.replace(out); names.remove(name); state['completed']+=1
                state['remaining_in_shard']=len(names); dump(O/'extract_state.json',state)
                if not names: break
        print('SHARD',shard,'missing',len(names),'seconds',time.monotonic()-start,flush=True)
    missing = []
    for f in files:
        path = root/f['filename']
        if not path.is_file(): missing.append(f['filename']); continue
        if f['kind']=='RGB':
            with Image.open(path) as image: image.verify()
        else: assert np.fromfile(path,np.float32).size%5==0
    dump(O/'missing_extra_inputs.json', {'files':missing})
    if missing: raise RuntimeError(f'公共数据缺{len(missing)}文件，保留清单')
    plan = read(O/'manifest.json')
    for c in plan['cases']:
        for r in c['references']:
            if r.get('path') is None and r.get('filename'):
                r['path']=str(root/r['filename']); r['available']=True
        for scan in c.get('bev_scans',[]):
            if not scan.get('path'): scan['path']=str(root/scan['filename'])
    dump(O/'manifest.json',plan)
    state.update(stage='complete', decoded_files=len(files), bytes=sum((root/f['filename']).stat().st_size for f in files))
    dump(O/'extract_state.json',state); print('EXTRA_INPUTS_COMPLETE',len(files),flush=True)


if __name__=='__main__': main()
