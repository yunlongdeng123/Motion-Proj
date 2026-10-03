"""只补固定窗口所需的小LiDAR文件；串行读公共tar，绝不展开整库。"""
from common import *
import tarfile, time, shutil, fcntl
from collections import defaultdict

def main():
    lock=open(O/'extract.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(O/'manifest.json')
    missing={s['filename'] for c in manifest['cases'] for s in c['scans'] if not s.get('path')}
    roots=O/'lidar';roots.mkdir(exist_ok=True)
    missing={p for p in missing if not (roots/p).is_file()}
    if missing:
        src={c['source_id']:c for c in read(T/'r8/factory/source_manifest.json')['clips']}
        cat={c['case_id']:c for c in read(T/'r8/dataset_catalog.json')['cases']}
        exp={c['clip_id']:c for c in read(A/'expected_sources.json')['clips']}
        rgb_to_cases=defaultdict(list)
        for c in manifest['cases']:
            if c['kind']=='real':rgb_to_cases[exp[c['case_id'].split('_')[0]]['frames'][c['frame_indices'][0]]['filename']].append(c['case_id'])
        real_shards={}
        for p in (A/'pub_member_lists').glob('*.txt'):
            for line in p.open():
                for cid in rgb_to_cases.get(line.strip().removeprefix('./'),[]):real_shards[cid]=p.stem
        wanted=defaultdict(set)
        absent=read(O/'archive_absent.json') if (O/'archive_absent.json').exists() else {}
        alternatives=read(O/'lidar_shards.json') if (O/'lidar_shards.json').exists() else {}
        for c in manifest['cases']:
            shards=src[cat[c['case_id']]['source_id']]['shards'] if c['kind']=='synthetic' else [real_shards[c['case_id']]]
            assert len(shards)==1,(c['case_id'],shards)
            for scan in c['scans']:
                name=scan['filename']
                if name not in missing:continue
                choices=[shards[0]]+[s for s in alternatives.get(name,[]) if s!=shards[0]]
                choices=[s for s in choices if name not in absent.get(s,[])]
                if not choices:raise RuntimeError(f'有界分片清单均无此文件: {name}')
                wanted[choices[0]].add(name)
        dump(O/'extraction_plan.json',{'by_shard':{k:sorted(v) for k,v in wanted.items() if v},'max_readers':1})
        state={'stage':'extracting','completed':[],'pid':os.getpid(),'started':time.time()}
        for shard,names in sorted(wanted.items()):
            if not names:continue
            pub=Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{shard}_blobs.tgz'
            state.update(current_shard=shard,remaining=len(names));dump(O/'extract_state.json',state)
            with tarfile.open(pub,'r|gz',bufsize=1024*1024) as tar:
                for member in tar:
                    name=member.name.removeprefix('./')
                    if name not in names:continue
                    assert member.isfile() and 0<member.size<10000000
                    out=roots/name;out.parent.mkdir(parents=True,exist_ok=True)
                    temp=out.with_suffix('.partial')
                    with tar.extractfile(member) as inp, temp.open('wb') as dst:shutil.copyfileobj(inp,dst)
                    assert temp.stat().st_size==member.size;temp.replace(out);names.remove(name)
                    state['completed'].append(name);state['remaining']=len(names);dump(O/'extract_state.json',state)
                    print('EXTRACT',shard,name,flush=True)
                    if not names:break
            if names:
                absent[shard]=sorted(set(absent.get(shard,[]))|names);dump(O/'archive_absent.json',absent)
                state.update(stage='missing_in_suggested_shard',missing_files=sorted(names));dump(O/'extract_state.json',state)
                raise RuntimeError(f'公共分片缺文件: {sorted(names)}；保留记录，下次只尝试尚未读过的候选分片')
    for c in manifest['cases']:
        for d in c['scans']:
            if d.get('path'):continue
            p=roots/d['filename'];assert p.is_file();d['path']=str(p)
    if not (O/'manifest.before_lidar.json').exists():shutil.copy2(O/'manifest.json',O/'manifest.before_lidar.json')
    dump(O/'manifest.json',manifest)
    dump(O/'extract_state.json',{'stage':'complete','missing':0,'files':len(list(roots.rglob('*.bin'))),
         'bytes':sum(p.stat().st_size for p in roots.rglob('*.bin'))})

if __name__=='__main__':main()
