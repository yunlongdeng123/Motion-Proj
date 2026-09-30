"""CPU r4：新receiver窗口先选配对素材，真实GT与旧数据不改动。"""
import argparse, copy, os, sys, time
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_source_pool import read,dump,resolve
from prepare_context_extract import context,extract

def main(root,parent,meta,pub,native_ten=False):
    root.mkdir(exist_ok=True,parents=True)
    factory=root/('native10_factory' if native_ten else 'factory');factory.mkdir(exist_ok=True)
    old=[]
    for rel in ['source_manifest.json','source_extension/source_manifest.json']:
        old+=read(parent/rel)['clips']
    identity=lambda c:(c['scene'],c['camera'],c['start_keyframe'])
    used={identity(c) for c in old}
    pool=read(parent/'candidate_pool.json')['candidates']
    candidates=[c for c in pool if c['location']=='boston-seaport' and set(c['shards']).issubset({'03','07'}) and identity(c) not in used]
    # 仅变更receiver采样；不重跑已穷举的旧窗口，也不调整合成物理门槛。
    dense=sorted([c for c in candidates if len(c['actors'])>=2],key=lambda c:(c['scene'],c['camera'],c['start_keyframe']))
    chosen=copy.deepcopy(dense);scenes={c['scene'] for c in chosen}
    prior_scenes={c['scene'] for c in old}
    singles=sorted([c for c in candidates if len(c['actors'])==1],key=lambda c:(c['scene'] in prior_scenes,abs(c['actors'][0]['anchor_projection']['depth']-20),c['scene'],c['start_keyframe'],c['camera']))
    for c in singles:
        if c['scene'] in scenes:continue
        chosen.append(copy.deepcopy(c));scenes.add(c['scene'])
        if len(chosen)>=32:break
    for i,c in enumerate(chosen):c['source_id']=f'C{i+1:03}'
    cfg={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r4','cpu_only':True,'cpu_threads':1,'seed':42,
         'max_new_receiver_windows':32,'archive_scope':['03','07'],'window':'fixed frames 10:20 of resolved 30-exposure sequence; selected before masks or synthesis',
         'source_role':'train development data factory; not validation/test','planning_only_until_actual_masks_and_independent_QA':True,
         'stop_rule':'one new receiver pool, original placement operator and geometry gates; no per-case threshold changes; stop before new SAM2 inference',
         'no_repeat_old_receiver_windows':True,'remaining_old_pool_total':len(candidates),'new_dense_metadata_windows':len(dense),
         'selection_reason':'new unprocessed receiver windows, prioritize multiple well-observed protected actors and new scenes; 2 shards bound CPU extraction cost'}
    if (root/'run_config.json').exists():assert read(root/'run_config.json')==cfg
    dump(root/'run_config.json',cfg);dump(factory/'source_selection.json',cfg|{'clips':chosen,'human_verdict':None})
    print('SELECTED',len(chosen),'SCENES',len(scenes),'DENSE',len(dense),'NEW_SCENES',len(scenes-prior_scenes),flush=True)
    if native_ten:
        from sample_windows import resolve_ten
        if not (factory/'source_manifest.json').exists():resolve_ten(meta,factory,{'clips':chosen})
        clips=read(factory/'source_manifest.json')['clips']
    else:
        if not (factory/'source_manifest_30.json').exists():
            resolve(meta,factory,{'clips':chosen})
            (factory/'source_manifest.json').rename(factory/'source_manifest_30.json')
        m=read(factory/'source_manifest_30.json');clips=[]
        for c in m['clips']:
            c=copy.deepcopy(c);c['frames']=c['frames'][10:20]
            for i,f in enumerate(c['frames']):f['frame']=i
            c['source_window']={'parent_frame_start':10,'parent_frame_stop_exclusive':20,'selection':'fixed before pixel/model review'}
            c['cpu_metadata_only']=True;clips.append(c)
        dump(factory/'source_manifest.json',m|{'run_id':'r4','clips':clips})
    if not (factory/'source_context.json').exists():context(factory,meta)
    required=sorted({f['filename'] for c in clips for f in c['frames']}|{c['frames'][i]['sensors']['LIDAR_TOP']['filename'] for c in read(factory/'source_context.json')['clips'] for i in [0,3,6]})
    # 不解包未参与本轮的六相机RGB与雷达；未来确需多视角再按清单提取。
    dump(factory/'required_files.json',required)
    reuse=([root/'factory/rgb',root/'factory/by_shard/03',root/'factory/by_shard/07'] if native_ten else [])+[parent/'rgb',parent/'source_extension/rgb',root.parent/'r3/factory/rgb']
    need=[];reused=0
    for name in required:
        dst=factory/'rgb'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        src=None
        for base in reuse:
            candidate=base/name
            if not candidate.is_file():continue
            try:
                if name.endswith('.jpg'):
                    from PIL import Image
                    with Image.open(candidate) as image:image.load();assert image.size==(1600,900)
                else:assert candidate.stat().st_size>0 and candidate.stat().st_size%20==0
                src=candidate;break
            except (OSError,AssertionError):continue
        if src:
            if not dst.exists():dst.symlink_to(src.resolve())
            reused+=1
        else:need.append(name)
    dump(factory/'all_required_files.json',required)
    dump(factory/'required_files.json',need)
    if need:extract(factory,pub)
    dump(factory/'required_files.json',required)
    if not (factory/'maps').exists():(factory/'maps').symlink_to(parent/'maps',target_is_directory=True)
    missing=[n for n in required if not (factory/'rgb'/n).is_file()]
    summary={'case_count':len(clips),'scene_count':len({c['scene'] for c in clips}),'dense_metadata_candidates':sum(len(c['actors'])>=2 for c in clips),
             'required_files':len(required),'reused_files':reused,'new_extraction_requests':len(need),'missing_files':missing,'actual_synthetic_cases':0,'training_ready':0}
    dump(root/('preparation_native10.json' if native_ten else 'preparation.json'),summary);print('PREPARED',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','parent','meta','pub']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--native-ten',action='store_true')
    a=p.parse_args();main(a.root,a.parent,a.meta,a.pub,a.native_ten)
