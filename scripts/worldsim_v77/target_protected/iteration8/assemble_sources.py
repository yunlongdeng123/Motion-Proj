"""复用真实已提取短窗与上下文；不重复曝光、不隐式修改历史来源。"""
from pathlib import Path
import sys,copy,json
from collections import Counter,defaultdict
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
from geometry_factory import read,dump
O=T/'r8';F=O/'factory'
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
def main():
    F.mkdir(exist_ok=True)
    if (F/'source_manifest.json').exists():print('source manifest frozen');return
    oldcatalog=read(T/'r6/dataset_catalog.json')['cases']
    oldtrain={s for c in oldcatalog if c['split']=='train' for s in [c['receiver_scene'],c['donor_scene']]}
    oldval={s for c in oldcatalog if c['split']=='validation' for s in [c['receiver_scene'],c['donor_scene']]}
    roots=[T/'r4/native10_factory',T/'r1/native_window_factory',T/'r1/expanded_factory',T/'r1']
    rows=[];seen=set();ctxs=[];counts=Counter();bad=[]
    for base in roots:
        clips=read(base/'source_manifest.json')['clips']
        context=read(base/'source_context.json')['clips']
        ctxby={r['source_id']:r for r in context}
        # native窗口上下文已改ID；若缺失，用父实际来源上下文，不改keyframe时间。
        for c in clips:
            if c['scene'] in oldval:continue
            ctx=ctxby.get(c['source_id'])
            if ctx is None:bad.append({'source_id':c['source_id'],'reason':'context_missing'});continue
            n=len(c['frames']);start=0 if n==10 else 10
            if n not in [10,30]:continue
            cc=copy.deepcopy(c);fs=cc['frames'][start:start+10]
            identity=tuple(f['filename'] for f in fs)
            if identity in seen:continue
            if counts[c['scene']]>=3:continue
            required=[f['filename'] for f in fs]+[ctx['frames'][i]['sensors']['LIDAR_TOP']['filename'] for i in [0,3,6]]
            if any(not (base/'rgb'/name).is_file() for name in required):bad.append({'source_id':c['source_id'],'reason':'RGB_or_LiDAR_missing'});continue
            if len({f['timestamp'] for f in fs})!=10 or any(not 0<b['timestamp']-a['timestamp']<=180000 for a,b in zip(fs,fs[1:])):continue
            if any(len(f['actors'])!=len(c['actors']) or any(a['projection'] is None for a in f['actors']) for f in fs):continue
            seen.add(identity);counts[c['scene']]+=1
            sid=f'N{len(rows)+1:03}';cc['source_id']=sid;cc['frames']=fs
            for i,f in enumerate(fs):f['frame']=i
            cc['reuse_provenance']={'factory':str(base),'source_id':c['source_id'],'slice':[start,start+10],'old_train_scene':c['scene'] in oldtrain}
            ctx=copy.deepcopy(ctx);ctx['source_id']=sid
            rows.append(cc);ctxs.append(ctx)
            for name in required:link(base/'rgb'/name,F/'rgb'/name)
            # 只复用实际同窗口10帧SAM结果；30帧mask也切成固定10帧。
            src=base/'segmented'/c['source_id']/'sam2_raw'
            masks=sorted(src.glob('*.png'))
            if len(masks)==n:
                for i,p in enumerate(masks[start:start+10]):link(p,F/'segmented'/sid/'sam2_raw'/f'{i:05}.png')
    dump(F/'source_manifest.json',{'clips':rows,'stage':'source_numeric_candidates_not_visual_admitted'})
    dump(F/'source_context.json',{'clips':ctxs})
    link(T/'r1/maps',F/'maps')
    dump(O/'source_pool_summary.json',{'source_windows':len(rows),'scenes':len(counts),'max_windows_per_scene':max(counts.values()),'multi_actor_windows':sum(len(c['actors'])>=2 for c in rows),'locations':dict(Counter(c['location'] for c in rows)),'old_train_scenes':sorted(oldtrain),'old_validation_scenes_excluded':sorted(oldval),'missing_or_bad':bad,'receiver_scene_counts':dict(counts)})
    print('ASSEMBLED',len(rows),'SCENES',len(counts),'DENSE',sum(len(c['actors'])>=2 for c in rows))
if __name__=='__main__':main()
