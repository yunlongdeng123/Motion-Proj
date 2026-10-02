"""由真实观测过程选择新长窗；不再以旧一秒虚拟锚点作为来源门槛。"""
from pathlib import Path
import json,sys,random,copy,shutil
from collections import defaultdict,Counter
import numpy as np
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected'
sys.path[:0]=[str(S),str(S/'iteration11')]
import long_sources as long
T=long.T;O=T/'r23';ROOT=O/'factory'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def freeze():
    if (O/'source_selection.json').exists():return read(O/'source_selection.json')['sources']
    O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True)
    sources=read(T/'r8/factory/source_manifest.json')['clips']
    train_scenes=set(read(T/'r8/source_split.json')['old_r7_training_scenes']);validation_scenes={'scene-0245','scene-0246'}
    for rel in ['r8/dataset_catalog.json','r10/dataset_catalog.json']:
        for c in read(T/rel)['cases']:
            (train_scenes if c['split']=='train' else validation_scenes).add(c['receiver_scene'])
    validation_scenes|={r['scene'] for r in read(T/'r16/source_selection.json')['sources'] if r['split']=='validation'}
    assert not train_scenes&validation_scenes,'历史split冲突先排查，不能静默重分'
    unseen=sorted({c['scene'] for c in sources}-train_scenes-validation_scenes);random.Random(2301).shuffle(unseen)
    nval=max(1,round(len(unseen)*.2));validation_scenes.update(unseen[:nval]);train_scenes.update(unseen[nval:])
    by=defaultdict(list)
    for c in sources:
        anns=c['actors'][0].get('keyframe_annotations',[])
        if len(anns)!=8 or not all(anns):continue
        fs=c['frames'];boxes=np.array([f['actors'][0]['projection']['box_xyxy'] for f in fs]);cent=(boxes[:,:2]+boxes[:,2:])/2
        eg=np.array([np.asarray(f['camera_to_world'])[:3,3] for f in fs]);ego=float(np.linalg.norm(np.diff(eg,axis=0),axis=1).sum())
        relative=float(np.linalg.norm(cent[-1]-cent[0])/max(1,np.median(boxes[:,2]-boxes[:,0])))
        # 只用真实输入几何排序；优先侧向显露与多个真实保护对象。
        score=relative+min(ego,10)/10+.2*min(len(c['actors'])-1,2)
        by[c['scene']].append({'source_id':c['source_id'],'scene':c['scene'],'split':'validation' if c['scene'] in validation_scenes else 'train','score_input_only':score,'source_camera':c['camera'],'ego_one_second_path_m':ego,'relative_image_change':relative,'real_actor_count':len(c['actors'])})
    rows=[]
    for scene,pool in by.items():rows.append(max(pool,key=lambda r:(r['score_input_only'],r['source_id'])))
    rows=sorted(rows,key=lambda r:(r['split']=='train',-r['score_input_only'],r['source_id']))[:80]
    plan='''# r23：补足真正可显露的长窗来源，50条工厂pilot的来源阶段

task WS-V77-TARGET-PROTECTED-20260929/r23，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 S[既有真实source池307窗口] --> F[冻结最多80来源／每scene一条]
 F --> R[30真实曝光＋完整几何上下文]
 R --> M[真实保护实例SAM，仅质量标签]
 M --> P[后续道路可用位置与真实过程合成]
 P --> Q[空间合同＋遮后合法条件覆盖＋独立QA]
 Q --> V[约50条pilot人工逐帧页面]
```

r22发现合法车面条件精度可用但洞内覆盖有限，背景正证据更稀少。新来源不再限制必须在旧一秒窗口中已有合格虚拟锚点；按真实相机运动、目标相对图像位移、多actor数量确定性排序，每scene最多一条，最多80来源。所有原训练scene继续train，既有合成DEV验证scene继续validation；其余场景以seed2301预先固定20%验证。此处全部是nuScenes train数据池的工程train/DEV-validation，不调用最终留出集，不使用模型输出选择来源。

只准备真实曝光与输入，本阶段训练0步。曝光沿用有序唯一30帧、目标10Hz、单次窗2.9秒；不复制或插帧。全部CPU抽取按需复用缓存/公共盘，当前预计新增RGB约1GB，剩余至少30GB才继续，不删除原始数据和权重。

后续50条完整工厂pilot目标40train/10DEV-val，30保护显露、13密集交通已知背景、7普通背景；数量不足如实记录，不放宽ego/道路/深度/连续性/全覆盖合同。mask形状来源也隔离：train只用sedan，DEV-val只用suv；这仍只有两种已曝光DEV形状，存在类别差异，不能称最终泛化。原始X的车辆影响是明确轮廓，model_H与真实v2入口同合同，alpha写回单独记录。

候选生成后单独记录有界几何规划规则；不恢复r16已关闭的同来源速度搜索。基于全窗口实际显露检查，允许单帧完整遮挡，拒绝全窗没有足够证据的protected-reveal例。旧完整Y分割仅评价；构建条件需在每个遮后输入上重跑SAM。场景数不是过程充分性的替代。50条质量pilot通过后才登记A/B/C训练，surfel未启动。
'''
    result={'sources':rows,'maximum_sources':80,'max_per_scene':1,'seed':2301,'split_counts':dict(Counter(r['split'] for r in rows)),'scene_isolation':True,'train_mask_asset':'sedan','validation_mask_asset':'suv','data_domain':'nuScenes train pool; validation here is DEV split','final_used':False}
    E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r23';E.mkdir(parents=True,exist_ok=True)
    run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r23','stage':'registered_source_preparation','source_count':len(rows),'training_steps':0,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','human_verdict':None}
    for root in [O,E]:dump(root/'source_selection.json',result);dump(root/'run.json',run);(root/'plan.md').write_text(plan)
    before=O/'before_changes';before.mkdir(exist_ok=True)
    for name in ['RESEARCH_STATUS.md','EXPERIMENTS.md']:shutil.copy2(P/'docs'/name,before/name)
    p=P/'docs/EXPERIMENTS.md';lines=p.read_text().splitlines();at=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r23 | 最多80个隔离scene长窗来源；50条Temporal reveal pilot的来源准备，训练0 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r23/plan.md) |');p.write_text('\n'.join(lines)+'\n')
    print('SOURCE_REGISTERED',len(rows),result['split_counts'],flush=True);return rows

def main():
    rows=freeze();assert shutil.disk_usage(O).free>30*2**30,'保留至少30GB，先排查可回收临时物'
    long.O=O;long.ROOT=ROOT;long.freeze=lambda:rows
    if not (ROOT/'source_manifest.json').exists():
        try:long.resolve()
        except KeyError as error:
            # 旧池混有重命名source ID。下方按八个真实sample token恢复上下文。
            if not (ROOT/'source_manifest.json').exists():raise
            dump(O/'context_alias_error.json',{'error':repr(error),'action':'resolve_by_actual_keyframe_tokens_not_ID'})
    p=ROOT/'source_manifest.json';d=read(p);d['run_id']='r23';dump(p,d)
    bases=[T/'r8/factory',T/'r16/factory',T/'r8/short_sources/factory',T/'r4/factory',T/'r3/factory',T/'r1/expanded_factory',T/'r1/source_extension',T/'r1']
    contexts={}
    for base in bases:
        if not (base/'source_context.json').exists():continue
        for c in read(base/'source_context.json')['clips']:contexts[tuple(f['sample_token'] for f in c['frames'])]=c
    absent=[c for c in d['clips'] if tuple(c['keyframe_tokens']) not in contexts]
    if absent:
        from prepare_context_extract import context
        missing_root=O/'context_recovery';missing_root.mkdir(exist_ok=True)
        dump(missing_root/'source_manifest.json',{'clips':absent})
        context(missing_root,long.META)
        for c in read(missing_root/'source_context.json')['clips']:contexts[tuple(f['sample_token'] for f in c['frames'])]=c
    selected_context=[]
    for c in d['clips']:
        ctx=copy.deepcopy(contexts[tuple(c['keyframe_tokens'])]);ctx['source_id']=c['source_id'];selected_context.append(ctx)
    dump(ROOT/'source_context.json',{'clips':selected_context,'source_identity':'exact_all_8_keyframe_sample_tokens','purpose':'quality geometry and declared fixed GT pose auxiliary; no hidden RGB'})
    byid={c['source_id']:c for c in selected_context};names={f['filename'] for c in d['clips'] for f in c['frames']}
    for c in d['clips']:
        ta,tb=c['frames'][0]['timestamp'],c['frames'][-1]['timestamp']
        for i,f in enumerate(byid[c['source_id']]['frames']):
            lidar=f['sensors']['LIDAR_TOP']
            if i in [0,3,6] or ta<=lidar['timestamp']<=tb:names.add(lidar['filename'])
    (ROOT/'rgb').mkdir(exist_ok=True);missing=[]
    for name in sorted(names):
        dest=ROOT/'rgb'/name
        if dest.is_file():continue
        old=next((base/'rgb'/name for base in bases if (base/'rgb'/name).is_file()),None)
        if old is None:missing.append(name);continue
        dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(old.resolve())
    if not (ROOT/'maps').exists():(ROOT/'maps').symlink_to((T/'r8/factory/maps').resolve(),target_is_directory=True)
    dump(ROOT/'required_files.json',missing);dump(O/'RGB_request.json',{'total_files':len(names),'cache_reused':len(names)-len(missing),'missing':missing,'bytes_estimate':len(missing)*350000,'space_free_bytes':shutil.disk_usage(ROOT).free})
    print('R23_MATERIALIZE',len(d['clips']),'missing',len(missing),flush=True)
    if missing:
        from prepare_context_extract import extract
        extract(ROOT,Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval'))
    from PIL import Image
    for c in d['clips']:
        for f in c['frames']:
            with Image.open(ROOT/'rgb'/f['filename']) as im:im.load();assert im.size==(1600,900)
    dump(O/'source_ready.json',{'stage':'ready_for_quarantined_SAM2','clips':len(d['clips']),'frames':len(d['clips'])*30,'actors':sum(len(c['actors']) for c in d['clips']),'source_quality_inherited':False,'synthetic_admission':0,'training_steps':0,'human_verdict':None})
    print('R23_SOURCE_READY',len(d['clips']),flush=True)
if __name__=='__main__':main()
