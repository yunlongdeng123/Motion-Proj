"""r28可复现恢复：原30来源/车道/速度，按所有帧的实例疑点冻结轨迹清单。

旧首个拒绝原因不能重建候选清单。保留已有72例和原登记，补齐语义选择器
发现的原轨迹；完整Y只送隔离质检，原r21 model H不裁减、不放宽准入。
"""
from pathlib import Path
import sys,time,os
sys.path.insert(0,str(Path(__file__).parent))
import prepare_instance_audit as p
f=p.f;np=p.np;Image=p.Image;O=p.O
STATIC={'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}


def candidate_key(sid,tr):
    return (sid,tr['lane_token'],round(tr['lane_midpoint_distance_m'],9),float(tr['speed_mps']))


def pixels(box):
    xy0=np.maximum(np.floor(box[:2]).astype(int),0)
    xy1=np.minimum(np.ceil(box[2:]).astype(int),[1024,576])
    return int(xy0[0]),int(xy0[1]),int(xy1[0]),int(xy1[1])


def overlapping(holes,obstacles):
    hits={}
    for i,(hole,objects) in enumerate(zip(holes,obstacles)):
        threshold=max(12,.01*hole.sum())
        for ob in objects:
            pr=ob['_projection']
            if pr is None:continue
            x0,y0,x1,y1=pixels(pr['box'])
            if x1>x0 and y1>y0 and hole[y0:y1,x0:x1].sum()>threshold:
                tok=ob['instance_token']
                hits.setdefault(tok,{'category':ob['category'],'frames':[]})['frames'].append(i)
    return {k:hits[k] for k in sorted(hits)}


def unknown_hits(holes,obstacles,protected):
    return {k:v for k,v in overlapping(holes,obstacles).items() if k not in protected and v['category'] not in STATIC}


def load_masks(folder,role):
    return [np.asarray(Image.open(folder/role/f'{i:05}.png'))>0 for i in range(30)]


def select(g):
    path=O/'candidate_roster.json'
    if path.exists():return f.read(path)
    old={candidate_key(x['source_id'],x['trajectory']):x for x in
         (f.read(z) for z in sorted((O/'observed').glob('Q*/case.json')))}
    assert len(old)==72,'恢复起点必须是已保存的72例；如有新清单必须读取它'
    plan=f.read(O/'run.json');api=f.NuScenesMap(dataroot=str(f.ROOT),map_name='boston-seaport')
    assets={a:dict(np.load(f.T/'r8/assets'/f'{a}.npz')) for a in ['sedan','suv']}
    rows=[];seen=set();next_id=73;started=time.time()
    for sid in plan['sources']:
        c=g.sources[sid];g.prepare(sid);f.legacy.support(g,sid);pm=f.legacy.protections(g,sid)
        asset=f.POLICY['split_shape'][c['source_split']];mesh=assets[asset]
        size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        for q in f.centers(g,api,c,size):
            for speed in f.POLICY['speeds_mps']:
                tr,why=f.trajectory(g,c,q,speed,asset)
                if tr is None:continue
                key=candidate_key(sid,tr);previous=old.get(key)
                if previous:
                    # 只复用完全相同几何与相机的像素；不按相近位置偷换样本。
                    assert p.clean(tr)==previous['trajectory'],key
                    assert p.clean(c['frames'])==previous['frames'],('camera/source changed',key)
                    masks=load_masks(Path(previous['observed_folder']),'influence')
                else:
                    masks=[f.legacy.old.silhouette(mesh['vertices'],mesh['faces'],a['actor'],fr) for a,fr in zip(tr['frames'],c['frames'])]
                holes=[f.prepare_masks(a)[0]['model_mask'] for a in masks]
                hits=unknown_hits(holes,g.obstacles[sid],pm)
                if not hits:
                    assert previous is None,('旧样本不符合语义选择器，需人工排查',key)
                    continue
                if previous:cid=previous['case_id'];seen.add(key)
                else:cid=f'Q{next_id:03}';next_id+=1
                _,first=f.exact(g,c,tr,masks,pm)
                rows.append({'case_id':cid,'source_id':sid,'trajectory':p.clean(tr),'unknown_envelope_hits':hits,
                             'sorted_first_rejection':first,'reuse_saved_pixels':previous is not None})
        print('ROSTER',sid,len(rows),flush=True)
    assert seen==set(old),('旧轨迹未全部复现',set(old)-seen)
    rows.sort(key=lambda x:x['case_id'])
    result={'selection_id':'all_frames_unknown_envelope_v1','selection_frozen_before_Y_SAM':True,
            'original_first_rejection_count':plan['max_existing_trajectories'],'saved_pixel_reuse':len(seen),
            'cases':rows,'count':len(rows),'sources':plan['sources'],
            'no_new_positions_speeds_thresholds':True,'new_independent_trajectories_not_first_reason_count':True,
            'training_admission':0,'training_steps':0,'seconds':time.time()-started,
            'reason':'first-failure counts vary with set iteration; freeze semantic keys and all-frame overlapping unknown instances'}
    f.dump(path,result);return result


def prepare(g,roster,recovery=True):
    assert not (O/'prepared.json').exists(),'已有队列不能覆盖'
    quality_jobs={};method_jobs=[];quality_completed=[];cases=[];started=time.time()
    assets={a:dict(np.load(f.T/'r8/assets'/f'{a}.npz')) for a in ['sedan','suv']}
    for item in roster['cases']:
        cid=item['case_id'];sid=item['source_id'];tr=item['trajectory'];c=g.sources[sid]
        if sid not in g.obstacles:g.prepare(sid)
        pm=f.legacy.protections(g,sid);source=O/'quality_labels/sources'/sid;dest=O/'observed'/cid
        for role in ['Y','sam_rgb','H']:(source/role).mkdir(parents=True,exist_ok=True)
        for i,fr in enumerate(c['frames']):
            if not (source/'Y'/f'{i:05}.png').exists():
                y=Image.open(f.ROOT/'rgb'/fr['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
                y.save(source/'Y'/f'{i:05}.png');y.save(source/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
                Image.fromarray(np.zeros((576,1024),'uint8')).save(source/'H'/f'{i:05}.png')
        reused=item['reuse_saved_pixels']
        if reused:masks=load_masks(dest,'influence');guards=load_masks(dest,'H');holes=load_masks(dest,'proposal_H')
        else:
            for role in ['rgb','sam_rgb','H','proposal_H','influence']:(dest/role).mkdir(parents=True,exist_ok=True)
            mesh=assets[tr['asset']]
            masks=[f.legacy.old.silhouette(mesh['vertices'],mesh['faces'],a['actor'],fr) for a,fr in zip(tr['frames'],c['frames'])]
            guards=[];holes=[]
            for i,m in enumerate(masks):
                contract,_=f.prepare_masks(m);guard=contract['write_mask'];hole=contract['model_mask']
                y=np.asarray(Image.open(source/'Y'/f'{i:05}.png'));x=y.copy();x[guard]=127
                Image.fromarray(x).save(dest/'rgb'/f'{i:05}.png');Image.fromarray(x).save(dest/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
                for role,a in [('H',guard),('proposal_H',hole),('influence',m)]:Image.fromarray(a.astype('uint8')*255).save(dest/role/f'{i:05}.png')
                guards.append(guard);holes.append(hole)
        neighbors=overlapping(holes,g.obstacles[sid]);identities=[]
        for tok,hit in neighbors.items():
            annotation=[];boxes=[];quality_areas=[];visible_areas=[]
            for i,objects in enumerate(g.obstacles[sid]):
                ob=next((v for v in objects if v['instance_token']==tok),None)
                if ob is None or ob['_projection'] is None:
                    annotation.append(p.clean(ob) if ob is not None else None);boxes.append(None);quality_areas.append(0);visible_areas.append(0);continue
                bb=ob['_projection']['box'];boxes.append(bb.tolist());annotation.append(p.clean(ob))
                x0,y0,x1,y1=pixels(bb);quality_areas.append(max(0,x1-x0)*max(0,y1-y0))
                visible_areas.append(int((~guards[i][y0:y1,x0:x1]).sum()) if x1>x0 and y1>y0 else 0)
            qid=sid+'_'+tok[:8];prompt=int(np.argmax(quality_areas));mprompt=int(np.argmax(visible_areas))
            identities.append({'instance_token':tok,'category':hit['category'],'annotations':annotation,'boxes':boxes,
                'Y_quality_job':qid,'visible_X_job':cid+'_'+tok[:8] if visible_areas[mprompt]>=100 else None,
                'quality_input_available':max(quality_areas)>=100,'curated_source_actor':tok in pm})
            if max(quality_areas)>=100 and qid not in quality_jobs:
                job={'case_id':sid,'job_id':qid,'instance_token':tok,'category':hit['category'],'prompt_frame':prompt,
                    'prompt_box':boxes[prompt],'frames':30,'input_folder':str(source),'role':'evaluation_only_full_Y',
                    'forbidden_for_condition_building':True};quality_jobs[qid]=job
                out=O/'quality_labels/observed_masks'/qid
                if (out/'result.json').exists():
                    assert all((out/f'{i:05}.png').exists() for i in range(30)),qid
                    quality_completed.append(f.read(out/'result.json'))
                elif tok in pm:
                    out.mkdir(parents=True,exist_ok=True)
                    for i,m in enumerate(pm[tok]):Image.fromarray(m.astype('uint8')*255).save(out/f'{i:05}.png')
                    actor_index=next(j for j,a in enumerate(c['actors']) if a['instance_token']==tok)
                    original_dir=f.ROOT/('segmented' if actor_index==0 else 'segmented_secondary')/(sid if actor_index==0 else sid+'_'+tok[:8])
                    actual=f.read(original_dir/'mask_manifest.json');pf=actual['prompt_frame']
                    actual_box=next(a for a in c['frames'][pf]['actors'] if a['instance_token']==tok)['projection']['box_xyxy']
                    rec=job|{'reused_from':f'{f.O.name} technically passed Y-SAM','quality':'pending_case_identity_review','human_verdict':None,
                        'requested_prompt_frame':job['prompt_frame'],'prompt_frame':pf,'prompt_box':actual_box,
                        'actual_SAM_rgb_folder':str(f.ROOT/'SAM2_rgb'/sid),'actual_mask_manifest':str(original_dir/'mask_manifest.json'),
                        'checkpoint':actual['checkpoint'],'config':actual['config']}
                    f.dump(out/'result.json',rec);quality_completed.append(rec)
            if visible_areas[mprompt]>=100:
                method_jobs.append({'case_id':cid,'job_id':cid+'_'+tok[:8],'instance_token':tok,'category':hit['category'],
                    'prompt_frame':mprompt,'prompt_box':boxes[mprompt],'frames':30,'input_folder':str(dest),
                    'role':'mask_planning_from_guard_erased_X','full_Y_access':False,'target_RGB_access':False,
                    'not_final_actor_state_features':True})
        row={'case_id':cid,'source_id':sid,'scene':c['scene'],'split':c['source_split'],'asset':tr['asset'],
            'trajectory':tr,'camera':c['camera'],'frames':p.clean(c['frames']),'source_Y_quality_only':str(source/'Y'),
            'observed_folder':str(dest),'retained_instances':identities,'original_rejection':item['sorted_first_rejection'],
            'candidate_H':'unchanged r21 rectangle','alternative_visible_H':'diagnostic only after r26 regression',
            'selector':roster['selection_id'],'reuse_saved_pixels':reused,'training_admission':False,'human_verdict':None}
        # 已有case描述另留备份；图像不覆盖。
        old=dest/'case.json';backup=dest/'case_before_roster_recovery.json'
        if old.exists() and not backup.exists():backup.write_bytes(old.read_bytes())
        row=p.clean(row);f.dump(old,row);cases.append(row)
        f.dump(O/'preparation_state.json',{'stage':'recovering_queues','pid':os.getpid(),'prepared_cases':len(cases),
            'frozen_cases':roster['count'],'reused_pixels':sum(x['reuse_saved_pixels'] for x in cases),'seconds':time.time()-started})
        print('RECOVER',cid,sid,'reused' if reused else 'new_existing_trajectory',len(identities),flush=True)
    f.dump(O/'quality_labels/observed_queue.json',{'jobs':list(quality_jobs.values()),'role':'Y_evaluation_only'})
    f.dump(O/'quality_labels/segmentation_state.json',{'completed':quality_completed,'stage':'pending_remaining_Y_labels'})
    f.dump(O/'method_masks/observed_queue.json',{'jobs':method_jobs,'role':'target_guard_erased_X_only'})
    f.dump(O/'prepared.json',{'cases':cases,'method_jobs':len(method_jobs),'quality_jobs':len(quality_jobs),
        'reused_quality_jobs':len(quality_completed),'selector':roster['selection_id'],'training_admission':0,'training_steps':0})
    if recovery:f.dump(O/'selection_amendment.json',{'original_registration_preserved':'run.json','selector':roster['selection_id'],
        'original_count_was_first_failure_counter':73,'saved_pixel_reuse':roster['saved_pixel_reuse'],'actual_frozen_count':roster['count'],
        'sources_unchanged':True,'geometry_and_thresholds_unchanged':True,'Y_SAM_not_started_before_freeze':True,
        'training_steps':0,'failure_ledger_refs':['V77-F02']})
    f.dump(O/'preparation_state.json',{'stage':'complete_pending_SAM','pid':os.getpid(),'prepared_cases':len(cases),
        'method_SAM_jobs':len(method_jobs),'quality_SAM_jobs':len(quality_jobs),'quality_reused':len(quality_completed),'seconds':time.time()-started})


if __name__=='__main__':
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT
    geometry=f.legacy.geometry();roster=select(geometry);prepare(geometry,roster)
