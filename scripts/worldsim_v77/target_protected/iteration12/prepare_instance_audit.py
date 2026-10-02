"""r28：只补r23未核验实例，不增加位置/速度搜索，不自动形成训练集。

两套目录分离：quality_labels完整Y仅供评价；observed只含擦除target guard后的X。
这轮只造可复核候选和SAM队列，r26失败的洞裁减不升级为默认推理规则。
"""
from pathlib import Path
import sys,os,time
sys.path.insert(0,str(Path(__file__).parent))
import reveal_factory as f
import numpy as np
from PIL import Image

O=f.T/'r28'


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def roi(box):
    x0,y0,x1,y1=np.rint(box).astype(int);x0,x1=np.clip([x0,x1],0,1024);y0,y1=np.clip([y0,y1],0,576)
    return x0,y0,x1,y1


def prepare():
    O.mkdir(exist_ok=True);assert not (O/'prepared.json').exists(),'已准备，不重复'
    records=[f.read(p) for p in sorted((f.O/'lane_candidates').glob('*.json'))]
    sources=[r['source_id'] for r in records if r['rejects'].get('unreviewed_dynamic_envelope')]
    budget=sum(r['rejects'].get('unreviewed_dynamic_envelope',0) for r in records)
    f.dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r28',
        'phase':'complete_instance_labels_for_existing_rejections','source_scenes':len(sources),'sources':sources,
        'max_existing_trajectories':budget,'no_new_positions_speeds_or_thresholds':True,
        'main_inference_default':'r21 sam_full_v2; r26 not promoted',
        'full_Y_role':'separate quality labels only','method_RGB':'X with complete target guard erased before JPEG',
        'mask_planning_not_actor_state_features':True,'frames':30,'model_seed':42,
        'training_steps':0,'training_admission':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None})
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry()
    api=f.NuScenesMap(dataroot=str(f.ROOT),map_name='boston-seaport')
    quality_jobs={};quality_completed=[];method_jobs=[];cases=[];started=time.time()
    assets={a:dict(np.load(f.T/'r8/assets'/f'{a}.npz')) for a in ['sedan','suv']}
    (O/'quality_labels').mkdir(exist_ok=True);(O/'method_masks').mkdir(exist_ok=True)
    for sid in sources:
        c=g.sources[sid];g.prepare(sid);f.legacy.support(g,sid);pm=f.legacy.protections(g,sid)
        asset=f.POLICY['split_shape'][c['source_split']];size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        mesh=assets[asset];source=O/'quality_labels/sources'/sid
        for role in ['Y','sam_rgb','H']:(source/role).mkdir(parents=True,exist_ok=True)
        for i,fr in enumerate(c['frames']):
            if (source/'Y'/f'{i:05}.png').exists():continue
            y=Image.open(f.ROOT/'rgb'/fr['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
            y.save(source/'Y'/f'{i:05}.png');y.save(source/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
            Image.fromarray(np.zeros((576,1024),'uint8')).save(source/'H'/f'{i:05}.png')
        for q in f.centers(g,api,c,size):
            for speed in f.POLICY['speeds_mps']:
                tr,why=f.trajectory(g,c,q,speed,asset)
                if tr is None:continue
                masks=[f.legacy.old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],fr) for p,fr in zip(tr['frames'],c['frames'])]
                _,why=f.exact(g,c,tr,masks,pm)
                if why!='unreviewed_dynamic_envelope':continue
                cid=f'Q{len(cases)+1:03}';dest=O/'observed'/cid
                for role in ['rgb','sam_rgb','H','proposal_H','influence']:(dest/role).mkdir(parents=True,exist_ok=True)
                guards=[];holes=[];neighbors={}
                for i,m in enumerate(masks):
                    contract,_=f.prepare_masks(m);guard=contract['write_mask'];hole=contract['model_mask']
                    y=np.asarray(Image.open(source/'Y'/f'{i:05}.png')).copy();x=y.copy();x[guard]=127
                    Image.fromarray(x).save(dest/'rgb'/f'{i:05}.png');Image.fromarray(x).save(dest/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
                    for role,a in [('H',guard),('proposal_H',hole),('influence',m)]:Image.fromarray(a.astype('uint8')*255).save(dest/role/f'{i:05}.png')
                    assert np.array_equal(x[~guard],y[~guard]) and (x[guard]==127).all()
                    guards.append(guard);holes.append(hole)
                    for ob in g.obstacles[sid][i]:
                        pr=ob['_projection']
                        if pr is None:continue
                        x0,y0,x1,y1=roi(pr['box'])
                        if x1<=x0 or y1<=y0 or hole[y0:y1,x0:x1].sum()<=max(12,.01*hole.sum()):continue
                        tok=ob['instance_token'];neighbors[tok]=ob['category']
                identities=[]
                for tok,category in sorted(neighbors.items()):
                    annotation=[];boxes=[];quality_areas=[];visible_areas=[]
                    for i,objects in enumerate(g.obstacles[sid]):
                        ob=next((v for v in objects if v['instance_token']==tok),None)
                        if ob is None or ob['_projection'] is None:
                            annotation.append(None);boxes.append(None);quality_areas.append(0);visible_areas.append(0);continue
                        pr=ob['_projection'];bb=pr['box'];boxes.append(bb.tolist());annotation.append(clean(ob))
                        x0,y0,x1,y1=roi(bb)
                        quality_areas.append(max(0,x1-x0)*max(0,y1-y0))
                        visible_areas.append(int((~guards[i][y0:y1,x0:x1]).sum()) if x1>x0 and y1>y0 else 0)
                    qid=sid+'_'+tok[:8];prompt=int(np.argmax(quality_areas));mprompt=int(np.argmax(visible_areas))
                    identities.append({'instance_token':tok,'category':category,'annotations':annotation,'boxes':boxes,
                        'Y_quality_job':qid,'visible_X_job':cid+'_'+tok[:8] if visible_areas[mprompt]>=100 else None,
                        'quality_input_available':max(quality_areas)>=100,'curated_source_actor':tok in pm})
                    if max(quality_areas)>=100 and qid not in quality_jobs:
                        job={'case_id':sid,'job_id':qid,'instance_token':tok,'category':category,'prompt_frame':prompt,
                            'prompt_box':boxes[prompt],'frames':30,'input_folder':str(source),'role':'evaluation_only_full_Y',
                            'forbidden_for_condition_building':True}
                        quality_jobs[qid]=job
                        if tok in pm:
                            out=O/'quality_labels/observed_masks'/qid;out.mkdir(parents=True,exist_ok=True)
                            for i,m in enumerate(pm[tok]):Image.fromarray(m.astype('uint8')*255).save(out/f'{i:05}.png')
                            rec=job|{'reused_from':'r23 technically passed Y-SAM','quality':'pending_case_identity_review','human_verdict':None}
                            f.dump(out/'result.json',rec);quality_completed.append(rec)
                    if visible_areas[mprompt]>=100:
                        method_jobs.append({'case_id':cid,'job_id':cid+'_'+tok[:8],'instance_token':tok,'category':category,
                            'prompt_frame':mprompt,'prompt_box':boxes[mprompt],'frames':30,'input_folder':str(dest),
                            'role':'mask_planning_from_guard_erased_X','full_Y_access':False,
                            'target_RGB_access':False,'not_final_actor_state_features':True})
                row={'case_id':cid,'source_id':sid,'scene':c['scene'],'split':c['source_split'],'asset':asset,
                    'trajectory':tr,'camera':c['camera'],'frames':clean(c['frames']),'source_Y_quality_only':str(source/'Y'),
                    'observed_folder':str(dest),'retained_instances':identities,'original_rejection':why,
                    'candidate_H':'unchanged r21 rectangle','alternative_visible_H':'diagnostic only after r26 regression',
                    'training_admission':False,'human_verdict':None}
                row=clean(row)
                f.dump(dest/'case.json',row);cases.append(row)
                f.dump(O/'preparation_state.json',{'stage':'preparing','pid':os.getpid(),'prepared_cases':len(cases),
                    'method_SAM_jobs':len(method_jobs),'quality_SAM_jobs':len(quality_jobs),'seconds':time.time()-started})
                print('INSTANCE_PREP',cid,sid,len(identities),flush=True)
    assert len(cases)==budget,(len(cases),budget)
    f.dump(O/'quality_labels/observed_queue.json',{'jobs':list(quality_jobs.values()),'role':'Y_evaluation_only'})
    f.dump(O/'quality_labels/segmentation_state.json',{'completed':quality_completed,'stage':'pending_remaining_Y_labels'})
    f.dump(O/'method_masks/observed_queue.json',{'jobs':method_jobs,'role':'target_guard_erased_X_only'})
    f.dump(O/'prepared.json',{'cases':cases,'method_jobs':len(method_jobs),'quality_jobs':len(quality_jobs),
        'reused_quality_jobs':len(quality_completed),'training_admission':0,'training_steps':0})
    f.dump(O/'preparation_state.json',{'stage':'complete_pending_SAM','pid':os.getpid(),'prepared_cases':len(cases),
        'method_SAM_jobs':len(method_jobs),'quality_SAM_jobs':len(quality_jobs),'quality_reused':len(quality_completed),'seconds':time.time()-started})


if __name__=='__main__':prepare()
