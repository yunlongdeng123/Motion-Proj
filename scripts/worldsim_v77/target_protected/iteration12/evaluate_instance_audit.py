"""r28固定候选的Y实例质检；只解除包络假交叠，不放宽真实遮挡/空间规则。

完整Y-SAM只决定离线候选资格，绝不保存为actor-state或推理条件。
先做本检查；失败候选不再花GPU跑方法侧SAM。
"""
from pathlib import Path
import sys,copy
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import reveal_factory as f
import numpy as np
from PIL import Image,ImageDraw

O=f.T/'r28'


def main(ready_only=False,frame_start=0,frame_count=30,output_root=O,case_ids=None,reveal_policy='all_affected'):
    plan=f.read(O/'prepared.json');label_state=f.read(O/'quality_labels/segmentation_state.json')
    if not ready_only:assert label_state['stage']=='complete_pending_identity_review'
    assert 0<=frame_start and frame_start+frame_count<=30 and frame_count in [20,30]
    if case_ids is not None:plan['cases']=[c for c in plan['cases'] if c['case_id'] in set(case_ids)]
    output_root.mkdir(exist_ok=True,parents=True)
    assert not (output_root/'instance_quality.json').exists(),'已有评价不能覆盖'
    window=slice(frame_start,frame_start+frame_count)
    completed={r['job_id'] for r in label_state['completed']}
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry();rows=[]
    contacts=output_root/'instance_contacts';contacts.mkdir(exist_ok=True)
    cached={}
    for case in plan['cases']:
        if ready_only and any(a['quality_input_available'] and a['Y_quality_job'] not in completed for a in case['retained_instances']):continue
        cid=case['case_id'];sid=case['source_id'];c=g.sources[sid]
        if sid not in g.obstacles:g.prepare(sid)
        original_obs=g.obstacles[sid];obs=copy.deepcopy(original_obs[window]);aug=copy.deepcopy(c);aug['frames']=aug['frames'][window]
        masks=[np.asarray(Image.open(Path(case['observed_folder'])/'influence'/f'{i:05}.png'))>0 for i in range(frame_start,frame_start+frame_count)]
        holes=[np.asarray(Image.open(Path(case['observed_folder'])/'proposal_H'/f'{i:05}.png'))>0 for i in range(frame_start,frame_start+frame_count)]
        pm={t:ms[window] for t,ms in f.legacy.protections(g,sid).items()};all_masks={};identity_flags=[];scope_flags=[];removed_envelopes=0
        for actor in case['retained_instances']:
            tok=actor['instance_token'];jid=actor['Y_quality_job'];path=O/'quality_labels/observed_masks'/jid
            if not actor['quality_input_available'] or not path.is_dir():
                identity_flags.append({'instance':tok,'issue':'no_usable_visible_GT_box'});continue
            if jid not in cached:cached[jid]=[np.asarray(Image.open(path/f'{i:05}.png'))>0 for i in range(30)]
            mm=cached[jid][window];all_masks[tok]=mm
            if not any(m.any() for m in mm):
                identity_flags.append({'instance':tok,'issue':'all_empty_SAM_not_evidence_of_absence'});continue
            all_poses=all(a is not None and not a.get('interpolation_uncertain',False) for a in actor['annotations'][window])
            if actor['category'].startswith('vehicle.') and all_poses:
                pm[tok]=mm
                for i,fr in enumerate(aug['frames']):
                    if not any(a['instance_token']==tok for a in fr['actors']):fr['actors'].append(actor['annotations'][i+frame_start])
            elif actor['category'] in {'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}:
                pass  # 后续沿用原静态物体深度检查；不能把它们当新保护车。
            elif any((h&m).any() for h,m in zip(holes,mm)):
                scope_flags.append({'instance':tok,'category':actor['category'],'issue':'H touches nonvehicle or incomplete geometry; outside first-stage scope'})
            # 只有真实可见mask与H完全无交叠的当帧，才解除GT矩形包络的假阳性。
            for i,objects in enumerate(obs):
                if (holes[i]&mm[i]).any():continue
                for ob in objects:
                    if ob['instance_token']==tok and ob['_projection'] is not None:
                        ob['_projection']=None;removed_envelopes+=1
        tokens=sorted(all_masks)
        for j,a in enumerate(tokens):
            for b in tokens[j+1:]:
                overlaps=[float((x&y).sum()/max(1,(x|y).sum())) for x,y in zip(all_masks[a],all_masks[b])]
                if sum(v>.35 for v in overlaps)>=3:
                    identity_flags.append({'instances':[a,b],'issue':'large_SAM_instance_overlap','max_iou':max(overlaps)})
        if identity_flags:quality=None;why='instance_identity_or_visibility_uncertain'
        elif scope_flags:quality=None;why='nonvehicle_or_incomplete_protection'
        else:
            g.obstacles[sid]=obs
            tr=dict(case['trajectory']);tr['frames']=tr['frames'][window]
            try:quality,why=f.exact(g,aug,tr,masks,pm,reveal_policy=reveal_policy)
            finally:g.obstacles[sid]=original_obs
        outcome={'case_id':cid,'source_id':sid,'scene':case['scene'],'split':case['split'],
            'technical_candidate':quality is not None,'reason':why,'quality':quality,
            'removed_empty_H_envelopes':removed_envelopes,'identity_flags':identity_flags,'scope_flags':scope_flags,
            'Y_masks_role':'evaluation_only_not_condition','independent_quality':'pending_if_selected','training_admission':False,'human_verdict':None,
            'frame_start':frame_start,'frame_count':frame_count}
        rows.append(outcome)
        if quality is not None:
            sheet=Image.new('RGB',(1536,966),(18,24,32));draw=ImageDraw.Draw(sheet)
            for r,i in enumerate([0,frame_count//2,frame_count-1]):
                y=np.asarray(Image.open(Path(case['source_Y_quality_only'])/f'{i+frame_start:05}.png')).copy()
                label=y.copy();label[masks[i]]=np.rint(label[masks[i]]*.3+np.array([255,190,30])*.7).astype('uint8')
                import cv2
                for tok in quality['protected_instances']:
                    cv2.drawContours(label,cv2.findContours(pm[tok][i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(35,235,110),2)
                model=y.copy();model[holes[i]]=127
                for k,(role,im) in enumerate([('Y evaluation',y),('A yellow / active B green',label),('unchanged H input',model)]):
                    draw.text((k*512+5,r*322+7),f'{cid} source f{i+frame_start} '+role,fill='white');sheet.paste(Image.fromarray(im).resize((512,288)),(k*512,r*322+28))
            sheet.save(contacts/f'{cid}.jpg',quality=95)
        f.dump(output_root/'instance_quality_partial.json',{'cases':rows,'training_admission':0})
        print('INSTANCE_QA',cid,quality['data_family'] if quality else why,flush=True)
    summary={'cases':rows,'counts':dict(Counter(r['quality']['data_family'] if r['quality'] else r['reason'] for r in rows)),
        'input_rule_unchanged':True,'training_admission':0,'new_training_steps':0,
        'boundary':'SAM empty alone cannot certify object-free; positive candidates still need independent visual QA'}
    summary['complete']=not ready_only
    summary['pending_cases']=len(plan['cases'])-len(rows)
    summary.update(frame_start=frame_start,frame_count=frame_count,reveal_policy=reveal_policy,Y_annotation_context_frames=30,
                   method_window_must_be_rebuilt=True,source_run='r28')
    f.dump(output_root/('instance_quality_ready.json' if ready_only else 'instance_quality.json'),summary)
    print('INSTANCE_AUDIT_READY' if ready_only else 'INSTANCE_AUDIT_DONE',summary['counts'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--ready-only',action='store_true')
    main(parser.parse_args().ready_only)
