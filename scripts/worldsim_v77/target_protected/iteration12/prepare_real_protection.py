"""r26统一真实DEV入口：目标膨胀影响先擦除，SAM可见邻车约束H。"""
from pathlib import Path
import sys,os,argparse
sys.path.insert(0,str(Path(__file__).parent))
from refine_visible_hole import refine_hole,read,dump
import numpy as np
import cv2
from PIL import Image,ImageDraw
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r26'


def prepare():
    O.mkdir(exist_ok=True);assert not (O/'observed_queue.json').exists()
    audit=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1')
    geometry={c['clip_id']:c for c in read(audit/'clip_geometry.json')['clips']}
    cases=read(T/'r21/real_input_plan.json')['cases'];jobs=[]
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r26','role':'engineering_input_control_not_new_model',
        'source_cases':[c['eval_id'] for c in cases],'arms':['base','r7'],'frames':10,'resolution':[576,1024],
        'seed':42,'steps':25,'previous_condition':False,'max_new_windows':16,
        'new_rule':'H_v2 minus visible retained masks inferred from target-write-mask-erased RGB',
        'target_guard':'r21 full SAM plus 7px ellipse write mask; never GT-clipped',
        'sampling_rule':'all existing8 realDEV fixed windows; no generation-based selection',
        'init_or_backbone_change':False,'Y_or_hidden_GT':False,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    dump(O/'run.json',plan)
    for c in cases:
        cid=c['eval_id'];src=Path(c['folder']);ids=c['frames'];out=O/'observed'/cid
        for role in ['rgb','H','sam_rgb']:(out/role).mkdir(parents=True,exist_ok=True)
        frames=[geometry[c['clip_id']]['frames'][i] for i in ids];guard=[];holes=[]
        for j,i in enumerate(ids):
            x=np.asarray(Image.open(src/'rgb'/f'{i:05}.jpg').convert('RGB')).copy()
            core=np.asarray(Image.open(src/'write_mask'/f'{i:05}.png'))>0
            h=np.asarray(Image.open(src/'model_mask'/f'{i:05}.png'))>0
            x[core]=127;guard.append(core);holes.append(h)
            Image.fromarray(x).save(out/'rgb'/f'{j:05}.png');Image.fromarray(core.astype('uint8')*255).save(out/'H'/f'{j:05}.png')
            Image.fromarray(x).save(out/'sam_rgb'/f'{j:05}.jpg',quality=98,subsampling=0)
        candidates={}
        for j,f in enumerate(frames):
            for n in f['neighbors']:
                tok=n['instance_token'];bb=np.asarray(n['box_xyxy']);x0,y0,x1,y1=np.rint(bb).astype(int);x0,x1=np.clip([x0,x1],0,1024);y0,y1=np.clip([y0,y1],0,576)
                if x1<=x0 or y1<=y0:continue
                visible=(~guard[j])[y0:y1,x0:x1];touch=holes[j][y0:y1,x0:x1]&visible
                if touch.sum()<20 or visible.sum()<100:continue
                candidate={'frame':j,'box':bb.tolist(),'visible_bbox_pixels':int(visible.sum()),'H_overlap':int(touch.sum())}
                if tok not in candidates or candidate['visible_bbox_pixels']>candidates[tok]['visible_bbox_pixels']:candidates[tok]=candidate
        for tok,q in sorted(candidates.items()):
            jobs.append({'case_id':cid,'job_id':cid+'_'+tok[:8],'instance_token':tok,'prompt_frame':q['frame'],
                'prompt_box':q['box'],'input_folder':str(out),'frames':10,'selection':'all neighbor envelopes touching H outside target guard; maximum visible bbox area prompt'})
        dump(out/'meta.json',{'case':c,'geometry_frames':frames,'target_guard':'write_mask_v2','neighbor_jobs':len(candidates)})
    dump(O/'observed_queue.json',{'jobs':jobs,'no_Y_used':True,'no_target_RGB_used':True,'method':'mask planning only, not state feature extraction'})
    print('REAL_PROTECTED_QUEUE',len(jobs))


def finish():
    assert read(O/'segmentation_state.json')['stage']=='complete_pending_identity_review'
    jobs=read(O/'observed_queue.json')['jobs'];cases=read(T/'r21/real_input_plan.json')['cases'];rows=[]
    contacts=O/'contacts';contacts.mkdir(exist_ok=True)
    for c in cases:
        cid=c['eval_id'];src=Path(c['folder']);dest=O/'real_inputs'/c['clip_id'];ids=c['frames'];metrics=[];panels=[]
        for role in ['rgb','model_mask','alpha','visible_protected','target_guard']:(dest/role).mkdir(parents=True,exist_ok=True)
        for j,i in enumerate(ids):
            im=np.asarray(Image.open(src/'rgb'/f'{i:05}.jpg').convert('RGB'))
            guard=np.asarray(Image.open(src/'write_mask'/f'{i:05}.png'))>0
            oldH=np.asarray(Image.open(src/'model_mask'/f'{i:05}.png'))>0
            oldalpha=np.asarray(Image.open(src/'alpha'/f'{i:05}.png')).astype('float32')/255
            visible=np.zeros_like(guard)
            for job in jobs:
                if job['case_id']==cid:visible|=np.asarray(Image.open(O/'observed_masks'/job['job_id']/f'{j:05}.png'))>0
            h=refine_hole(guard,visible,oldH);alpha=oldalpha.copy();alpha[~h]=0
            assert np.all(h[guard]) and np.all(alpha[guard]==1)
            for role,arr in [('model_mask',h.astype('uint8')*255),('alpha',np.rint(alpha*255).astype('uint8')),
                             ('visible_protected',visible.astype('uint8')*255),('target_guard',guard.astype('uint8')*255)]:Image.fromarray(arr).save(dest/role/f'{i:05}.png')
            if not (dest/'rgb'/f'{i:05}.jpg').exists():os.link(src/'rgb'/f'{i:05}.jpg',dest/'rgb'/f'{i:05}.jpg')
            metrics.append({'frame':i,'target_guard_leak':0,'target_guard_alpha_not_one':0,'observed_pixels_preserved':int((oldH&~h).sum()),'known_protected_outside_core_only':True})
            if j in [0,5,9]:
                label=im.copy();label[guard]=np.rint(label[guard]*.4+np.array([255,180,20])*.6).astype('uint8')
                cv2.drawContours(label,cv2.findContours(visible.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(50,230,120),2)
                before=im.copy();before[oldH]=127;after=im.copy();after[h]=127;panels.append((j,[label,before,after]))
        sheet=Image.new('RGB',(1536,970),(18,24,32));draw=ImageDraw.Draw(sheet)
        for r,(j,p) in enumerate(panels):
            for k,arr in enumerate(p):
                draw.text((k*512+6,r*323+5),cid+f' f{j} '+['A guard / visible neighbors','old H input','new H input'][k],fill='white')
                sheet.paste(Image.fromarray(arr).resize((512,288)),(k*512,r*323+26))
        sheet.save(contacts/f'{cid}.jpg',quality=95)
        rows.append(c|{'folder':str(dest),'previous_folder':str(src),'mask_policy':'visible_protection_v1','metrics':metrics,'independent_input_review':'pending'})
    dump(O/'real_input_plan.json',{'cases':rows,'baseline':str(T/'r21/baseline'),'new_model_training':False})
    print('REAL_PROTECTED_INPUTS',len(rows),'restored',sum(m['observed_pixels_preserved'] for r in rows for m in r['metrics']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','finish']);a=p.parse_args()
    prepare() if a.phase=='prepare' else finish()
