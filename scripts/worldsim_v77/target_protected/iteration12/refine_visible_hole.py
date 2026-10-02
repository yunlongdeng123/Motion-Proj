"""r25：目标已擦除后的可见保护实例，约束外扩H；不读取隐藏Y。

准备/改洞方法无Y接口；评价程序才读取Y和完整视频SAM。
"""
from pathlib import Path
import os, sys, json, argparse
import numpy as np
from PIL import Image
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0]=[str(S),str(S/'iteration12'),str(S.parent/'delete_audit')]
from geometry_factory import read,dump
from mask_contract import prepare_masks
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r25'
PROBES=[('R001','r23','r23'),('L001','r16','r16'),('L009','r16','r16')]


def refine_hole(core, visible, proposed_hole):
    core=np.asarray(core,bool);visible=np.asarray(visible,bool);proposed_hole=np.asarray(proposed_hole,bool)
    if np.any(core & visible):raise ValueError('可见保护实例与完整目标影响区冲突')
    if not np.all(proposed_hole[core]):raise ValueError('外扩洞未完整覆盖目标影响')
    hole=proposed_hole & ~visible
    assert np.all(hole[core]) and not (hole & visible).any()
    return hole


def prepare():
    O.mkdir(exist_ok=True)
    assert not (O/'observed_queue.json').exists(),'不覆盖既有输入'
    dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r25',
        'phase':'visible_protection_mask_engineering','cases':[p[0] for p in PROBES],
        'mask_source':'SAM2 on X with full target influence already erased',
        'forbidden_input':'hidden Y / Y masks to method','same_window_frames':30,
        'candidate_H':'sam_full_v2 on complete influence',
        'final_H':'candidate_H minus SAM-visible retained actor outside influence',
        'method_not_default_until_checked':True,'training_steps':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None})
    jobs=[]
    for cid,run,srun in PROBES:
        original=T/run/'synthetic'/cid;pair=read(original/'pair_manifest.json')
        c=next(c for c in read(T/srun/'factory/source_manifest.json')['clips'] if c['source_id']==pair['source_id'])
        out=O/'observed'/cid
        for role in ['rgb','H','sam_rgb']:(out/role).mkdir(parents=True,exist_ok=True)
        holes=[]
        for i in range(30):
            x=np.asarray(Image.open(original/'X'/f'{i:03}.png').convert('RGB')).copy()
            core=np.asarray(Image.open(original/'influence'/f'{i:03}.png'))>0
            x[core]=127;holes.append(core)
            Image.fromarray(x).save(out/'rgb'/f'{i:05}.png');Image.fromarray(core.astype('uint8')*255).save(out/'H'/f'{i:05}.png')
            Image.fromarray(x).save(out/'sam_rgb'/f'{i:05}.jpg',quality=98,subsampling=0)
        meta={'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'frames':c['frames'],'actors':c['actors'],
            'source_run':run,'source_factory':srun,'source_pair':str(original),'RGB_input':'X erased by full influence BEFORE SAM JPEG',
            'gt_read_for_method':False,'window_start':c['frames'][0]['timestamp'],'window_end':c['frames'][-1]['timestamp']}
        dump(out/'observations.json',meta)
        for actor in c['actors']:
            tok=actor['instance_token'];areas=[];boxes=[]
            for fr,h in zip(c['frames'],holes):
                a=next(a for a in fr['actors'] if a['instance_token']==tok);bb=np.array(a['projection']['box_xyxy'])
                x0,y0,x1,y1=np.rint(bb).astype(int);x0,x1=np.clip([x0,x1],0,1024);y0,y1=np.clip([y0,y1],0,576)
                areas.append(int((~h[y0:y1,x0:x1]).sum()));boxes.append(bb.tolist())
            prompt=int(np.argmax(areas))
            if areas[prompt]<100:continue
            jobs.append({'case_id':cid,'instance_token':tok,'job_id':cid+'_'+tok[:8],
                'prompt_frame':prompt,'prompt_box':boxes[prompt],'input_folder':str(out),'frames':30,
                'selection':'maximum unmasked bbox area, no Y read','no_Y_or_full_RGB_access':True})
    dump(O/'observed_queue.json',{'jobs':jobs,'purpose':'mask planning only; actor state later uses final masked RGB'})


def finish():
    assert read(O/'segmentation_state.json')['stage']=='complete_pending_identity_review'
    jobs=read(O/'observed_queue.json')['jobs'];rows=[]
    for cid,run,srun in PROBES:
        meta=read(O/'observed'/cid/'observations.json');folder=Path(meta['source_pair']);out=O/'refined'/cid
        for role in ['model_hole','condition','protected_visible','proposed_H']:(out/role).mkdir(parents=True,exist_ok=True)
        metrics=[]
        for i in range(30):
            core=np.asarray(Image.open(folder/'influence'/f'{i:03}.png'))>0
            x=np.asarray(Image.open(folder/'X'/f'{i:03}.png').convert('RGB')).copy()
            visible=np.zeros_like(core)
            for job in jobs:
                if job['case_id']==cid:visible|=np.asarray(Image.open(O/'observed_masks'/job['job_id']/f'{i:05}.png'))>0
            assert not np.any(visible&core)
            proposed=prepare_masks(core)[0]['model_mask'];h=refine_hole(core,visible,proposed)
            cc=x.copy();cc[h]=127
            # 完整target影响区修改不能穿过mask。对照真实磁盘条件，禁止先resize后擦除。
            alt=x.copy();alt[core]=255-alt[core];alt[h]=127;assert np.array_equal(cc,alt)
            for role,arr in [('model_hole',h.astype('uint8')*255),('condition',cc),('protected_visible',visible.astype('uint8')*255),('proposed_H',proposed.astype('uint8')*255)]:
                Image.fromarray(arr).save(out/role/f'{i:03}.png')
            assert np.all(np.asarray(Image.open(out/'model_hole'/f'{i:03}.png'))[core]==255)
            metrics.append({'frame':i,'target_outside_H':0,'target_RGB_mutation_changes_condition':False,
                'proposed_H_visible_protection_pixels':int((proposed&visible).sum()),
                'refined_H_visible_protection_pixels':0,'restored_observed_pixels':int((proposed&~h).sum())})
        row={'case_id':cid,'source_id':meta['source_id'],'scene':meta['scene'],'metrics':metrics,
             'read_hidden_Y_for_method':False,'mask_default_promoted':False,'human_verdict':None}
        dump(out/'result.json',row);rows.append(row)
    dump(O/'method_result.json',{'cases':rows,'stage':'complete_pending_independent_QA','training_steps':0,'human_verdict':None})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','finish']);p=a.parse_args()
    prepare() if p.phase=='prepare' else finish()
