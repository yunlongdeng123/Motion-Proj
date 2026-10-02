"""r28逐像素输入/队列合同，检查隔离与覆盖，不给模型效果或人工评分。"""
from pathlib import Path
import sys,json,time
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from prepare_instance_audit import O


def read(p):return json.loads(p.read_text())


def main():
    plan=read(O/'prepared.json');roster=read(O/'candidate_roster.json');rows=[];start=time.time()
    ids=[x['case_id'] for x in plan['cases']]
    assert len(ids)==len(set(ids))==roster['count']
    ycache={};scene_splits={}
    for c in plan['cases']:
        folder=Path(c['observed_folder']);source=Path(c['source_Y_quality_only']);errors=[]
        stamps=[x['timestamp'] for x in c['frames']]
        assert len(stamps)==30 and np.diff(stamps).min()>0
        scene_splits.setdefault(c['scene'],set()).add(c['split'])
        if str(source) not in ycache:ycache[str(source)]=[np.asarray(Image.open(source/f'{i:05}.png').convert('RGB')) for i in range(30)]
        counts=Counter()
        for i,y in enumerate(ycache[str(source)]):
            def mask(role):return np.asarray(Image.open(folder/role/f'{i:05}.png'))>0
            m=mask('influence');g=mask('H');h=mask('proposal_H');x=np.asarray(Image.open(folder/'rgb'/f'{i:05}.png').convert('RGB'))
            assert m.shape==g.shape==h.shape==(576,1024) and x.shape==y.shape==(576,1024,3)
            counts['influence_outside_guard']+=int((m&~g).sum())
            counts['guard_outside_final_H']+=int((g&~h).sum())
            counts['changed_pixels_outside_guard']+=int(((x!=y).any(-1)&~g).sum())
            counts['uncleared_guard_pixels']+=int(((x!=127).any(-1)&g).sum())
            counts['H_in_ego_band']+=int(h[512:].sum())
        if any(counts.values()):errors.append('pixel_contract')
        tr=c['trajectory']
        if tr['min_GT_clearance_m']<.3 or tr['max_ground_support_distance_m']>2.5:errors.append('geometry')
        rows.append({'case_id':c['case_id'],'passed':not errors,'errors':errors,'counts':dict(counts),'frames':30,
                     'span_seconds':(stamps[-1]-stamps[0])/1e6,'scope':'mask-planning RGB; final condition must additionally erase proposal_H'})
    assert all(len(s)==1 for s in scene_splits.values()),'source scene crossed train/validation'
    quality=read(O/'quality_labels/observed_queue.json')['jobs'];method=read(O/'method_masks/observed_queue.json')['jobs']
    for j in quality:
        assert j['role']=='evaluation_only_full_Y' and j['forbidden_for_condition_building']
        assert Path(j['input_folder']).is_relative_to(O/'quality_labels/sources')
    for j in method:
        assert j['role']=='mask_planning_from_guard_erased_X' and not j['full_Y_access'] and not j['target_RGB_access']
        assert j['not_final_actor_state_features'] and Path(j['input_folder']).is_relative_to(O/'observed')
    result={'all_passed':all(r['passed'] for r in rows),'cases':rows,'count':len(rows),'decoded_RGB_frames':len(rows)*30,
            'quality_jobs':len(quality),'method_jobs_not_run':len(method),'world_split_overlap':0,
            'seconds':time.time()-start,'training_admission':0,'human_verdict':None,
            'boundary':'工程合同通过不等于实例分割正确、数据可训练或DELETE有效；方法状态必须由最终H遮后RGB重建'}
    (O/'input_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('INPUT_VALIDATION',result['all_passed'],len(rows),len(rows)*30,flush=True)
    assert result['all_passed']


if __name__=='__main__':main()
