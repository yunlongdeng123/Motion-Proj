"""r46时间语义修复：保留r45现场，比较真实条件张量后决定是否重训。"""
from common import *
import argparse, shutil
import numpy as np


def stage():
    assert O.name=='r46'
    old=read(T/'r45/manifest.json');new=read(O/'manifest.json')
    paths={s['filename']:s['path'] for c in old['cases'] for s in c['scans'] if s.get('path')}
    for c in new['cases']:
        for s in c['scans']:
            if not s.get('path'):s['path']=paths[s['filename']]
            assert Path(s['path']).is_file()
    new.update(parent_run='r45',geometry_time_policy='official NuScenes.get_boxes for every actual camera sample_data',
        failure_ledger_delta='V77-F02: associated keyframes were discarded/interpolated by timestamp-only path; corrected uniformly')
    dump(O/'manifest.json',new)
    state=read(T/'r45/evaluation/state.json');state.update(stage='interrupted_for_condition_time_bug',
        completed_windows=len(state['completed']),new_run='r46',human_verdict=None)
    dump(T/'r45/evaluation/state.json',state)
    dump(O/'correction_plan.json',{'parent':'r45','scope':'all24 actual camera geometry; compare tensor before retrain decision',
        'checkpoint_if_reused':str(T/'r45/training/adapter_0160.safetensors'),
        'same_data_budget_modules_seed':True,'old_windows_preserved':len(state['completed']),'human_verdict':None})
    print('R46_STAGE_READY',len(state['completed']))


def compare():
    old={c['case_id']:c for c in read(T/'r45/manifest.json')['cases']}
    new=read(O/'manifest.json');rows=[];changed_train=[];changed_eval=[]
    for c in new['cases']:
        cid=c['case_id'];before=old[cid]
        assert all(c[k]==before[k] for k in ['kind','split','scene','type','folder','frame_indices','target_token'])
        assert all(fr[k]==bf[k] for fr,bf in zip(c['frames'],before['frames']) for k in ['timestamp','camera_to_world','intrinsics_1024'])
        changes=[]
        for i in range(10):
            a=np.load(T/'r45/conditions'/cid/f'{i:05}.npz');b=np.load(O/'conditions'/cid/f'{i:05}.npz')
            diff={k:int(np.count_nonzero(a[k]!=b[k])) for k in ['O','N','U','Q']}
            changes.append({'frame':i,'changed_pixels':diff,'old_actors':len(before['frames'][i]['actors']),
                'new_actors':len(c['frames'][i]['actors']),'keyframe':c['frames'][i]['is_key_frame'],
                'camera_minus_sample_us':c['frames'][i]['timestamp']-c['frames'][i]['sample_timestamp']})
        changed=any(any(fr['changed_pixels'].values()) for fr in changes)
        if changed:(changed_train if c['split']=='train' else changed_eval).append(cid)
        row={'case_id':cid,'split':c['split'],'condition_changed':changed,'frames':changes};rows.append(row)
    # 真实回归：不能把相机时间早于关联sample的有标注关键帧判成无世界。
    for cid in ['A022','A013','A007']:
        c=next(c for c in new['cases'] if c['case_id']==cid);fr=c['frames'][0]
        assert fr['is_key_frame'] and fr['timestamp']<fr['sample_timestamp'] and fr['actors']
        assert all(a['pose_source']=='SDK associated keyframe' for a in fr['actors'])
        a=np.load(O/'conditions'/cid/'00000.npz');assert (a['O']|a['N']).any() and not a['U'].all()
    need=bool(changed_train)
    # 原模型关闭分支完全不依赖O/N/U/Q；已有原图/H/alpha及时间和相机已逐项确认不变。
    reused=[]
    for c in new['cases']:
        if c['split']=='train':continue
        for arm in new['arms']:
            eligible=arm=='adapter_off' or (not need and (arm=='all_unknown' or c['case_id'] not in changed_eval))
            src=T/'r45/evaluation'/c['case_id']/arm;dst=O/'evaluation'/c['case_id']/arm
            if eligible and (src/'result.json').exists() and not dst.exists():
                shutil.copytree(src,dst);r=read(dst/'result.json');r['reused_from']='r45/'+c['case_id']+'/'+arm
                dump(dst/'result.json',r);reused.append({'case_id':c['case_id'],'arm':arm})
    if not need:
        (O/'training').symlink_to(T/'r45/training',target_is_directory=True)
    result={'stage':'time_fix_tensor_comparison_complete','need_retrain':need,'changed_train_cases':changed_train,
        'changed_eval_cases':changed_eval,'reuse_windows':reused,'new_windows_required':36-len(reused),
        'data_camera_H_unchanged':True,'regression_cases':['A022','A013','A007'],'cases':rows,'human_verdict':None}
    dump(O/'time_fix_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','compare']);a=p.parse_args()
    stage() if a.action=='stage' else compare()
