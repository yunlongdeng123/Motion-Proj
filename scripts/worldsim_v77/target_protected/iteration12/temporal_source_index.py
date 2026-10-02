"""r42：仅取消旧每scene前三窗口截断，检查是否丢失既定时序过程。"""
from pathlib import Path
import sys,copy,random,time,traceback,shutil,fcntl,os
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parent))
import static_parallax_sources as prior
import space_source_pilot as p
from prepare_source_pool import make_index
import numpy as np

O=p.T/'r42';ROOT=O/'metadata_probe';E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r42')


def key(c):return (c['scene'],c['camera'],c['start_keyframe'])


def metadata_filter(pool,old,policy=None):
    clips=[];counts=Counter();policy=prior.POLICY if policy is None else policy
    for c in pool:
        if c['scene'] in old:continue
        for a in c['actors']:
            if a['category']!=policy['class']:continue
            ann=a['keyframe_annotations'];pr=a['keyframe_projections']
            if len(ann)!=8 or not all(ann) or len(pr)!=8 or not all(p.good(r) for r in pr):continue
            counts['candidate_actor_windows']+=1
            xyz=np.array([r['translation'] for r in ann]);motion=float(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1).max())
            if motion>policy['B_world_max_displacement_m']:counts['moving_B']+=1;continue
            if any(r['visibility_token']!=policy['all8_visibility'] for r in ann):counts['visibility']+=1;continue
            bb=np.array([r['box_xyxy'] for r in pr]);wh=bb[:,2:]-bb[:,:2]
            if np.any(wh.min(0)<policy['min_B_box_wh']):counts['small_keyframe_box']+=1;continue
            if a['max_front_box_overlap']>policy['max_front_box_overlap']:counts['front_overlap']+=1;continue
            if policy['min_B_depth_m'] is not None and min(r['depth'] for r in pr)<policy['min_B_depth_m']:counts['too_near_B']+=1;continue
            row=copy.deepcopy(c);row['actors']=[copy.deepcopy(a)]+[copy.deepcopy(b) for b in c['actors'] if b['instance_token']!=a['instance_token']]
            row.update(source_id=f'K{len(clips)+1:03}',source_split='unassigned_before_final_scene_split',static_B_max_displacement_m=motion)
            clips.append(row)
    return clips,counts


def main():
    O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True);E.mkdir(exist_ok=True,parents=True)
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r42','phase':'candidate_truncation_control',
        'change':'only remove per-scene top-three truncation in the old metadata index; all per-actor and source gates unchanged',
        'control':'r41 saved 440-window metadata pool','policy':prior.POLICY,
        'input_role':'official train metadata and GT camera/track POC; no RGB, hidden condition appearance or model output',
        'resource_budget':'one CPU metadata pass; no archive extraction or GPU inference in this run',
        'stop':'one untruncated pass; do not weaken static/visibility/depth/camera-travel requirements',
        'training_steps':0,'training_admission':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert p.read(O/'run.json')==plan
    else:p.dump(O/'run.json',plan)
    p.dump(E/'run.json',plan)
    state={'stage':'full_metadata_index','pid':os.getpid(),'started':time.time(),'training_steps':0};p.dump(O/'controller_state.json',state)
    try:
        assert shutil.disk_usage(O).free>30*2**30
        old_pool=p.read(p.T/'r1/candidate_pool.json')['candidates']
        if not (ROOT/'candidate_pool.json').exists():
            indexed=make_index(p.META,ROOT,max_per_scene=None)
            indexed.update(run_id='r42',generator_schema='legacy_prepare_source_pool_r1',max_per_scene=None)
            p.dump(ROOT/'candidate_pool.json',indexed)
        pool=p.read(ROOT/'candidate_pool.json')['candidates'];by=defaultdict(list)
        for c in pool:by[c['scene']].append(c)
        replay=[c for rows in by.values() for c in rows[:3]]
        old_by={key(c):c for c in old_pool};replay_by={key(c):c for c in replay}
        assert old_by==replay_by,'同元数据重新前三截断未重现旧池，先修工程差异'
        old={c['scene'] for run in ['r8','r37'] for c in p.read(p.T/run/'factory/source_manifest.json')['clips']}
        old|=set(p.read(p.T/'r8/source_split.json')['old_r7_training_scenes'])
        # 同一纯函数必须重现r41门前9条与各拒绝计数，防止顺手改变来源标准。
        baseline,baseline_counts=metadata_filter(old_pool,old);r41=p.read(p.T/'r41/summary.json')
        assert len(baseline)==r41['metadata_actor_windows'] and dict(baseline_counts)==r41['filters']
        clips,counts=metadata_filter(pool,old)
        p.dump(ROOT/'source_selection.json',{'clips':clips,'policy':prior.POLICY,'excluded_old_scenes':sorted(old)})
        comparison={'old_windows':len(old_pool),'untruncated_windows':len(pool),'old_top_three_exactly_reproduced':True,
            'same_eligible_scene_universe':set(by)=={c['scene'] for c in old_pool},'r41_filter_exactly_reproduced':True,
            'same_per_actor_gates':True,'same_scene_exclusions':True,'metadata_windows_before':len(baseline),'metadata_windows_after':len(clips)}
        p.dump(O/'index_comparison.json',comparison);state.update(stage='unique_exposure_resolve',metadata_windows=len(clips));p.dump(O/'controller_state.json',state)
        if not (ROOT/'source_manifest.json').exists():p.resolve(p.META,ROOT,{'clips':clips},exposure_matcher=p.match_exposures)
        resolved=p.read(ROOT/'source_manifest.json');eligible=defaultdict(list);details=[];policy=prior.POLICY
        for c in resolved['clips']:
            cams=np.array([f['camera_to_world'] for f in c['frames']])[:,:3,3]
            travel=float(np.linalg.norm(np.diff(cams[:,:2],axis=0),axis=1).sum());reasons=[]
            if c['source_geometry_status']=='geometry_reject':reasons.append('geometry_reject')
            if not policy['max_ego_camera_travel_m'][0]<=travel<=policy['max_ego_camera_travel_m'][1]:reasons.append('ego_motion_not_slow_parallax')
            if not reasons:
                bb=np.array([f['actors'][0]['projection']['box_xyxy'] for f in c['frames']]);wh=bb[:,2:]-bb[:,:2]
                if np.any(wh.min(0)<policy['min_B_box_wh']):reasons.append('small_between_keyframes')
            details.append({'source_id':c['source_id'],'scene':c['scene'],'camera':c['camera'],'camera_travel_m':travel,
                'in_old_top_three':key(c) in old_by,'reasons':reasons})
            if not reasons:c['measured_camera_travel_m']=travel;eligible[c['scene']].append(c)
        chosen=[]
        for scene,rows in sorted(eligible.items()):
            chosen.append(min(rows,key=lambda c:(abs(c['measured_camera_travel_m']-4.),c['start_keyframe'],c['camera'],c['source_id'])))
        chosen=chosen[:policy['max_sources']];scenes=[c['scene'] for c in chosen];random.Random(policy['split_seed']).shuffle(scenes)
        val=set(scenes[:max(1,round(len(scenes)*.2))]) if scenes else set()
        for c in chosen:c['source_split']='validation' if c['scene'] in val else 'train'
        p.dump(O/'selected_sources.json',{'clips':chosen,'policy':policy,'no_RGB_review_yet':True,'training_admission':0,'human_verdict':None})
        summary={'metadata_actor_windows':len(clips),'resolved_windows':len(resolved['clips']),
            'sampling_rejected':len(resolved['sampling_rejected']),'selected_scenes':len(scenes),'filters':dict(counts),
            'resolved_checks':details,'required_shards':sorted({s for c in chosen for s in c['shards']}),
            'split_counts':dict(Counter(c['source_split'] for c in chosen)),'source_Y_quality_not_checked':True,
            'training_admission':0,'training_steps':0,'human_verdict':None}
        p.dump(O/'summary.json',summary)
        for name in ['index_comparison.json','summary.json']:p.dump(E/name,p.read(O/name))
        state.update(stage='bounded_metadata_control_complete',selected_scenes=len(scenes),seconds=time.time()-state['started']);p.dump(O/'controller_state.json',state)
        print('R42_INDEX',comparison,flush=True);print('R42_SOURCES',{k:v for k,v in summary.items() if k!='resolved_checks'},flush=True)
    except Exception as error:
        state.update(stage='engineering_error',error=repr(error),traceback=traceback.format_exc());p.dump(O/'controller_state.json',state);raise


if __name__=='__main__':main()
