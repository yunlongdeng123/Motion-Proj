"""r41只做元数据筛选：静止清楚B＋缓慢移动相机，先确认过程再提RGB。"""
from pathlib import Path
import sys,copy,random,shutil
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parent))
import space_source_pilot as p
import numpy as np
O=p.T/'r41';ROOT=O/'metadata_probe'
POLICY={'B_world_max_displacement_m':.5,'all8_visibility':'4','min_B_box_wh':[100,60],
    'max_front_box_overlap':.10,'max_ego_camera_travel_m':[1.,8.],'min_B_depth_m':16.,
    'new_scene_only':True,'max_sources':20,'max_per_scene':1,'frames':30,'split_seed':4101,
    'sampling_error_ms':55,'max_exposure_gap_ms':180,'class':'vehicle.car',
    'RGB_or_generation_outputs_used':False}


def main():
    O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r41','phase':'static_B_slow_ego_metadata_source_census',
        'question':'find the intended parallax process before expensive A/H synthesis, not just more arbitrary scenes',
        'policy':POLICY,'universe':'existing440 nuScenes train candidates, not new val/final test',
        'no_new_RGB_extraction_in_this_run':True,'no_new_SAM_or_training':True,'training_steps':0,
        'stop':'one fixed metadata filter; report deficits, do not tune thresholds to obtain a target count',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert p.read(O/'run.json')==plan
    else:p.dump(O/'run.json',plan)
    E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r41');E.mkdir(exist_ok=True,parents=True);shutil.copy2(O/'run.json',E/'run.json')
    old={c['scene'] for run in ['r8','r37'] for c in p.read(p.T/run/'factory/source_manifest.json')['clips']}
    old|=set(p.read(p.T/'r8/source_split.json')['old_r7_training_scenes'])
    clips=[];counts=Counter()
    for c in p.read(p.T/'r1/candidate_pool.json')['candidates']:
        if c['scene'] in old:continue
        for a in c['actors']:
            if a['category']!=POLICY['class']:continue
            ann=a['keyframe_annotations'];pr=a['keyframe_projections']
            if len(ann)!=8 or not all(ann) or len(pr)!=8 or not all(p.good(r) for r in pr):continue
            counts['candidate_actor_windows']+=1
            xyz=np.array([r['translation'] for r in ann]);motion=float(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1).max())
            if motion>POLICY['B_world_max_displacement_m']:counts['moving_B']+=1;continue
            if any(r['visibility_token']!=POLICY['all8_visibility'] for r in ann):counts['visibility']+=1;continue
            bb=np.array([r['box_xyxy'] for r in pr]);wh=bb[:,2:]-bb[:,:2]
            if np.any(wh.min(0)<POLICY['min_B_box_wh']):counts['small_keyframe_box']+=1;continue
            if a['max_front_box_overlap']>POLICY['max_front_box_overlap']:counts['front_overlap']+=1;continue
            if min(r['depth'] for r in pr)<POLICY['min_B_depth_m']:counts['too_near_B']+=1;continue
            row=copy.deepcopy(c);row['actors']=[copy.deepcopy(a)]+[copy.deepcopy(b) for b in c['actors'] if b['instance_token']!=a['instance_token']]
            row.update(source_id=f'J{len(clips)+1:03}',source_split='unassigned_before_final_scene_split',static_B_max_displacement_m=motion)
            clips.append(row)
    selection={'clips':clips,'policy':POLICY,'excluded_old_scenes':sorted(old),'counts':dict(counts)}
    p.dump(ROOT/'source_selection.json',selection)
    if not (ROOT/'source_manifest.json').exists():p.resolve(p.META,ROOT,selection,exposure_matcher=p.match_exposures)
    resolved=p.read(ROOT/'source_manifest.json');by=defaultdict(list);details=[]
    for c in resolved['clips']:
        cams=np.array([f['camera_to_world'] for f in c['frames']])[:,:3,3]
        travel=float(np.linalg.norm(np.diff(cams[:,:2],axis=0),axis=1).sum())
        reason=None
        if c['source_geometry_status']=='geometry_reject':reason='geometry_reject'
        if not POLICY['max_ego_camera_travel_m'][0]<=travel<=POLICY['max_ego_camera_travel_m'][1]:reason='ego_motion_not_slow_parallax'
        if reason is None:
            bb=np.array([f['actors'][0]['projection']['box_xyxy'] for f in c['frames']]);wh=bb[:,2:]-bb[:,:2]
            if np.any(wh.min(0)<POLICY['min_B_box_wh']):reason='small_between_keyframes'
        details.append({'source_id':c['source_id'],'scene':c['scene'],'camera':c['camera'],'camera_travel_m':travel,'reason':reason})
        if reason is None:c['measured_camera_travel_m']=travel;by[c['scene']].append(c)
    chosen=[]
    for scene,rows in sorted(by.items()):
        c=min(rows,key=lambda c:(abs(c['measured_camera_travel_m']-4.),c['start_keyframe'],c['camera'],c['source_id']));chosen.append(c)
    chosen=chosen[:POLICY['max_sources']];scenes=[c['scene'] for c in chosen];random.Random(POLICY['split_seed']).shuffle(scenes)
    val=set(scenes[:max(1,round(len(scenes)*.2))]) if scenes else set()
    for c in chosen:c['source_split']='validation' if c['scene'] in val else 'train'
    p.dump(O/'selected_sources.json',{'clips':chosen,'count':len(chosen),'scenes':len(scenes),'policy':POLICY,'no_RGB_review_yet':True,'human_verdict':None})
    summary={'metadata_actor_windows':len(clips),'resolved_windows':len(resolved['clips']),'sampling_rejected':len(resolved['sampling_rejected']),
        'selected_scenes':len(scenes),'split_counts':dict(Counter(c['source_split'] for c in chosen)),
        'required_shards':sorted({s for c in chosen for s in c['shards']}),'filters':dict(counts),
        'resolved_checks':details,'source_Y_quality_not_checked':True,'synthetic_training_admission':0,'training_steps':0}
    p.dump(O/'summary.json',summary);p.dump(E/'summary.json',summary)
    p.dump(O/'controller_state.json',{'stage':'metadata_census_complete_no_extraction','selected_sources':len(chosen),'training_steps':0})
    print('STATIC_PARALLAX_CENSUS', {k:v for k,v in summary.items() if k!='resolved_checks'},flush=True)


if __name__=='__main__':main()
