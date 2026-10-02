"""仅恢复已知但画面外的3D位姿；不外推缺失GT，不修改任何RGB、H或SAM。"""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent))
import prepare_instance_audit as p
f=p.f;O=p.O


def main():
    assert f.read(O/'controller_state.json')['stage']=='Y_evaluation_SAM','不能与质量判定并发改元数据'
    b=O/'before_pose_fix';b.mkdir(exist_ok=True)
    for name in ['prepared.json','instance_quality_ready.json','instance_quality_partial.json']:
        path=O/name
        if path.exists() and not (b/name).exists():shutil.copy2(path,b/name)
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry();plan=f.read(O/'prepared.json');changes=[]
    for c in plan['cases']:
        sid=c['source_id']
        if sid not in g.obstacles:g.prepare(sid)
        touched=[]
        for actor in c['retained_instances']:
            frames=[]
            for i,(annotations,old) in enumerate(zip(g.obstacles[sid],actor['annotations'])):
                ob=next((o for o in annotations if o['instance_token']==actor['instance_token']),None)
                if old is None and ob is not None:
                    assert ob['_projection'] is None and actor['boxes'][i] is None
                    actor['annotations'][i]=p.clean(ob);frames.append(i)
            if frames:touched.append({'instance':actor['instance_token'],'category':actor['category'],'frames':frames,
                'complete_known_track':all(a is not None and not a.get('interpolation_uncertain',False) for a in actor['annotations'])})
        if touched:
            dest=Path(c['observed_folder']);backup=dest/'case_before_pose_fix.json'
            if not backup.exists():shutil.copy2(dest/'case.json',backup)
            c['pose_metadata_rule']='known_3D_pose_retained_even_without_2D_projection'
            f.dump(dest/'case.json',c);changes.append({'case_id':c['case_id'],'actors':touched})
    f.dump(O/'prepared.json',plan)
    result={'changes':changes,'cases':len(changes),'case_instances':sum(len(r['actors']) for r in changes),
        'pixel_and_mask_changes':0,'SAM_restarts':0,'missing_world_pose_extrapolated':False,
        'source':'already available GT POC trajectory; not image-derived new evidence',
        'scope':'r28 metadata bug, not an explanation of old r7/r14 model failure'}
    f.dump(O/'pose_metadata_repair.json',result);print('POSE_REPAIR',len(changes),result['case_instances'],flush=True)


if __name__=='__main__':main()
