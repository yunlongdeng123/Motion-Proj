"""独立QA后冻结最多50训练例，优先场景覆盖；拒绝/待定完整保留。"""
from pathlib import Path
import sys,copy
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read,dump

def select(cases,limit=50):
    chosen=[];scene_counts=Counter()
    # 先每个scene一例，密集/单保护优先；再按scene轮转填补，最多3例。
    priority={'dense_actors':0,'single_actor':1,'background':2}
    ordered=sorted(cases,key=lambda c:(priority[c['type']],c['case_id']))
    for rank in range(3):
        for scene in sorted({c['scene'] for c in cases}):
            options=[c for c in ordered if c['scene']==scene and c not in chosen]
            if options and scene_counts[scene]<=rank and len(chosen)<limit:
                chosen.append(options[0]);scene_counts[scene]+=1
    return chosen

def main():
    dest=O/'dataset_catalog.json'
    if dest.exists():print('catalog already frozen');return
    manifest=read(O/'data_review/synthetic_manifest.json');qa=read(O/'independent_data_reviews.json');tech=read(O/'technical_checks.json')
    assert tech['stage']=='complete' and qa['model']=='gpt-6-sol' and qa['reasoning_effort']=='xhigh' and qa['fast'] is False
    byid={r['case_id']:r for r in qa['cases']};checks={r['case_id']:r for r in tech['cases']}
    ids={c['case_id'] for c in manifest['clips']};assert set(byid)==ids and set(checks)==ids
    eligible=[c for c in manifest['clips'] if checks[c['case_id']]['technical_pass'] and byid[c['case_id']]['assistant_score']==2 and byid[c['case_id']]['decision']=='pass']
    train=select([c for c in eligible if c['candidate_role']=='train']);val=select([c for c in eligible if c['candidate_role']=='validation'],15)
    ts={c['scene'] for c in train};vs={c['scene'] for c in val};assert not ts&vs
    counts=Counter(c['type'] for c in train);scenes=Counter(c['scene'] for c in train)
    summary={'training_cases':len(train),'training_scene_count':len(ts),'max_training_cases_per_scene':max(scenes.values(),default=0),'training_type_counts':dict(counts),'training_scene_counts':dict(scenes),'validation_cases':len(val),'validation_scene_count':len(vs),'validation_type_counts':dict(Counter(c['type'] for c in val)),'validation_scenes':sorted(vs),'QA_total':len(ids),'QA_rejected_or_uncertain':len(ids)-len(eligible),'qualified_unused':len(eligible)-len(train)-len(val),'human_verdict':None}
    ready=len(train)>=45 and len(ts)>=20 and max(scenes.values(),default=0)<=3 and all(counts[k]>0 for k in ['background','single_actor','dense_actors']) and len(vs)>=3
    dump(O/'admission_result.json',dict(summary,ready=ready,rule='45–50 around50 training cases; >=20 worlds/max3 per scene/all three types; >=3 independent validation worlds; all actual-input technical gates and independent visual score2'))
    assert ready,summary
    chosen=train+val;catalog=[]
    for c in chosen:
        catalog.append({'dataset_id':'r8/'+c['case_id'],'case_id':c['case_id'],'folder':c['folder'],'frame_count':10,'type':c['type'],'receiver_scene':c['scene'],'split':c['candidate_role'],'assistant_score':2,'technical_pass':True,'independent_QA':byid[c['case_id']],'human_verdict':None,'source_id':c['source_id'],'occluder_asset':c['asset'],'occluder_template_scope':'two retained exposed DEV mesh shapes shared across splits; shape novelty not claimed'})
    dump(dest,{'cases':catalog,'summary':summary,'GT':'real RGB Y; synthetic RGB entirely erased before conditions','architecture_unchanged':True,'original_checkpoint_start':True,'same_recipe_as_r7':True,'data_changes':['receiver scene coverage','three-dimensional silhouette template instead of restricted affine source cutout','actual one-second observed-context indexing'],'excluded':[{'case_id':c['case_id'],'decision':byid[c['case_id']]['decision'],'score':byid[c['case_id']]['assistant_score'],'reason':byid[c['case_id']].get('reason'),'human_verdict':None} for c in manifest['clips'] if c not in eligible]});print(summary,flush=True)
if __name__=='__main__':main()
