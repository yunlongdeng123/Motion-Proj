"""在模型输出前固定均衡数据、独立来源验证和有限QA备用例。"""
from pathlib import Path
import sys,copy
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,read,dump
def main():
    if (O/'asset_selected.json').exists():print('already selected');return
    parts=[read(O/(f'all_asset_coarse_candidates_{i}.json' if (O/f'all_asset_coarse_candidates_{i}.json').exists() else f'all_asset_candidates_{i}.json')) for i in range(4)];assert all(p['stage']=='complete' for p in parts)
    focus=[read(O/f'all_asset_focused_candidates_{i}.json') for i in range(4)]
    assert all(p['stage']=='complete' for p in focus)
    intended={c['source_id'] for c in read(F/'source_manifest.json')['clips'] if c['source_id'].startswith('N')}
    assert {r['source_id'] for p in parts for r in p['rows']}==intended
    assert {r['source_id'] for p in focus for r in p['rows']}==intended
    pool=[p for part in parts+focus for p in part['candidates']]
    unique={}
    for p in pool:unique[(p['source_id'],p['asset'],p['offset_longitudinal_m'],p['offset_lateral_m'])]=p
    pool=list(unique.values())
    dump(O/'all_asset_candidates.json',{'candidates':pool,'scene_count':len({p['scene'] for p in pool}),'type_counts':dict(Counter(p['type'] for p in pool))})
    original=read(O/'source_split.json');oldtrain=set(original['old_r7_training_scenes']);available={p['scene'] for p in pool}
    first=[s for s in original['synthetic_validation_scenes'] if s in available]
    reserve=sorted(available-oldtrain-set(first),key=lambda s:(-len({p['type'] for p in pool if p['scene']==s}),s))
    val=(first+reserve)[:5]
    split={'synthetic_validation_scenes':val,'original_geometric_only_validation_scenes':original['synthetic_validation_scenes'],
           'rule':'retain original five if actual data available; fill only data-feasible receiver scenes absent from every r7 train scene, type coverage then lexicographic. Frozen before training or new model outputs.',
           'train_scene_max_cases':3,'real_video_receiver_disjoint':True,'shared_occluder_shapes':True,'shape_generalization_not_claimed':True,'no_model_outputs_used':True}
    dump(O/'evaluation_source_split.json',split)
    chosen=[];scene_counts=Counter();window_counts=Counter();type_counts=Counter()
    # 多取有限QA备用例，但最终训练仍最多50，scene最多3，不重复拿待定当通过。
    for role,limits in [('validation',{'dense_actors':5,'single_actor':8,'background':5}),('train',{'dense_actors':18,'single_actor':26,'background':30})]:
        for kind,limit in limits.items():
            pp=[p for p in pool if (p['scene'] in val)==(role=='validation') and p['type']==kind]
            for rank in range(3):
                for scene in sorted({p['scene'] for p in pp}):
                    choices=[p for p in pp if p['scene']==scene and p not in chosen and window_counts[p['source_id']]<2]
                    if not choices or scene_counts[scene]>=3 or type_counts[role,kind]>=limit:continue
                    p=copy.deepcopy(choices[0]);p['candidate_role']=role;chosen.append(p);scene_counts[scene]+=1;window_counts[p['source_id']]+=1;type_counts[role,kind]+=1
    for i,p in enumerate(chosen):
        p.update(case_id=f'M{i+1:03}',edge_mode=['near_hard','feather_05','feather_10'][i%3],mask_dilation_px=[2,3,4][i%3],human_verdict=None,training_ready=False,quality_status='pending_render_all_frame_gates_and_independent_QA')
    summary={'selected_for_QA':len(chosen),'train_candidate_count':sum(p['candidate_role']=='train' for p in chosen),'training_candidate_scenes':len({p['scene'] for p in chosen if p['candidate_role']=='train'}),'validation_candidate_scenes':len({p['scene'] for p in chosen if p['candidate_role']=='validation'}),'role_type_counts':{f'{r}:{k}':n for (r,k),n in type_counts.items()},'max_cases_per_scene':max(scene_counts.values()) if scene_counts else 0}
    dump(O/'asset_selected.json',{'selected':chosen,'summary':summary,'full_pool_types':dict(Counter(p['type'] for p in pool)),'full_pool_scenes':len(available),'qualified_count':0});print(summary,flush=True)
if __name__=='__main__':main()
