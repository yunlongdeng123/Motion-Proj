"""新多视角供体无空间可行解后，复用已审干净来源的几何配对控制。

不扩offset、不放宽门槛、不增加模型；来源先与receiver配对，避免独立挑好看来源。
"""
import argparse,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from build_pairs import main as search

def main(root,parent):
    factory=root/'factory';old=read(factory/'pair_candidates.json');assert not old['selected']
    backup=factory/'world_offset_multiview_control';backup.mkdir(exist_ok=True)
    for n in ['pair_candidates.json','synthesis_config.json','pair_search_progress.json','subagent_source_reviews.json','mask_review/independent_mask_reviews.json']:
        target=backup/Path(n).name;assert not target.exists();shutil.copy2(factory/n,target)
    for rel in ['subagent_source_reviews.json','mask_review/independent_mask_reviews.json']:
        legacy={c['source_id']:c for c in read(parent/rel)['clips']};data=read(factory/rel)
        data['clips']=[legacy.get(c['source_id'],c) for c in data['clips']];data['source_pool_revision']='r3 geometry-coupled approved source pool; known contamination still blocked in both entries';dump(factory/rel,data)
    assembly=read(factory/'assembly.json');dump(backup/'assembly.json',assembly)
    assembly.update(only_new_donors=False,source_pool_revision='historically approved clean sources plus three new temporal sources; all recheck donor gate before geometry')
    dump(factory/'assembly.json',assembly)
    search(factory,same_log=True,prefix='R',mode='donor_camera_replay',allow_downsample=True,max_donors=6,ego_bottom_guard_px=64,run_id='r3')
    pairs=read(factory/'pair_candidates.json');sources={c['source_id']:c for c in read(factory/'source_manifest.json')['clips']}
    prior=read(parent/'synthetic_review/synthetic_manifest.json')['clips']
    def identity(c):return (c['source_id'],c['donor_source_id'],c['placement_mode'],c['offset_longitudinal_m'],c['offset_lateral_m'])
    old_ids={identity(c) for c in prior};duplicates=[];fresh=[]
    for case in pairs['selected']:
        if identity(case) in old_ids:duplicates.append(case);continue
        case['case_id']=f'R{len(fresh)+1:03}'
        case['cohort']='R3 / geometry-coupled clean donor, 10 frames'
        case['donor_view_provenance']=sources[case['donor_source_id']].get('seed_view')
        case['uses_new_multiview_donor']=case['donor_source_id'].startswith('V')
        fresh.append(case)
    pairs['selected_before_duplicate_exclusion']=len(pairs['selected']);pairs['duplicates_preserved_not_rerendered']=duplicates;pairs['selected']=fresh
    from collections import Counter
    pairs['selected_counts']=dict(Counter(c['type'] for c in fresh));pairs['source_pool_revision']='jointly pair approved sources before new segmentation; no independent-view quality claim'
    dump(factory/'pair_candidates.json',pairs)
    dump(root/'coupled_source_control.json',{'new_multiview_only_world_selected':0,'new_multiview_only_replay_selected':0,'approved_pool_selected':len(fresh),'duplicates_not_rerendered':len(duplicates),'unchanged_geometry_thresholds':True,'donor_blacklist_and_bottom_guard_retained':True,'max_ranked_donors':6,'new_multiview_donor_cases':sum(c['uses_new_multiview_donor'] for c in fresh)})
    print('FRESH_COUPLED',len(fresh),'DUPLICATE',len(duplicates),pairs['selected_counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();main(a.root,a.parent)
