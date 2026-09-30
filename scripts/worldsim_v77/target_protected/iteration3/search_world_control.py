"""相机相对回放无可行提案后，仅做已有固定世界位移的有界强控制。"""
import argparse,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from build_pairs import main as search

def main(root):
    factory=root/'factory';old=read(factory/'pair_candidates.json')
    assert not old['selected'] and old['config']['placement_mode']=='donor_camera_replay'
    backup=factory/'camera_replay_control';backup.mkdir(exist_ok=True)
    for n in ['pair_candidates.json','synthesis_config.json','pair_search_progress.json']:
        assert not (backup/n).exists();shutil.copy2(factory/n,backup/n)
    search(factory,same_log=True,prefix='R',mode='world_offset',allow_downsample=True,max_donors=None,ego_bottom_guard_px=64,run_id='r3')
    pairs=read(factory/'pair_candidates.json');sources={c['source_id']:c for c in read(factory/'source_manifest.json')['clips']}
    for case in pairs['selected']:
        case['cohort']='R3 / real multiview donor, fixed world offset, 10 frames'
        case['donor_view_provenance']=sources[case['donor_source_id']]['seed_view']
    dump(factory/'pair_candidates.json',pairs)
    dump(root/'placement_control.json',{'first':'camera_replay_control','first_selected':0,'first_rejections':old['rejection_counts'],'second':'fixed_world_offset','second_selected':len(pairs['selected']),'second_rejections':pairs['rejection_counts'],'same_inputs_and_gates':True,'relaxed_gates':False,'more_offset_or_model_sweeps':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
