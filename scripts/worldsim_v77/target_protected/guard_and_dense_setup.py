"""新候选避开底部64px保守自车前景带；原对照不覆盖。"""
import argparse
from pathlib import Path
from collections import Counter
from geometry_factory import read,dump

def main(root):
    path=root/'pair_candidates.json';p=read(path);original=root/'pair_candidates_before_ego_guard.json'
    if original.exists():raise RuntimeError('guard already applied; do not overwrite original')
    dump(original,p);dump(root/'synthesis_config_before_ego_guard.json',read(root/'synthesis_config.json'))
    keep=[];reject=[]
    for r in p['selected']:
        bottom=max(f['box'][3] for f in r['frames'])+6
        if bottom>=512:reject.append({'case_id':r['case_id'],'reason':'conservative_bottom_64px_ego_foreground_band','max_box_bottom_plus_hole_upper':bottom})
        else:keep.append(r|{'ego_image_guard':'projected bottom + 6px hole support <512; conservative exclusion, not exact ego segmentation'})
    config=p['config']|{'image_bottom_guard_px':64,'reason':'D006/D009 actual ego-hood overwrite found by independent QA; preserve original candidates and reject risk before new synthesis'}
    dump(path,p|{'config':config,'selected':keep,'selected_counts':dict(Counter(r['type'] for r in keep)),'pre_render_ego_guard_rejected':reject})
    dump(root/'synthesis_config.json',config)
    dense=root.parent/'dense_window_factory';dense.mkdir(exist_ok=True)
    for name in ['source_manifest.json','source_context.json','subagent_source_reviews.json','mask_review','mask_review_secondary','segmented','segmented_secondary','geometry','rgb','maps']:
        dst=dense/name
        if not dst.exists():dst.symlink_to(root/name,target_is_directory=(root/name).is_dir())
    print('GUARD',len(keep),'kept',len(reject),'excluded; dense root',dense,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
