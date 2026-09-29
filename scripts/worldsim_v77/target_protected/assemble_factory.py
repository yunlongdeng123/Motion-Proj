"""合并已质检来源的只读链接；保留两轮原始清单与审核，不覆盖。"""
import argparse
from pathlib import Path
from geometry_factory import read,dump

def main(root):
    extra=root/'source_extension';out=root/'expanded_factory';out.mkdir(exist_ok=True)
    for fn,key in [('source_manifest.json','clips'),('source_context.json','clips'),('source_geometry_validation.json','clips'),('subagent_source_reviews.json','clips'),('mask_review/mask_audit.json','clips'),('mask_review/independent_mask_reviews.json','clips')]:
        a=read(root/fn);b=read(extra/fn);aa=a[key];bb=b[key]
        assert not {r['source_id'] for r in aa}&{r['source_id'] for r in bb}
        merged=a|{key:aa+bb,'assembly_sources':[str(root/fn),str(extra/fn)]}
        if fn=='source_manifest.json':
            merged['sampling_rejected']=a['sampling_rejected']+b['sampling_rejected']
            merged['unique_rgb_file_count']=len({f['filename'] for c in aa+bb for f in c['frames']})
            merged['selection']=[str(root/'source_selection.json'),str(extra/'source_selection.json')]
        if fn=='subagent_source_reviews.json':merged['review_scope']='Assembly of independent source reviews. Original reviewer/stage metadata retained in assembly_sources.'
        dump(out/fn,merged)
    for fn in ['secondary_source_reviews.json','secondary_segmentation_queue.json','mask_review_secondary/independent_secondary_mask_reviews.json']:
        a=read(root/fn);b=read(extra/fn) if (extra/fn).exists() else {'clips':[],'jobs':[]};key='jobs' if 'jobs' in a else 'clips';dump(out/fn,a|{key:a[key]+b.get(key,[])})
    for parent in [root,extra]:
        for sub in ['segmented','segmented_secondary','geometry']:
            if not (parent/sub).exists():continue
            dest=out/sub;dest.mkdir(exist_ok=True)
            for path in (parent/sub).iterdir():
                if path.name=='ground_summary.json':continue
                link=dest/path.name
                if not link.exists():link.symlink_to(path,target_is_directory=path.is_dir())
        for row in read(parent/'source_file_inventory.json'):
            path=parent/'rgb'/row['filename'];link=out/'rgb'/row['filename'];link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(path)
    if not (out/'maps').exists():(out/'maps').symlink_to(root/'maps',target_is_directory=True)
    dump(out/'assembly.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','parents':[str(root),str(extra)],'overwrites_original_manifests':False})
    print('ASSEMBLED',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
