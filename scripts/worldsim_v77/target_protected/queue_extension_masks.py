"""只队列已逐例复核、实际完整解码的补充来源；解包可仍在另一个分片继续。"""
import argparse
from pathlib import Path
from PIL import Image
from geometry_factory import read,dump

def main(root):
    sources={c['source_id']:c for c in read(root/'source_manifest.json')['clips']};gates={c['source_id']:c for c in read(root/'source_geometry_validation.json')['clips']}
    context={c['source_id']:c for c in read(root/'source_context.json')['clips']}
    qa=read(root/'subagent_source_reviews.json');jobs=[]
    for q in qa['clips']:
        sid=q['source_id'];c=sources[sid];roles=[role for role in ['receiver','donor'] if q[role+'_status']=='pass']
        if not roles:continue
        assert gates[sid]['primary_geometry_pass'];assert {0,15,29}<=set(q['reviewed_frames'])
        for f in c['frames']:
            name=f['filename'];p=next((root/'by_shard'/s/name for s in ['03','07'] if (root/'by_shard'/s/name).is_file()),None);assert p is not None,name
            with Image.open(p) as im:im.load();assert im.size==(1600,900)
            link=root/'rgb'/name;link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(p)
        for cf in context[sid]['frames']:
            for row in cf['sensors'].values():
                if not row['materialize']:continue
                name=row['filename'];p=next((root/'by_shard'/s/name for s in ['03','07'] if (root/'by_shard'/s/name).is_file()),None)
                if p is not None:
                    link=root/'rgb'/name;link.parent.mkdir(parents=True,exist_ok=True)
                    if not link.exists():link.symlink_to(p)
        jobs.append({'source_id':sid,'scene':c['scene'],'instance_token':c['actors'][0]['instance_token'],'role':'primary_reviewed_actor','eligible_source_roles':roles,
                     'prompt_frame':15,'frame_filenames':[f['filename'] for f in c['frames']],'purpose':'reviewed additional source masks only'})
    assert jobs
    if not (root/'maps').exists():(root/'maps').symlink_to(root.parent/'maps',target_is_directory=True)
    dump(root/'segmentation_queue.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','stage':'awaiting_gpu','seed':42,'jobs':jobs,'reviewed_source_subset':True})
    print('ADDITIONAL_MASK_QUEUE',len(jobs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
