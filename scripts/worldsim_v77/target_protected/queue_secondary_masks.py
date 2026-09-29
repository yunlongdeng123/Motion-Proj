"""仅将经过独立来源复核的protected次要车辆送SAM2。"""
import argparse
from pathlib import Path
from geometry_factory import read,dump
def main(root):
    sources={c['source_id']:c for c in read(root/'source_manifest.json')['clips']};qa=read(root/'secondary_source_reviews.json')['clips'];jobs=[]
    for r in qa:
        if r['protected_status']!='pass':continue
        c=sources[r['source_id']];ai=next(i for i,a in enumerate(c['actors']) if a['instance_token']==r['instance_token'])
        jobs.append({'job_id':f'{r["source_id"]}__a{ai}','source_id':r['source_id'],'scene':c['scene'],'instance_token':r['instance_token'],'role':'secondary_reviewed_protected_actor',
                     'eligible_source_roles':['protected'],'prompt_frame':15,'frame_filenames':[f['filename'] for f in c['frames']]})
    dump(root/'secondary_segmentation_queue.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','stage':'awaiting_gpu','jobs':jobs,'seed':42})
    print('SECONDARY_JOBS',len(jobs),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
