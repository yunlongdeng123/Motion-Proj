"""固定分成三个不重叠10帧窗口；匹配官方训练长度，不伪称3秒验证。"""
import argparse,copy
from pathlib import Path
from geometry_factory import read,dump

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())

def main(parent,out):
    out.mkdir(parents=True,exist_ok=True)
    meta=read(parent/'source_manifest.json');contexts={c['source_id']:c for c in read(parent/'source_context.json')['clips']}
    sq={r['source_id']:r for r in read(parent/'subagent_source_reviews.json')['clips']}
    mq={r['source_id']:r for r in read(parent/'mask_review/independent_mask_reviews.json')['clips']}
    numeric={r['source_id']:r for r in read(parent/'mask_review/mask_audit.json')['clips']}
    secondary=read(parent/'mask_review_secondary/independent_secondary_mask_reviews.json')['clips']
    sources=[];ctx=[];srows=[];mrows=[];nrows=[];secondary_rows=[]
    for c in meta['clips']:
        sid=c['source_id']
        if sid not in mq or not numeric[sid]['numeric_gate_pass']:continue
        assert len(c['frames'])==30
        for start in [0,10,20]:
            wid=f'{sid}w{start//10}';window=copy.deepcopy(c);window['source_id']=wid
            window['frames']=window['frames'][start:start+10]
            for j,f in enumerate(window['frames']):f['parent_frame']=f['frame'];f['frame']=j
            window['window_provenance']={'parent_root':str(parent),'parent_source_id':sid,'parent_frame_start':start,'parent_frame_stop_exclusive':start+10,'selection':'fixed nonoverlapping windows 0:10,10:20,20:30; not picked from rendered outputs','official_train_num_frames':10,'long_temporal_validation':False}
            sources.append(window);ctx.append(contexts[sid]|{'source_id':wid})
            for source,target in [(sq,srows),(mq,mrows),(numeric,nrows)]:
                target.append(source[sid]|{'source_id':wid,'inherited_from_full_30_frame_source':sid})
            for j in range(10):link(parent/'segmented'/sid/'sam2_raw'/f'{start+j:05}.png',out/'segmented'/wid/'sam2_raw'/f'{j:05}.png')
            if (parent/'geometry'/f'{sid}.json').exists():
                dump(out/'geometry'/f'{wid}.json',read(parent/'geometry'/f'{sid}.json')|{'source_id':wid,'ground_parent_source':sid})
                if (parent/'geometry'/f'{sid}_ground.npz').exists():link(parent/'geometry'/f'{sid}_ground.npz',out/'geometry'/f'{wid}_ground.npz')
            for r in secondary:
                if r['source_id']!=sid or r['protected_mask_status']!='pass':continue
                job=r['job_id'];wjob=f'{wid}__{job.split("__",1)[1]}'
                secondary_rows.append(r|{'source_id':wid,'job_id':wjob,'inherited_from_full_30_frame_job':job})
                for j in range(10):link(parent/'segmented_secondary'/job/'sam2_raw'/f'{start+j:05}.png',out/'segmented_secondary'/wjob/'sam2_raw'/f'{j:05}.png')
    for fn,rows in [('source_manifest.json',sources),('source_context.json',ctx),('subagent_source_reviews.json',srows),('mask_review/mask_audit.json',nrows),('mask_review/independent_mask_reviews.json',mrows),('mask_review_secondary/independent_secondary_mask_reviews.json',secondary_rows)]:
        dump(out/fn,{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','window_admission':'full parent source/mask independent review inherited; every synthetic case still requires fresh independent review','clips':rows})
    for sub in ['rgb','maps']:link(parent/sub,out/sub)
    dump(out/'assembly.json',{'parent':str(parent),'source_windows':len(sources),'parent_sources':len({c['window_provenance']['parent_source_id'] for c in sources}),'ground_truth_generated':False,'duration_policy':'10 actual exposures (~1s preview); not a 3s or long-term evaluation','official_reference':'DriveEditor/configs/train.yaml data.params.num_frames=10; sgm/data/nus.py get_blank'})
    print('WINDOWS',len(sources),'PARENTS',len(mq),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.parent,a.out)
