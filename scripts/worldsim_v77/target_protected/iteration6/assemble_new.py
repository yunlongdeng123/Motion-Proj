"""真实保护mask通过原精确关卡后复用旧渲染器；不重新选位姿。"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from iteration5.admit_exact import main as admit
from iteration5.planning import inputs
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.resolve()==src.resolve()
    else:dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
def main(parent,root):
    admit(parent,root)
    for n in ['exact_admission.json','exact_pair_candidates.json']:
        data=read(root/n);data['run_id']='r6';dump(root/n,data)
    selected=read(root/'exact_pair_candidates.json')['selected']
    if not selected:print('NO_NEW_EXACT_PAIRS');return
    sources,donors,donor_root,_,_=inputs(parent);out=root/'factory';out.mkdir(exist_ok=True)
    jobs={j['job_id']:j for j in read(parent/'gpu_queue.json')['jobs']};manifest=[];secondary=[]
    context={c['source_id']:c for c in read(parent/'native10_factory/source_context.json')['clips']};ctx=[]
    used=set()
    for p in selected:
        sid=p['source_id'];c=sources[sid];ctx.append(context[sid])
        primary=c['actors'][0]['instance_token'];jid=sid+'_'+primary[:8]
        link(parent/'segmented_receivers'/jid,out/'segmented'/sid)
        for tok in p['occlusion_fraction']:
            if tok==primary:continue
            jid=sid+'_'+tok[:8];link(parent/'segmented_receivers'/jid,out/'segmented_secondary'/jid)
            secondary.append({'source_id':sid,'instance_token':tok,'job_id':jid,'protected_mask_status':'pass'})
        for suffix in ['.json','_ground.npz']:link(parent/'native10_factory/geometry'/f'{sid}{suffix}',out/'geometry'/f'{sid}{suffix}')
        for ob,base in [(c,parent/'native10_factory'),(donors[p['donor_source_id']],donor_root)]:
            if ob['source_id'] in used:continue
            manifest.append(ob);used.add(ob['source_id'])
            for f in ob['frames']:link(base/'rgb'/f['filename'],out/'rgb'/f['filename'])
            if ob['source_id'] in donors:link(donor_root/'segmented'/ob['source_id'],out/'segmented'/ob['source_id'])
    link(parent/'native10_factory/maps',out/'maps')
    dump(out/'source_manifest.json',{'clips':manifest});dump(out/'source_context.json',{'clips':ctx})
    dump(out/'mask_review_secondary/independent_secondary_mask_reviews.json',{'clips':secondary})
    dump(out/'pair_candidates.json',{'config':{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r6','image_bottom_guard_px':64},'selected':selected})
    print('ASSEMBLED_NEW',len(selected))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.parent,a.root)
