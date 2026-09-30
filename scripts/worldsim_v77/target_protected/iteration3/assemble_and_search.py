"""复用已审receiver，只引入本轮连续多视角供体；旧产物不变。"""
import argparse,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from build_pairs import main as search

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())

def main(root,parent):
    out=root/'factory';out.mkdir(exist_ok=True);tem=root/'temporal_windows'
    reviewed={r['source_id']:r for r in read(tem/'independent_mask_reviews.json')['clips']}
    numeric={r['source_id']:r for r in read(tem/'mask_review/mask_audit.json')['clips']}
    fresh=[c for c in read(tem/'source_manifest.json')['clips'] if c['geometry_pass'] and numeric.get(c['source_id'],{}).get('numeric_gate_pass') and reviewed.get(c['source_id'],{}).get('donor_mask_status')=='pass']
    assert fresh,'无连续供体通过，不启动无效候选搜索'
    sources=read(parent/'source_manifest.json')['clips'];context=read(parent/'source_context.json')['clips']
    sq=read(parent/'subagent_source_reviews.json')['clips'];mq=read(parent/'mask_review/independent_mask_reviews.json')['clips'];nq=read(parent/'mask_review/mask_audit.json')['clips']
    sq=[r|{'donor_status':'not_selected_r3_new_sources_only'} for r in sq]
    mq=[r|{'donor_mask_status':'not_selected_r3_new_sources_only'} for r in mq]
    for c in sources:
        sid=c['source_id'];link(parent/'segmented'/sid,out/'segmented'/sid)
        for f in c['frames']:link(parent/'rgb'/f['filename'],out/'rgb'/f['filename'])
        for suffix in ['.json','_ground.npz']:
            p=parent/'geometry'/f'{sid}{suffix}'
            if p.exists():link(p,out/'geometry'/p.name)
    for c in fresh:
        sid=c['source_id'];sources.append(c);sq.append({'source_id':sid,'receiver_status':'not_receiver','donor_status':'pass','provenance':'r3 still plus continuous SAM2 source QA'})
        mq.append(reviewed[sid]|{'receiver_mask_status':'not_receiver'});nq.append(numeric[sid])
        link(tem/'segmented'/sid,out/'segmented'/sid)
        for f in c['frames']:link(tem/'rgb'/f['filename'],out/'rgb'/f['filename'])
    for fn,rows in [('source_manifest.json',sources),('source_context.json',context),('subagent_source_reviews.json',sq),('mask_review/independent_mask_reviews.json',mq),('mask_review/mask_audit.json',nq)]:dump(out/fn,{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r3','clips':rows})
    for sub in ['maps','segmented_secondary','mask_review_secondary']:link(parent/sub,out/sub)
    dump(out/'assembly.json',{'parent_receiver_factory':str(parent),'donor_temporal_root':str(tem),'new_donors':[c['source_id'] for c in fresh],'only_new_donors':True,'original_Y_preserved':True,'multiview_scope':'same-instance real source views; each target clip chooses one whole donor camera-track, no framewise view switching','camera_views_not_joint_neural_condition':True})
    search(out,same_log=True,prefix='R',mode='donor_camera_replay',allow_downsample=True,max_donors=None,ego_bottom_guard_px=64,run_id='r3')
    pairs=read(out/'pair_candidates.json');byid={c['source_id']:c for c in fresh}
    for case in pairs['selected']:
        case['cohort']='R3 / new real multiview donor, continuous 10 frames'
        case['donor_view_provenance']=byid[case['donor_source_id']]['seed_view']
    dump(out/'pair_candidates.json',pairs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();main(a.root,a.parent)
