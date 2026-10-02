"""只诊断既有87条轨迹的保护对象对齐，不改H、不增轨迹、不准入训练。"""
from pathlib import Path
import json, os, time
from collections import Counter
import numpy as np
from PIL import Image

T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r31'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load(folder):return [np.asarray(Image.open(folder/f'{i:05}.png'))>0 for i in range(30)]

def main():
    O.mkdir(exist_ok=False)
    dump(O/'run.json',{'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r31',
        'hypothesis':'midpoint GT-envelope ranking may select holes that miss the curated well-observed B',
        'source':'r28 fixed 87 trajectories / 30 sources','new_positions':False,'new_speeds':False,
        'mask_rule_changed':False,'Y_SAM_role':'offline diagnosis only','GPU_jobs':0,'training_steps':0,
        'stop_rule':'one complete audit; only change proposal ranking if measured mismatch supports it',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None})
    sources={c['source_id']:c for c in read(T/'r23/factory/source_manifest.json')['clips']}
    quality={c['case_id']:c for c in read(T/'r28/instance_quality.json')['cases']}
    rows=[];cache={};started=time.time()
    for c in read(T/'r28/prepared.json')['cases']:
        sid=c['source_id'];source=sources[sid];folder=Path(c['observed_folder'])
        h=load(folder/'proposal_H');a=load(folder/'influence');tracks=[]
        curated={v['instance_token'] for v in source['actors']}
        actors={v['instance_token']:v for v in c['retained_instances']}
        # 源保护车即使完全未触及H也统计，不把候选清单的包络选择当分母。
        for j,orig in enumerate(source['actors']):
            tok=orig['instance_token']
            if tok not in actors:
                actors[tok]={'instance_token':tok,'category':orig['category'],
                             'Y_quality_job':sid+'_'+tok[:8],'quality_input_available':True}
        for tok,info in actors.items():
            path=T/'r28/quality_labels/observed_masks'/info['Y_quality_job']
            if not (path/'00000.png').exists():
                k=next((j for j,v in enumerate(source['actors']) if v['instance_token']==tok),None)
                if k is None:continue
                seg=T/'r23/factory'/('segmented' if k==0 else 'segmented_secondary')/(sid if k==0 else sid+'_'+tok[:8])
                if not (seg/'mask_manifest.json').exists() or not read(seg/'mask_manifest.json')['technical_temporal_mask_pass']:continue
                path=seg/'sam2_raw'
            key=str(path)
            if key not in cache:cache[key]=load(path)
            masks=cache[key];areas=np.array([m.sum() for m in masks],dtype=float)
            hi=np.array([(hh&m).sum() for hh,m in zip(h,masks)],dtype=float)
            ai=np.array([(aa&m).sum() for aa,m in zip(a,masks)],dtype=float)
            fraction=hi/np.maximum(1,areas)
            widths=[];heights=[]
            for m in masks:
                yy,xx=np.where(m);widths.append(int(np.ptp(xx)+1) if len(xx) else 0);heights.append(int(np.ptp(yy)+1) if len(yy) else 0)
            tracks.append({'instance':tok,'category':info['category'],'curated_source_actor':tok in curated,
                'median_visible_area_px':float(np.median(areas)),'median_width_px':float(np.median(widths)),
                'median_height_px':float(np.median(heights)),'nonempty_frames':int((areas>0).sum()),
                'H_occlusion_fraction':fraction.tolist(),'H_intersection_px':hi.astype(int).tolist(),
                'silhouette_intersection_px':ai.astype(int).tolist(),'max_H_occlusion':float(fraction.max()),
                'frames_over_5pct':int((fraction>.05).sum()),'range_H_occlusion':float(np.ptp(fraction)),
                'occlusion_amount_gate':bool(fraction.max()>=.30 and sum(fraction>.05)>=3)})
        base=[v for v in tracks if v['curated_source_actor']]
        q=quality[c['case_id']]
        rows.append({'case_id':c['case_id'],'source_id':sid,'scene':c['scene'],'split':c['split'],
            'reason':q['quality']['data_family'] if q['quality'] else q['reason'],
            'curated_track_count':len(base),'curated_any_H_overlap':any(max(v['H_intersection_px'])>0 for v in base),
            'curated_occlusion_amount_gate':any(v['occlusion_amount_gate'] for v in base),
            'curated_amount_and_fraction_change':any(v['occlusion_amount_gate'] and v['range_H_occlusion']>=.15 for v in base),
            'identity_flags':q['identity_flags'],'scope_flags':q['scope_flags'],'tracks':tracks})
        print('ALIGNMENT',c['case_id'],rows[-1]['curated_occlusion_amount_gate'],flush=True)
    by_reason={}
    for reason in sorted({r['reason'] for r in rows}):
        rr=[r for r in rows if r['reason']==reason]
        by_reason[reason]={'count':len(rr),'curated_any_H_overlap':sum(r['curated_any_H_overlap'] for r in rr),
                          'curated_occlusion_amount_gate':sum(r['curated_occlusion_amount_gate'] for r in rr)}
    result={'cases':rows,'summary':{'count':len(rows),'sources':len({r['source_id'] for r in rows}),
        'curated_any_H_overlap':sum(r['curated_any_H_overlap'] for r in rows),
        'curated_occlusion_amount_gate':sum(r['curated_occlusion_amount_gate'] for r in rows),
        'curated_amount_and_fraction_change':sum(r['curated_amount_and_fraction_change'] for r in rows),
        'by_reason':by_reason},'seconds':time.time()-started,'training_admission':0,
        'boundary':'SAM is reference, not manual GT; amount gate alone does not certify reveal, ordering or data quality'}
    dump(O/'alignment.json',result);dump(O/'controller_state.json',{'stage':'complete_pending_interpretation','pid':os.getpid()})
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
