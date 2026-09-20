"""仅在有限队列及CPU评价终态后汇总，不将帧当独立样本。"""
import csv
import json
from pathlib import Path
import av
import numpy as np
from select_target import P1

def main():
    queue=json.loads((P1/'queue.json').read_text())
    post=json.loads((P1/'postprocess.json').read_text())
    assert queue['status']==post['status']=='complete'
    calibration=json.loads((P1/'evaluator_calibration.json').read_text())
    rows=[]
    for stem in queue['completed']:
        run=json.loads((P1/'rollouts'/f'{stem}.json').read_text())
        assert run['status']=='complete' and run['frames']==237
        with av.open(str(P1/'rollouts'/f'{stem}.mp4')) as c:
            count=sum(1 for _ in c.decode(video=0))
        assert count==237
        gpu=list(csv.DictReader((P1/f'{stem}.gpu.csv').open()))
        peak=max(float(r[' memory.used [MiB]'].split()[0]) for r in gpu)
        row={'case':run['case'],'seed':run['seed'],'frames':237,'video_decode_passed':True,
             'wall_s':run['wall_s'],'peak_allocated_gib':run['peak_allocated_gib'],
             'sampled_gpu_peak_mib':peak,'oom':False}
        detpath=P1/'rollouts'/f'{stem}.detections.json'
        if detpath.exists():
            detections=json.loads(detpath.read_text())
            row['matched_frames']=detections['matched']
            row['evaluated_frames']=len(detections['frames'])
        if run['case']!='clean':
            response=json.loads((P1/'rollouts'/f'{stem}.response.json').read_text())
            row['image_response_windows']=response['windows']
            if detpath.exists():
                clean=json.loads((P1/'rollouts'/f'clean-seed{run["seed"]}.detections.json').read_text())
                pairs=[]
                accepted=calibration['status']=='passed' and clean['matched']>=6
                for orig,changed in zip(clean['frames'],detections['frames']):
                    f=orig['frame']
                    assert changed['frame']==f
                    delta=None
                    if accepted and orig['match'] and changed['match']:
                        delta=(np.asarray(changed['match']['center'])-orig['match']['center']).tolist()
                    pairs.append({'frame':f,'delta_center_xy_px':delta,
                                  'center_distance_px':float(np.linalg.norm(delta)) if delta is not None else None})
                windows={}
                for name,lo,hi in [('biased_window',5,36),('recovery_window',37,53)]:
                    group=[r for r in pairs if lo<=r['frame']<=hi]
                    valid=[r for r in group if r['center_distance_px'] is not None]
                    windows[name]={'paired_matches':len(valid),'scheduled_frames':len(group),
                                   'mean_center_distance_px':float(np.mean([r['center_distance_px'] for r in valid])) if valid else None,
                                   'mean_delta_x_px':float(np.mean([r['delta_center_xy_px'][0] for r in valid])) if valid else None}
                row['detection_measurement_accepted']=accepted
                row['detection_windows']=windows
                row['detection_pairs']=pairs
        rows.append(row)
    controls=[]
    for seed in [42,43]:
        for sign in ['negative','positive']:
            a=next(r for r in rows if r['case']==sign+'_persistent' and r['seed']==seed)
            b=next(r for r in rows if r['case']==sign+'_restore' and r['seed']==seed)
            key='recovery_window'
            controls.append({'seed':seed,'sign':sign,
                 'persistent_roi_mae':a['image_response_windows'][key]['mean_roi_mae_8bit'],
                 'restored_roi_mae':b['image_response_windows'][key]['mean_roi_mae_8bit'],
                 'persistent_center_distance_px':a.get('detection_windows',{}).get(key,{}).get('mean_center_distance_px'),
                 'restored_center_distance_px':b.get('detection_windows',{}).get(key,{}).get('mean_center_distance_px')})
    result={'task_id':'WS-V75-LOCALIZE-01','run_id':'20260920-r1','status':'complete',
            'scene_count':1,'target_count':1,'seeds':[42,43],'rollouts':10,'generation_blocks':300,
            'scope':'synthetic condition sensitivity; paired two-dimensional observations, no future RGB GT or policy feedback',
            'results':rows,'restoration_comparisons':controls,
            'evaluator_calibration_status':calibration['status'],
            'human_verdict':None,'failure_ledger_delta':'none',
            'failure_ledger_refs':['V75-F01','V74-H2-F20','V74-H2-F21','V74-H2-F22']}
    (P1/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','restore':controls,'peak_mib':max(r['sampled_gpu_peak_mib'] for r in rows)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
