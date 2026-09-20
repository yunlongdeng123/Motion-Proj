"""按事前规则汇总唯一额外seed，保留基线失败与所有反向指标。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import av

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
RUN=ROOT/'WS-V75-SHAPE-CONFIRM-01/20260920-r1'
SOURCE=ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2'
ARMS=['dvgt_class_prior','dvgt_visible_extent']


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def absolute_gate(r):
    return (r['median_abs_action_error_at_own_ego_mps2']<=.5 and r['max_underbraking_mps2']<=2
            and r['max_overbraking_mps2']<=2 and r['max_route_lateral_m']<=1 and not r['overlap_frames'])


def main():
    queue=json.loads((RUN/'queue_result.json').read_text());assert queue['status']!='running'
    protocol=json.loads((RUN/'protocol.json').read_text())
    old=json.loads((SOURCE/'review/comparison.json').read_text())
    out=RUN/'review';assert not out.exists();out.mkdir()
    cases=[]
    for old_case in old['cases']:
        folder=RUN/old_case['log_id'];rows=[];series=[]
        for arm in ['gt_clean']+ARMS:
            source_row=next(r for r in old_case['rows'] if r['arm']==arm)
            rows.append({**source_row,'seed':42})
            f=folder/arm/'assessment.json'
            if f.exists():rows.append(json.loads(f.read_text()))
        pair=[]
        for seed in [42,43]:
            by_arm={r['arm']:r for r in rows if r['seed']==seed}
            if not all(a in by_arm for a in ARMS):continue
            c,e=(by_arm[a] for a in ARMS)
            rule={'progress_difference_gt_1m':e['progress_m']-c['progress_m']>1,
                  'mean_error_extent_gt_class':e['mean_abs_action_error_at_own_ego_mps2']>c['mean_abs_action_error_at_own_ego_mps2'],
                  'underbraking_extent_gt_class':e['max_underbraking_mps2']>c['max_underbraking_mps2']}
            pair.append({'seed':seed,'progress_extent_minus_class_m':e['progress_m']-c['progress_m'],
                         'mean_error_extent_minus_class_mps2':e['mean_abs_action_error_at_own_ego_mps2']-c['mean_abs_action_error_at_own_ego_mps2'],
                         'underbraking_extent_minus_class_mps2':e['max_underbraking_mps2']-c['max_underbraking_mps2'],
                         'overbraking_extent_minus_class_mps2':e['max_overbraking_mps2']-c['max_overbraking_mps2'],
                         'rule_terms':rule,'primary_conjunction':all(rule.values()),
                         'class_absolute_gate':absolute_gate(c),'extent_absolute_gate':absolute_gate(e)})
        for seed in [42,43]:
            for arm in ARMS:
                directory=(SOURCE if seed==42 else RUN)/old_case['log_id']/arm
                if not (directory/'assessment.json').exists() and seed==43:continue
                dec=json.loads((directory/'decisions.json').read_text())
                dense=json.loads((directory/'dense_replay.json').read_text())
                series.append({'seed':seed,'arm':arm,'time_s':[x['observation_frame']/30 for x in dec],
                               'action_error_mps2':[x['acceleration_difference_mps2'] for x in dec],
                               'actual_action_mps2':[x['policy']['acceleration_mps2'] for x in dec],
                               'dense_time_s':[x['time_s'] for x in dense['frames']],
                               'progress_m':[x['route_progress_lateral'][0] for x in dense['frames']],
                               'reference_clearance_m':[x['target_reference_clearance_m'] for x in dense['frames']]})
        new_pair=next((p for p in pair if p['seed']==43),None)
        case={'log_id':old_case['log_id'],'rows':rows,'pairs':pair,'series':series,
              'is_primary':old_case['log_id']==protocol['primary_log'],'human_verdict':None}
        if new_pair:
            # 视频固定完整117帧，列是先验，行是seed，不挑另一段片段救结果。
            streams=[(seed,arm,np.load((SOURCE if seed==42 else RUN)/old_case['log_id']/arm/'generated.npy',mmap_mode='r')) for seed in [42,43] for arm in ARMS]
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
            video=out/f'{old_case["log_id"][:8]}-two-seed.mp4'
            with av.open(str(video),'w') as writer:
                stream=writer.add_stream('libx264',rate=30);stream.width=1280;stream.height=780;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
                for f in range(117):
                    canvas=Image.new('RGB',(1280,780),'#152536');draw=ImageDraw.Draw(canvas)
                    for i,(seed,arm,frames) in enumerate(streams):
                        x=(i%2)*640;y=(i//2)*390
                        draw.text((x+8,y+6),f'Seed {seed} | '+['Fixed class','Visible extent'][i%2]+f' | t={f/30:.2f}s',font=font,fill='white')
                        canvas.paste(Image.fromarray(frames[f]).resize((640,352)),(x,y+36))
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):writer.mux(packet)
                for packet in stream.encode():writer.mux(packet)
            with av.open(str(video)) as reader:assert sum(1 for _ in reader.decode(video=0))==117
            case['video_frames']=117
        cases.append(case)
    primary=next(c for c in cases if c['is_primary']);p43=next((p for p in primary['pairs'] if p['seed']==43),None)
    decision='not_evaluable_baseline_or_engineering_stop' if p43 is None else ('primary_reproduced' if p43['primary_conjunction'] else 'primary_not_reproduced_close_candidate')
    result={'status':'complete','queue_status':queue['status'],'cases':cases,'decision':decision,
            'new_generated_frames':sum(x['result'].get('frames',0) for x in queue['runs']),
            'new_replay_frames':sum(x.get('assessment',{}).get('camera_replay_frames',0) for x in queue['runs']),
            'wall_s':queue['wall_s'],'peak_allocated_gib':max(x['result'].get('peak_allocated_gib',0) for x in queue['runs']),
            'interpretation':'one fixed additional seed on exposed tasks; no new-source independence, native shape-head or SOTA universality claim',
            'human_verdict':None,'failure_ledger_delta':'none'}
    save(out/'comparison.json',result)
    print(json.dumps({'decision':decision,'status':queue['status'],'frames':result['new_generated_frames'],
                      'pairs':[{c['log_id'][:8]:c['pairs']} for c in cases]}),flush=True)


if __name__=='__main__':main()
