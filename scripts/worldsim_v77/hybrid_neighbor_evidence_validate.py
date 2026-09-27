"""对实际r14–r17产物检查来源隔离、条件单变量与写回合同。"""
from pathlib import Path
import json,numpy as np
from PIL import Image
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
summary=read(BASE/'r14/visible_summary.json');index=read(BASE/'r14/source_index.json');pool=np.load(BASE/'r14/sources.npz');nchecks=0;rows=[]
for r in summary['queries']:
 p=BASE/'r14'/f'{r["frame"]:03}_{r["camera"]}'
 for arm in ['strict','dense']:
  z=np.load(p/f'{arm}_provenance.npz');accepted=np.array(Image.open(p/f'{arm}_accepted.png'))>0
  ids=z['source_id'];point=z['source_point'];second=z['second_source'];color=z['rgb'];single=ids>=0
  for j,src in enumerate(index):
   keep=single&(ids==j)
   assert src['frame']!=r['frame'] or not keep.any()
   assert np.array_equal(color[keep],pool[f'{j}_rgb'][point[keep]])
   if arm=='strict':assert pool[f'{j}_strict'][point[keep]].all()
  assert (ids[accepted]>=0).all() and (second[accepted]>=0).all()
  fs=np.array([i['frame'] for i in index]);assert (abs(fs[ids[accepted]]-fs[second[accepted]])>=5).all()
  assert (fs[ids[accepted]]!=r['frame']).all() and (fs[second[accepted]]!=r['frame']).all()
  rows.append(dict(query=r['frame'],camera=r['camera'],arm=arm,accepted=int(accepted.sum()),no_same_timestamp_source=True,source_rgb_exact=True,two_distinct_times=True));nchecks+=1
r15=read(BASE/'r15/state.json');assert r15['state']=='complete'
for arm in ['deletion','neighbor']:
 for i in range(10):
  source=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));raw=np.array(Image.open(BASE/'r15'/arm/'native'/f'{i:05}.png'));final=np.array(Image.open(BASE/'r15'/arm/'final'/f'{i:05}.png'));m=np.array(Image.open(BASE/'r15/write'/f'{i:05}.png'))>0
  assert np.array_equal(final[m],raw[m]) and np.array_equal(final[~m],source[~m])
v=read(BASE/'r16/condition_validation.json');assert v['all_other_conditions_exact_equal'] and v['frames_1_to_9_context_exact_equal'];assert v['first_frame_changed_pixels']==6910
r17=read(BASE/'r17/replay_validation.json');assert r17['reproduced_video_exact'] and r17['decoded_views']==21;assert all(r['max_pixel_difference']==0 for r in r17['video_checks'])
r18=read(BASE/'r18/state.json');assert r18['state']=='complete'
for i in range(10):
 source=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));raw=np.array(Image.open(BASE/'r18/native'/f'{i:05}.png'));final=np.array(Image.open(BASE/'r18/final'/f'{i:05}.png'));m=np.array(Image.open(BASE/'r15/write'/f'{i:05}.png'))>0
 assert np.array_equal(final[m],raw[m]) and np.array_equal(final[~m],source[~m])
out=dict(visible_source_contracts=rows,source_queries_arms_checked=nchecks,r15_writeback_frames_checked=20,r16_single_changed_condition=v,r17_exact_reproduction_frames=10,r17_decoded_views=21,r18_writeback_frames_checked=10,quality_pass=False,human_verdict=None)
(BASE/'neighbor_review/evidence_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('EVIDENCE_CONTRACTS_PASSED',nchecks,20,10,21)
