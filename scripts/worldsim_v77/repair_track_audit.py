"""静态DELETE指令身份、轨迹与新增占据检查；不批准MOVE。"""
from repair_common import *
rows=[]
for s in read(ROOT/'registration.json')['scenes']:
 frames=read(ROOT/s['name']/'frames.json');p=[];ids=[]
 for f in s['source_frames']:
  b=next(b for b in frames[str(f)]['all_boxes'] if b['actor_id']==s['actor']);p.append(np.array(b['pose'])[:3,3]);ids.append(b['track_id'])
 assert len(set(ids))==1 and ids[0]==s['track_id']
 p=np.array(p);rows.append({'scene':s['name'],'actor':s['actor'],'track_id':s['track_id'],'frames_with_identity':len(p),'start_to_end_xy_m':float(np.linalg.norm(p[-1,:2]-p[0,:2])),'max_xy_from_first_m':float(np.linalg.norm(p[:,:2]-p[0,:2],axis=1).max()),'command':'DELETE','new_occupied_volume':0,'legality_scope':'已核对目标身份与删除语义；只移除该目标，不执行MOVE，不声称完整交通规则审核。'})
dump(ROOT/'track_audit.json',rows);print(rows)
