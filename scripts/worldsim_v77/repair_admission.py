"""Ω调用方须通过此边界取候选RGB；guard失败、视觉反例或未检查均不给路径。"""
from repair_common import *
def approved_paths(root,scene,arm='evidence_first'):
 root=pathlib.Path(root);g=next(s for s in read(root/'guard_summary.json')['scenes'] if s['scene']==scene)
 if g['arms'][arm]['blocked_frames']:raise ValueError('blocked_vehicle: 生成区重新出现车辆')
 if g['arms'][arm]['source_miss_frames']:raise ValueError('guard_unreliable: 原图目标有漏检，不能自动准入')
 obs=read(root/'observations.json').get(scene,{})
 if obs.get('assistant_visual_candidate') is not True:raise ValueError('visual_not_admitted: 存在反例或尚未检查')
 s=next(s for s in read(root/'registration.json')['scenes'] if s['name']==scene)
 paths=[root/scene/arm/f'{i:05}.png' for i in range(len(s['source_frames']))]
 assert all(p.exists() for p in paths)
 return paths
def main():
 rows=[]
 for s in read(ROOT/'registration.json')['scenes']:
  try:paths=approved_paths(ROOT,s['name']);rows.append({'scene':s['name'],'candidate_paths':[str(p) for p in paths],'candidate':True,'scope':'仅所登记主相机；不称全六相机世界通过'})
  except ValueError as e:rows.append({'scene':s['name'],'candidate':False,'candidate_paths':[],'reason':str(e)})
 dump(ROOT/'background_admission.json',{'scenes':rows,'gate_enforced_before_omega':True,'human_verdict':None});print(rows)
if __name__=='__main__':main()
