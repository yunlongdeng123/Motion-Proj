"""Publish actual final full49 results in the user's shared-timeline template."""
from pathlib import Path
import json,re,sys,shutil
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
I=W.parent/'identity_completion';sys.path.insert(0,str(I));from build_sync_player import encode
ACTORS=[6,37,5,26,18]
def main():
 original=(O.parent/'identity_comparison_20261018/index.html').read_text(encoding='utf-8')
 data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',original,re.S).group(1))
 def prefix(x):
  if isinstance(x,str) and x.startswith('branch_videos/'):return '../identity_comparison_20261018/'+x
  if isinstance(x,dict):return {k:prefix(v) for k,v in x.items()}
  if isinstance(x,list):return [prefix(v) for v in x]
  return x
 data=prefix(data);videos=O/'branch_videos';videos.mkdir(exist_ok=True)
 ready=[]
 for model,label,src,allvehicles in [('matched5','新五身份：GN11 30K＋实例 30K',W/'direct_full49',False),('allvehicles','LiDAR∩相机20身份：第一阶段30K',W/'all_vehicle_full49',True)]:
  receipt=src/'receipt.json'
  if not receipt.exists():continue
  r=json.loads(receipt.read_text(encoding='utf-8'));assert r['frames']==49 and r['parameter_updates']==0
  m=dict(label=label,ready=True,base={},actors={},all_vehicle_stage1=allvehicles)
  def add(key,file,actor=None):
   dst=videos/(model+'_'+(f'a{actor:02d}_' if actor is not None else '')+key+'.mp4')
   encode(src,file,dst)
   (m['base'] if actor is None else m['actors'].setdefault(str(actor),{}))[key]='branch_videos/'+dst.name
  for k,f in [('observed','gt.png'),('full','full.png'),('dynamic','dynamic.png'),('dynamic_alpha','dynamic_alpha.png'),('static','static.png'),('static_alpha','static_alpha.png'),('delete_all','delete_all.png')]:add(k,f)
  if not allvehicles:
   for a in ACTORS:
    for k,f in [('isolated','isolated'),('isolated_alpha','alpha'),('delete','delete'),('dynamic_delete','dynamic_delete'),('dynamic_delete_alpha','dynamic_delete_alpha')]:add(k,f'a{a:02d}_{f}.png',a)
  data['models'][model]=m;ready.append(model)
 assert ready,'no actual final result; preserve existing report'
 s=(W/'fullframe_player_template.html').read_text(encoding='utf-8')
 s=s.replace('<title>五身份动态／静态／Delete 同步视频</title>','<title>新30K实验：完整画面同步视频</title>')
 needle=" const stat=alpha?b.static_alpha:"
 assert needle in s
 inject=""" if(m.all_vehicle_stage1){section.querySelector('.chips').textContent='LiDAR∩相机20身份 · 仅第一阶段30K · 其余车辆为RGB上下文';
 tile(grid,name,'观测完整RGB',b.observed);tile(grid,name,'重建完整场景',b.full);tile(grid,name,'全部纳入车辆动态分支'+(alpha?' alpha':''),alpha?b.dynamic_alpha:b.dynamic);
 tile(grid,name,'静态街景'+(alpha?' alpha':''),alpha?b.static_alpha:b.static);tile(grid,name,'删除全部纳入车辆 · 完整画面',b.delete_all,true);continue;}
 const stat=alpha?b.static_alpha:"""
 s=s.replace(needle,inject)
 s=s.replace('<a href="diagnostics.html">统计、固定锚帧与来源说明</a> · <a href="tristate/index.html">补充：Gaussian 中心与三值证据</a>',
  '<a href="index.html">实验状态与输入说明</a> · <a href="../identity_comparison_20261018/tristate/index.html">历史归属证据</a>')
 s=s.replace('@@DATA@@',json.dumps(data,ensure_ascii=False,separators=(',',':')))
 assert "$('zoom')" not in s and 'id="export"' not in s
 (O/'results.html').write_text(s,encoding='utf-8')
 (O/'results_status.json').write_text(json.dumps(dict(completed_models=ready,frames=49,ROI=False,full_frame=True,human_verdict=None),indent=2))
 print(json.dumps(dict(actual_models_added=ready,report=str(O/'results.html')),ensure_ascii=False))
if __name__=='__main__':main()
