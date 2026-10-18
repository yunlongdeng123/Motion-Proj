"""Publish actual final full49 results in the user's shared-timeline template."""
from pathlib import Path
import json,re,sys,shutil,subprocess,uuid
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
I=W.parent/'identity_completion';sys.path.insert(0,str(I));from build_sync_player import encode
ACTORS=[6,37,5,26,18]

def video_contract(path):
 probe=shutil.which('ffprobe')
 if not probe:raise FileNotFoundError('ffprobe')
 r=subprocess.run([probe,'-v','error','-select_streams','v:0','-count_frames','-show_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate:frame=key_frame,pict_type','-of','json',str(path)],capture_output=True,text=True,check=True)
 data=json.loads(r.stdout);s=data['streams'][0]
 assert (s['width'],s['height'])==(480,320) and int(s['nb_read_frames'])==49 and s['r_frame_rate']=='10/1',(path,s)
 assert len(data['frames'])==49 and all(f['key_frame']==1 and f['pict_type']=='I' for f in data['frames']),path
 return s

def encode_complete(source,filename,target):
 # Interrupted encodes stay recoverable, but are never reused as final media.
 if target.exists():
  try:video_contract(target);return
  except (subprocess.CalledProcessError,AssertionError,KeyError,ValueError):
   target.rename(target.with_name(target.stem+'.incomplete_'+uuid.uuid4().hex[:8]+'.mp4'))
 pending=target.with_name(target.stem+'.encoding.mp4')
 if pending.exists():
  try:video_contract(pending);pending.replace(target);return
  except (subprocess.CalledProcessError,AssertionError,KeyError,ValueError):
   pending.rename(pending.with_name(pending.stem+'.incomplete_'+uuid.uuid4().hex[:8]+'.mp4'))
 encode(source,filename,pending);video_contract(pending);pending.replace(target)
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
 for model,label,src,allvehicles in [('gn11five','新五身份 GN11 · 第一阶段30K',W/'GN11_stage1_full49',True),('matched5','新五身份：GN11 30K＋实例 30K',W/'direct_full49',False),('allvehicles','LiDAR∩相机20身份：第一阶段30K',W/'all_vehicle_full49',True)]:
  receipt=src/'receipt.json'
  if not receipt.exists():continue
  r=json.loads(receipt.read_text(encoding='utf-8'));assert r['frames']==49 and r['parameter_updates']==0
  if not allvehicles:
   assert set(r['actors'])==set(ACTORS) and len(r['records'])==245
   assert {(x['frame_id'],x['actor']) for x in r['records']}=={(f'0008{f:03d}',a) for f in range(89,138) for a in ACTORS}
  else:assert len(r['records'])==49
  m=dict(label=label,ready=True,base={},actors={},all_vehicle_stage1=allvehicles,physical_sun_integral=False)
  if allvehicles:m['stage1_scope_label']='五身份动态集合 · GN11第一阶段30K' if model=='gn11five' else 'LiDAR∩相机20身份 · 第一阶段30K · 其他车辆为上下文'
  def add(key,file,actor=None):
   dst=videos/(model+'_'+(f'a{actor:02d}_' if actor is not None else '')+key+'.mp4')
   encode_complete(src,file,dst)
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
 inject=""" if(m.all_vehicle_stage1){section.querySelector('.chips').textContent=m.stage1_scope_label||'第一阶段动态集合';
 tile(grid,name,'观测完整RGB',b.observed);tile(grid,name,'重建完整场景',b.full);tile(grid,name,'全部纳入车辆动态分支'+(alpha?' alpha':''),alpha?b.dynamic_alpha:b.dynamic);
 tile(grid,name,'静态街景'+(alpha?' alpha':''),alpha?b.static_alpha:b.static);tile(grid,name,'删除全部纳入车辆 · 完整画面',b.delete_all,true);continue;}
 const stat=alpha?b.static_alpha:"""
 s=s.replace(needle,inject)
 s=s.replace('let frame=113,entries=', 'let frame=128,entries=')
 s=s.replace("const available=Object.keys(DATA.models),choices=[", "const available=Object.keys(DATA.models),choices=[['gn11five','新五身份GN11第一阶段30K'],['gn11five,old5','新GN11第一阶段 与 历史旧5两阶段（阶段不同）'],")
 s=s.replace('...available.map(n=>[n,DATA.models[n].label])',"...available.filter(n=>n!=='gn11five').map(n=>[n,DATA.models[n].label])")
 s=s.replace('<a href="diagnostics.html">统计、固定锚帧与来源说明</a> · <a href="tristate/index.html">补充：Gaussian 中心与三值证据</a>',
  '<a href="index.html">实验状态与输入说明</a> · <a href="../identity_comparison_20261018/tristate/index.html">历史归属证据</a>')
 s=s.replace('@@DATA@@',json.dumps(data,ensure_ascii=False,separators=(',',':')))
 assert "$('zoom')" not in s and 'id="export"' not in s
 (O/'results.html').write_text(s,encoding='utf-8')
 (O/'results_status.json').write_text(json.dumps(dict(completed_models=ready,frames=49,ROI=False,full_frame=True,human_verdict=None),indent=2))
 print(json.dumps(dict(actual_models_added=ready,report=str(O/'results.html')),ensure_ascii=False))
if __name__=='__main__':main()
