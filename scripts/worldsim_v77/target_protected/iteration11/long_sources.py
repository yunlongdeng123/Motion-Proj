"""扩为30个真实曝光；先冻结来源，不复制RGB、不偷用旧一秒QA。"""
from pathlib import Path
import os,sys,copy,bisect,time,shutil
from collections import defaultdict,Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected'
sys.path.insert(0,str(S));sys.path.insert(0,str(S/'iteration4'));sys.path.insert(0,str(S/'iteration9'))
from prepare_source_pool import read,dump,stream,mat,project,good
from sample_windows import match_exposures
from temporal_factory import T,R8,F
from pyquaternion import Quaternion
import numpy as np
O=T/'r16';ROOT=O/'factory';META=T.parent/'WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval'
PLAN='''# r16：三秒真实曝光与遮挡过程数据

task WS-V77-TARGET-PROTECTED-20260929/r16，wm-3090-1001，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 S[旧合法空间来源／完整GT支撑] --> E[30个真实有序曝光，约3秒]
 E --> M[SAM2真实保护实例，隔离暂存]
 M --> A[旧虚拟路径延长＋相对速度]
 A --> Q[全帧空间／mask合同＋独立AI2]
 Q --> C[覆盖准入，随后单独登记训练]
```

r15已得到6扫过train/4world，但val扫过仍0。一秒可能截不到遮挡跨车身的过程；本轮只扩真实观测时长，不改变网络、loss或增加旧训练步数。依据旧合法空间锚点、完整8关键帧GT支撑冻结最多48来源/每world≤2；所有曾训练world只能train，其余曝光world作DEV val，未曝光final不使用。先每world一例，再按真实输入变化选第二例，不依据任何模型输出。

曝光固定原始keyframe起点后约50ms的0:30个10Hz目标，有序唯一匹配，误差≤55ms、间隔≤180ms、不复制、不插帧。只保留全窗口高质量primary，secondary需完整GT和同标准观测；其它真实参与者仍由全GT上下文约束，不能因不分割而忽略障碍。旧一秒AI2不传递到新三秒数据。真实Y不修改，全部合成X影响先擦除再resize/encoder。

只延长已有虚拟空间路径；最多4旧锚点、两种端点目标±.4拟合相对纵向速度（.25–8m/s），另外一个世界静止对照。全30帧复查原道路/地面支持≤2.5m/GT与ego距离≥.3m/禁压ego/尺寸/截边/深度/连续性/85%最大保护遮挡。可复用r12明确修正：原拟合平面±.08m真实地面返回用于支持，阈值不放宽，修正前后证据留存。独立gpt-6-sol xhigh无fast逐例0/15/29视觉，仅AI2可进入下一阶段。

目标结合旧合格过程达到8扫过train/≥3world、3扫过val/≥2未训练world，整体约50train/≥20world、单world≤3。两种端点拟合及一个静止对照只跑一次固定来源；不足则停止本次具体搜索、保留拒绝，不无限放宽阈值或速度网格。训练0；30帧训练/10帧稀疏采样及fps条件尚未选择，需后续明确采样合同和显存探针后单独登记，不能把3秒源数据冒充3秒模型验证。

真实DELETE＋补景跨至少两scene的目标删除/保护对象与视频收益仍是收口条件。新数据AI2、合成MAE下降或loss下降不能触发关机；人工0/1/2空。保留所有旧/新资产和拒绝，无新自动化。
'''

def freeze():
 O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True)
 if (O/'source_selection.json').exists():return read(O/'source_selection.json')['sources']
 sources={c['source_id']:c for c in read(F/'source_manifest.json')['clips']};pool=read(R8/'all_asset_candidates.json')['candidates'];anchors=Counter(c['source_id'] for c in pool)
 trained=set(read(R8/'source_split.json')['old_r7_training_scenes'])
 for cat in [R8/'dataset_catalog.json',T/'r10/dataset_catalog.json']:
  trained|={c['receiver_scene'] for c in read(cat)['cases'] if c['split']=='train'}
 candidates=[]
 for sid in sorted(anchors):
  c=sources[sid];primary=c['actors'][0];annotations=primary.get('keyframe_annotations',[])
  if len(annotations)!=8 or not all(annotations):continue
  fs=c['frames'];bb=np.array([f['actors'][0]['projection']['box_xyxy'] for f in fs]);width=bb[:,2]-bb[:,0];centers=(bb[:,:2]+bb[:,2:])/2
  score=float(np.linalg.norm(centers[-1]-centers[0])+12*np.linalg.norm(np.array(fs[-1]['camera_to_world'])[:3,3]-np.array(fs[0]['camera_to_world'])[:3,3]))
  candidates.append({'source_id':sid,'scene':c['scene'],'split':'train' if c['scene'] in trained else 'validation','score_input_only':score,'width_median':float(np.median(width)),'old_anchor_count':anchors[sid]})
 ordered=sorted(candidates,key=lambda r:(r['split']=='train',-r['score_input_only'],r['source_id']));selected=[];counts=Counter()
 for cap in [1,2]:
  for row in ordered:
   if len(selected)>=48:break
   if row in selected or counts[row['scene']]>=cap:continue
   selected.append(row);counts[row['scene']]+=1
 result={'sources':selected,'max_sources':48,'worlds':len(counts),'max_per_world':2,'prior_training_worlds':sorted(trained),'rule':'complete primary GT, old legal spatial anchors; first one/world then second by real input projection+ego displacement','uses_model_outputs':False,'human_verdict':None}
 dump(O/'source_selection.json',result)
 E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r16';E.mkdir(parents=True,exist_ok=True)
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r16','stage':'registered_long_real_exposure_data_only','host':'wm-3090-1001','source_count':len(selected),'worlds':len(counts),'frames':30,'training_steps':0,'seed_model':None,'final_used':False,'human_verdict':None,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','automatic_shutdown':False,'real_cross_case_benefit_demonstrated':False}
 for root in [O,E]:dump(root/'run.json',run);(root/'plan.md').write_text(PLAN)
 dump(E/'source_selection.json',result)
 b=O/'docs_before_r16';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']:shutil.copy2(P/rel,b/Path(rel).name)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();at=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r16 | 约3秒30真实曝光，来源冻结与过程质量先行；训练未登记、关机条件false | [预案](autoresearch/worldsim_v77/target_protected_20260929/r16/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 status=P/'docs/RESEARCH_STATUS.md';status.write_text(status.read_text().replace('下一步延长为约3秒真实连续曝光并补隔离验证过程。','当前唯一新run r16延长为约3秒真实连续曝光并补隔离验证过程，来源冻结最多48／每world最多2，训练0。'))
 print('FROZEN',len(selected),len(counts),dict(Counter(r['split'] for r in selected)),flush=True);return selected

def resolve():
 rows=freeze();all_sources={c['source_id']:c for c in read(F/'source_manifest.json')['clips']};sel=[all_sources[r['source_id']] for r in rows];wanted={(c['scene_token'],c['camera']) for c in sel}
 cameras={};cal={}
 for cp in [T/'r4/camera_metadata_cache.json',T/'r8/short_sources/camera_metadata_cache.json']:
  d=read(cp);cal.update(d['cal']);cameras.update({tuple(k.split('|')):v for k,v in d['camera'].items()})
 missing=wanted-cameras.keys()
 if missing:
  samples={r['token']:r for r in read(META/'sample.json')};ch={r['token']:r['channel'] for r in read(META/'sensor.json')};cal={r['token']:r for r in read(META/'calibrated_sensor.json')};additional=defaultdict(list)
  for r in stream(META/'sample_data.json'):
   key=(samples[r['sample_token']]['scene_token'],ch[cal[r['calibrated_sensor_token']]['sensor_token']])
   if key in missing:additional[key].append({k:r[k] for k in ['token','sample_token','timestamp','filename','width','height','ego_pose_token','calibrated_sensor_token']})
  cameras.update({k:sorted(v,key=lambda r:r['timestamp']) for k,v in additional.items()})
 dump(O/'camera_metadata_cache.json',{'camera':{'|'.join(k):v for k,v in cameras.items() if k in wanted},'cal':cal})
 clips=[];rejected=[];need=set();sampling=[]
 for src in sel:
  c=copy.deepcopy(src);exposures=cameras[c['scene_token'],c['camera']];ts=[r['timestamp'] for r in exposures];anchor=bisect.bisect_left(ts,c['start_timestamp_us']+50000)
  if anchor==len(ts):rejected.append({'source_id':c['source_id'],'reason':'no_start_exposure'});continue
  targets=[ts[anchor]+100000*i for i in range(30)];ix=match_exposures(exposures,targets);sampling.append({'source_id':c['source_id'],'targets':targets,'selected_timestamps':None if ix is None else [ts[j] for j in ix]})
  if ix is None:rejected.append({'source_id':c['source_id'],'reason':'no_ordered_unique_30_exposures'});continue
  c['actors']=[a for a in c['actors'] if len(a.get('keyframe_annotations',[]))==8 and all(a['keyframe_annotations'])];c['frames']=[]
  for i,(t,j) in enumerate(zip(targets,ix)):
   f=exposures[j]|{'frame':i,'requested_timestamp_us':t,'delta_ms':abs(t-ts[j])/1000};c['frames'].append(f);need.add(f['ego_pose_token'])
  clips.append(c)
 ego={r['token']:r for r in stream(META/'ego_pose.json') if r['token'] in need};valid=[];secondary_dropped=[]
 for c in clips:
  failures=Counter();bad_tokens=set()
  for f in c['frames']:
   ca=cal[f['calibrated_sensor_token']];e=ego[f['ego_pose_token']];c2w=mat(e['translation'],e['rotation'])@mat(ca['translation'],ca['rotation']);k=np.array(ca['camera_intrinsic'],float);k[0]*=1024/f['width'];k[1]*=576/f['height'];f['camera_to_world']=c2w.tolist();f['intrinsics_1024']=k.tolist();f['actors']=[]
   stamps=c['keyframe_timestamps'];i=bisect.bisect_right(stamps,f['timestamp'])
   if not 0<i<len(stamps) or stamps[i]-stamps[i-1]>600000:failures['annotation_unbracketed']+=1;continue
   u=(f['timestamp']-stamps[i-1])/(stamps[i]-stamps[i-1])
   for a in c['actors']:
    lo,hi=a['keyframe_annotations'][i-1:i+1];obj={'translation':((1-u)*np.array(lo['translation'])+u*np.array(hi['translation'])).tolist(),'size':((1-u)*np.array(lo['size'])+u*np.array(hi['size'])).tolist(),'rotation':Quaternion.slerp(Quaternion(lo['rotation']),Quaternion(hi['rotation']),amount=u).elements.tolist()};pr=project(obj,{'w2c':np.linalg.inv(c2w),'k':k});f['actors'].append(obj|{'instance_token':a['instance_token'],'category':a['category'],'projection':pr})
    if not good(pr):bad_tokens.add(a['instance_token'])
  primary=c['actors'][0]['instance_token']
  if primary in bad_tokens:failures['primary_weak_border_or_small_in_long_window']+=1
  if failures:rejected.append({'source_id':c['source_id'],'reasons':dict(failures)});continue
  secondary_dropped.append({'source_id':c['source_id'],'tokens':sorted(bad_tokens),'reason':'full-window high-quality requirement; remains obstacle in full context'})
  c['actors']=[a for a in c['actors'] if a['instance_token'] not in bad_tokens]
  for f in c['frames']:f['actors']=[a for a in f['actors'] if a['instance_token'] not in bad_tokens]
  c['source_window']={'selection':'ordered unique actual exposures 0:30 of original fixed10Hz grid','interpolated_images':False,'old_one_second_QA_inherited':False};c['source_split']=next(r['split'] for r in rows if r['source_id']==c['source_id']);valid.append(c)
 dump(ROOT/'source_manifest.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r16','clips':valid,'sampling_rejected':rejected});dump(O/'sampling_control.json',{'stage':'complete','selected':len(rows),'valid':len(valid),'frames':len(valid)*30,'rejected':rejected,'secondary_dropped':secondary_dropped,'matching':sampling,'no_interpolation':True,'human_verdict':None})
 context={c['source_id']:c for c in read(F/'source_context.json')['clips']};dump(ROOT/'source_context.json',{'clips':[context[c['source_id']] for c in valid],'purpose':'GT geometry and LiDAR only; never model conditioning'})
 names={f['filename'] for c in valid for f in c['frames']}
 for c in valid:
  for i in [0,3,6]:names.add(context[c['source_id']]['frames'][i]['sensors']['LIDAR_TOP']['filename'])
 (ROOT/'rgb').mkdir(exist_ok=True)
 bases=[F/'rgb',T/'r8/short_sources/factory/rgb',T/'r1/rgb',T/'r1/expanded_factory/rgb',T/'r1/source_extension/rgb',T/'r3/factory/rgb',T/'r4/factory/rgb'];missing=[]
 for name in sorted(names):
  p=next((b/name for b in bases if (b/name).is_file()),None);dest=ROOT/'rgb'/name
  if p is None:missing.append(name);continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(p.resolve())
 dump(ROOT/'required_files.json',missing);dump(O/'RGB_request.json',{'total_files':len(names),'cache_reused':len(names)-len(missing),'missing':missing,'scenes':sorted({c['scene'] for c in valid}),'actors_to_segment':sum(len(c['actors']) for c in valid),'bytes_estimate_from_cached_RGB':len(missing)*350000,'available_space_bytes':shutil.disk_usage(ROOT).free,'human_verdict':None})
 if not (ROOT/'maps').exists():(ROOT/'maps').symlink_to((F/'maps').resolve(),target_is_directory=True)
 print('LONG_SOURCE',len(valid),len(valid)*30,'actors',sum(len(c['actors']) for c in valid),'missing',len(missing),'rejections',len(rejected),flush=True)
 if missing:
  from prepare_context_extract import extract
  extract(ROOT,Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval'))
 from PIL import Image
 for c in valid:
  for f in c['frames']:
   with Image.open(ROOT/'rgb'/f['filename']) as im:im.load();assert im.size==(1600,900)
 dump(O/'source_ready.json',{'stage':'ready_for_quarantined_SAM2','clips':len(valid),'frames':len(valid)*30,'actors':sum(len(c['actors']) for c in valid),'source_quality_inherited':False,'synthetic_admission':0,'training_steps':0,'human_verdict':None});print('SOURCE_READY',len(valid),flush=True)

if __name__=='__main__':resolve()
