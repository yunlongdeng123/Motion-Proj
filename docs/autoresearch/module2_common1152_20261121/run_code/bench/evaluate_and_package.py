"""CPU-only fixed-protocol evaluation and small bundle after all new1152 students export."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,hashlib,shutil,tarfile,time,traceback
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_common1152_20261121')
Q=P.parent/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics'
sys.path.insert(0,str(Q))
from road_detail_metrics import DetailConfig,ResolutionSpec,TileLayout,evaluate_detail,_strict_core
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def path(value):
 p=Path(value);return p if p.is_absolute()else P/p
def rgb(a):
 a=np.asarray(a)
 if a.shape==(3,768,1152):a=a.transpose(1,2,0)
 if a.shape!=(768,1152,3):raise ValueError('must be actual1152x768 RGB, no resize')
 if a.dtype==np.uint8:a=a.astype(np.float64)/255
 if not np.isfinite(a).all():raise ValueError('nonfinite RGB')
 return a.astype(np.float64)
def saveimage(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(p)
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relative(p):return p.relative_to(P).as_posix()
def box(mask):
 y,x=np.nonzero(mask)
 if not len(x):return [0,0,128,128]
 # source-mask bounding region, fixed before observing students; no error-driven crop selection.
 x0=max(0,int(x.min())-8);x1=min(1152,int(x.max())+9);y0=max(0,int(y.min())-8);y1=min(768,int(y.max())+9)
 return [x0,y0,x1,y1]
def imagecopy(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 with Image.open(src)as im:
  if im.size!=(1152,768):raise ValueError('student PNG not native1152')
  im.convert('RGB').save(dest)
def blackgap(modelid,raw,source,domain,alpha,pred,teacher,packet):
 n=int(domain.sum());no=domain&(alpha<.05);thin=domain&(alpha>=.05)&(alpha<.95);dark=domain&(alpha>=.95)&(pred.mean(-1)<.05)&(teacher.mean(-1)>=.10)
 row=dict(id=modelid,raw=raw,source=source,denominator=n,no_support_n=int(no.sum()),low_alpha_n=int(thin.sum()),high_alpha_dark_n=int(dark.sum()),exact_zero_n=int((domain&(alpha<=1e-8)).sum()),no_support_fraction=float(no.sum()/n)if n else None,low_alpha_fraction=float(thin.sum()/n)if n else None,high_alpha_dark_fraction=float(dark.sum()/n)if n else None,joint_static_occluded_n=None,joint_static_occluded_fraction=None,available_joint_contribution=False)
 for key in ['road_final_weight','road_alphaT','road_joint_contribution_alpha','joint_road_weight']:
  if key in packet:
   w=np.asarray(packet[key]).squeeze()
   if w.shape==alpha.shape:
    hidden=domain&(alpha>=.95)&(w<.5);row.update(joint_static_occluded_n=int(hidden.sum()),joint_static_occluded_fraction=float(hidden.sum()/n)if n else None,available_joint_contribution=True,joint_contribution_key=key);break
 return row
def run(manifest):
 start=time.monotonic();bench=P/'bench';delivery=P/'delivery';bench.mkdir(exist_ok=True);delivery.mkdir(exist_ok=True)
 inp=json.loads(manifest.read_text());teacher_manifest=P/'teacher/manifest.json';tm=json.loads(teacher_manifest.read_text());rows=inp.get('rows',[])
 if not rows:
  for model in inp.get('models',[]):
   for raw,frame in model['frames'].items():rows.append(dict(group=model['round'],arm=model['id'],name=model['name'],budget=model.get('budget'),raw=int(raw),packet=frame['native_npz'],road_png=frame.get('road_png'),joint_png=frame.get('joint_png')))
 if not rows:raise ValueError('no exported students')
 protocol=dict(schema='road_common1152_protocol.v1',resolution=[1152,768],teacher_selection='fixed legacy T0seed101, no student-error selection',teacher_manifest_sha256=sha(teacher_manifest),signal='stable-alpha unpremultiplied road sRGB, teacher sRGB; premult and joint RGB errors also separate',source_domains='blackgap denominator: complete new1152 legal2D A/C/S & ~K, canonical missing retained; color core additionally canonical-valid',stable_alpha='fixed parent alpha>=.95; same masks for every student',highpass_sigma_native_px=[1,2,4],gaussian_truncate=3,strict_core='source-specific full finite gaussian support, radius round(3sigma)',edge_measurement='RGB tensor NMS; fixed strength2/255, nearest match1nativepx; appearance edges, no geometry-GT claim',flat_measurement='teacher per-pixel RGB HF RMS<=2/255, fixed',no_support='road alpha<.05',low_alpha='.05<=road alpha<.95',high_alpha_dark='alpha>=.95, unpremult meanRGB<.05, teacher meanRGB>=.10; diagnostic category only',static_occlusion='optional exported road alphaT contribution<.5 on high standalone alpha; absent => null, never infer from road alpha',coverage_warning='coverage/readonly residual cannot be attributed to missing fine detail; diagnostic only, no aggregate quality pass',source_sha256={'export_manifest':sha(manifest),'targets_manifest':sha(P/'cpu_inputs/targets_manifest.json')})
 write(bench/'protocol.json',protocol)
 fixed={};boxes={};teacherframes={}
 for raw in [86,110]:
  d=load(P/f'cpu_inputs/{raw:03d}_targets.npz');fixed[raw]=d;boxes[str(raw)]={s:box(d[s+'_source'])for s in ['A','C','S']};teacherframes[str(raw)]={}
  for s in ['A','C','S']:
   dest=delivery/f'teacher/{raw:03d}_{s}.png';a=rgb(d['target_'+s]);saveimage(dest,a);teacherframes[str(raw)][s]=relative(dest);x0,y0,x1,y1=boxes[str(raw)][s];crop=dest.with_name(f'{raw:03d}_{s}_crop.png');saveimage(crop,a[y0:y1,x0:x1]);teacherframes[str(raw)][s+'_crop']=relative(crop)
 models={};metricrows=[];blackrows=[];files={};config=DetailConfig(pixel_scales=(1.,2.,4.),flat_hf_thresholds=(2/255,),edge_thresholds=(2/255,),edge_tolerances_pixels=(1.,),edge_angle_tolerances_degrees=(30.,),worst_tiles=0)
 for item in rows:
  group=str(item['group']).lstrip('R');arm=str(item['arm']);modelid=arm if arm.startswith('R'+group+'_')else 'R'+group+'_'+arm;raw=int(item['raw']);packetpath=path(item.get('packet',item.get('native_npz')));packet=load(packetpath);road=rgb(packet.get('road_rgb',packet.get('road_only_rgb')));alpha=np.asarray(packet.get('road_alpha',packet.get('road_only_alpha'))).squeeze()
  if alpha.shape!=(768,1152):raise ValueError('alpha must1152x768')
  joint=rgb(packet['joint_rgb'])if 'joint_rgb'in packet else None;effective=np.divide(road,alpha[...,None],out=np.zeros_like(road),where=alpha[...,None]>1e-8)
  model=models.setdefault(modelid,dict(id=modelid,round='R'+group,name=item.get('name',arm),budget=item.get('budget',{}),frames={},authority=item.get('authority'),authority_SHA256=item.get('authority_SHA256'),renderer=item.get('renderer'),data_role=item.get('data_role','all_qualified_newteacher_fit')));frame={};folder=delivery/'models'/modelid
  rp=folder/f'{raw:03d}_road.png';saveimage(rp,road);frame['road']=relative(rp)
  if joint is not None:jp=folder/f'{raw:03d}_joint.png';saveimage(jp,joint);frame['joint']=relative(jp)
  if raw in fixed:
   d=fixed[raw];fixedalpha=d['fixed_alpha'];alpha_diff=float(np.max(np.abs(alpha-fixedalpha)));frame['alpha_parent_max_change']=alpha_diff
   for source in ['A','C','S']:
    domain=d[source+'_source'];context=domain&d['canonical_valid']&(fixedalpha>=.95);teacher=rgb(d['target_'+source]);bg=blackgap(modelid,raw,source,domain,alpha,effective,teacher,packet);bg['canonical_missing_n']=int((domain&~d['canonical_valid']).sum());blackrows.append(bg)
    # New native sigma filters and scores, never old-image resizing.
    report=evaluate_detail(effective,teacher,context_mask=context,score_mask=context,resolution=ResolutionSpec(1152,768,'new_native1152','srgb'),tile_layout=TileLayout(768,1152),teacher_role='real_observation'if source=='A'else'frozen_pseudo_teacher',config=config,color_space='stable_alpha_unpremult_srgb',input_conditioning={'alpha_parent_min':.95,'teacher_source':source})
    reportfile=bench/'details'/f'{modelid}_{raw:03d}_{source}.json';write(reportfile,report);edge=report['edges']['threshold_curves'][0]['curves'][0];prec,rec=edge['precision'],edge['recall'];f1=None if prec is None or rec is None else(2*prec*rec/(prec+rec)if prec+rec else 0.)
    for scale in report['scales']:
     sigma=scale['sigma_pixels'];core=context&_strict_core(context,int(3*sigma+.5));err=effective[core]-teacher[core];prem=road[core]-fixedalpha[core,None]*teacher[core];jerr=joint[core]-teacher[core]if joint is not None else None
     metricrows.append(dict(id=modelid,round='R'+group,raw=raw,source=source,scale=sigma,RGB_MAE=float(np.abs(err).mean())if err.size else None,RGB_RMSE=float(np.sqrt(np.mean(err**2)))if err.size else None,premult_RGB_MAE=float(np.abs(prem).mean())if prem.size else None,joint_RGB_MAE=float(np.abs(jerr).mean())if jerr is not None and jerr.size else None,HF_cosine=scale['core']['HF_cosine'],HF_energy_ratio=scale['core']['HF_energy_ratio'],HF_MAE=scale['core']['HF_MAE'],HF_RMSE=scale['core']['HF_RMSE'],edge_F1=f1,flat_pred_RMS=scale['teacher_flat_spurious'][0]['predicted_HF_RMS'],core_n=scale['core_pixels'],stable_score_n=int(context.sum()),blackgap_no_support_n=bg['no_support_n'],blackgap_thin_n=bg['low_alpha_n'],highalpha_dark_n=bg['high_alpha_dark_n'],joint_static_occluded_n=bg['joint_static_occluded_n'],alpha_parent_max_change=alpha_diff,detail_json=relative(reportfile)))
   for source in ['C','S']:
    x0,y0,x1,y1=boxes[str(raw)][source]
    for mode,a in [('road',road),('joint',joint)]:
     if a is not None:cp=folder/f'{raw:03d}_{source}_{mode}_crop.png';saveimage(cp,a[y0:y1,x0:x1]);frame[source+'_'+mode+'_crop']=relative(cp)
  model['frames'][str(raw)]=frame;files[str(packetpath)]=dict(sha256=sha(packetpath),authority=item.get('authority'),authority_SHA256=item.get('authority_SHA256'));print(json.dumps(dict(evaluated=modelid,raw=raw,seconds=time.monotonic()-start)),flush=True)
 metrics=dict(schema='road_common1152_metrics.v1',resolution=[1152,768],rows=metricrows,blackgap_rows=blackrows,protocol='bench/protocol.json',source_sha256=files,human_verdict=None,problem_solved=None);write(bench/'metrics.json',metrics)
 ledger=path(inp.get('resource_ledger','resource_ledger.json'))
 if not ledger.exists():write(P/'resource_ledger.json',dict(schema='common1152_resource_ledger.v1',source='controller logs/status; no invented totals',cpu_evaluation_seconds=time.monotonic()-start))
 latest=dict(schema='road_common1152_latest.v1',teacher=dict(resolution=[1152,768],selection='legacy_T0_seed101',source_manifest='teacher/manifest.json',source_sha256={'teacher_manifest':sha(teacher_manifest)},frames=teacherframes),models=list(models.values()),crop_boxes=boxes,metrics='bench/metrics.json',protocol='bench/protocol.json',resource_ledger=relative(ledger if ledger.exists()else P/'resource_ledger.json'),source_export_manifest='export_manifest.json',warnings=[protocol['coverage_warning'],'C/S是冻结二维伪教师；三诊断帧继承BUILD，不是盲测。']);write(delivery/'latest_manifest.json',latest)
 wanted=[delivery,bench/'metrics.json',bench/'protocol.json',bench/'details',P/'teacher/manifest.json',P/'teacher/qualification_cpu.json',P/'cpu_inputs/targets_manifest.json',P/'cpu_inputs/source_footprint.json',P/'cpu_inputs/manifest.json',P/'common_permission/receipt.json',P/'native/receipt.json',manifest,ledger]
 target=P/'final_bundle.tar.gz';tmp=P/'final_bundle.tar.gz.tmp';added=set()
 with tarfile.open(tmp,'w:gz',dereference=True)as tar:
  for root in wanted:
   if not root.exists():continue
   paths=sorted(root.rglob('*'))if root.is_dir()else[root]
   for f in paths:
    if not f.is_file():continue
    rel=relative(f)
    if rel in added:continue
    info=tar.gettarinfo(str(f),arcname=rel)
    with f.open('rb')as stream:tar.addfile(info,stream)
    added.add(rel)
 tmp.replace(target)
 receipt=dict(schema='common1152_cpu_postprocess.v1',complete=True,bundle=str(target),bundle_sha256=sha(target),bundle_bytes=target.stat().st_size,models=len(models),metric_rows=len(metricrows),blackgap_rows=len(blackrows),seconds=time.monotonic()-start,manifest='delivery/latest_manifest.json',metrics='bench/metrics.json',CUDA_calls=0,large_model_files_included=False);write(bench/'completed.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=P/'export_manifest.json');ap.add_argument('--interface-smoke',action='store_true');a=ap.parse_args()
 try:
  if a.interface_smoke:
   d=load(P/'cpu_inputs/110_targets.npz');n=load(P/'native/parent/110.npz');a0=np.asarray(n['road_alpha']);p0=rgb(n['road_rgb']);effective=np.divide(p0,a0[...,None],out=np.zeros_like(p0),where=a0[...,None]>1e-8);context=d['S_core95'];ref=rgb(d['target_S']);r=evaluate_detail(effective,ref,context_mask=context,score_mask=context,resolution=ResolutionSpec(1152,768,'new_native1152'),tile_layout=TileLayout(768,1152),teacher_role='frozen_pseudo_teacher',config=DetailConfig(flat_hf_thresholds=(2/255,),edge_thresholds=(2/255,),edge_tolerances_pixels=(1.,),edge_angle_tolerances_degrees=(30.,),worst_tiles=0));bg=blackgap('CPU_interface',110,'S',d['S_source'],a0,effective,ref,n);write(P/'bench/interface_smoke.json',dict(interface_pass=True,actual_native_shape=list(p0.shape),core_sizes=[s['core_pixels']for s in r['scales']],blackgap=bg,CUDA_calls=0,formal_metric=False));print('CPU interface smoke passed')
  else:run(path(a.manifest))
 except Exception as e:
  write(P/'bench/failed.json',dict(error=str(e),traceback=traceback.format_exc(),phase='CPU_evaluation'));raise
