"""CPU evaluation of real adaptive coverage stages, fixed full-source denominators."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,hashlib,tarfile,time,traceback,shutil
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1');OLD=P.parent/'road_common1152_20261121'
sys.path.insert(0,str(OLD.parent/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics'))
from road_detail_metrics import DetailConfig,ResolutionSpec,TileLayout,evaluate_detail,_strict_core
STAGES=['parent','activated','dense_init','coverage','appearance'];NAMES={'parent':'覆盖修复前','activated':'激活已有锚点 · 覆盖200步','dense_init':'真实追加高斯 · 新点初值','coverage':'表面覆盖优化 · 累计600步','appearance':'教师外观拟合 · 1200步'}
BUDGET={'parent':'0更新','activated':'覆盖200步','dense_init':'继承覆盖200步＋真实新点初值','coverage':'覆盖累计600步','appearance':'覆盖600步＋外观1200步'}
CFG=DetailConfig(pixel_scales=(1.,2.,4.),flat_hf_thresholds=(2/255,),edge_thresholds=(2/255,),edge_tolerances_pixels=(1.,),edge_angle_tolerances_degrees=(30.,),worst_tiles=0)
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def path(p):p=Path(p);return p if p.is_absolute()else P/p
def rel(p):return Path(p).relative_to(P).as_posix()
def rgb(a):
 a=np.asarray(a);a=a.transpose(1,2,0)if a.shape==(3,768,1152)else a
 if a.shape!=(768,1152,3):raise ValueError('actual1152x768 native RGB required')
 return a.astype(float)/255 if a.dtype==np.uint8 else a.astype(float)
def eff(packet):
 a=packet['road_alpha'];p=rgb(packet['road_rgb']);return np.divide(p,a[...,None],out=np.zeros_like(p),where=a[...,None]>1e-8)
def save(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(p)
def bounds(m):
 y,x=np.nonzero(m)
 return [max(0,int(x.min())-8),max(0,int(y.min())-8),min(1152,int(x.max())+9),min(768,int(y.max())+9)]if len(x)else[0,0,128,128]
def stats(p,t,m):
 e=p[m]-t[m];return dict(MAE=float(abs(e).mean())if e.size else None,RMSE=float(np.sqrt((e*e).mean()))if e.size else None,pixels=int(m.sum()))
def detail(p,t,context,score,role):
 return evaluate_detail(p,t,context_mask=context,score_mask=score,resolution=ResolutionSpec(1152,768,'new_native1152','srgb'),tile_layout=TileLayout(768,1152),teacher_role='real_observation'if role=='A'else'frozen_pseudo_teacher',config=CFG,color_space='stable_alpha_unpremult_srgb',input_conditioning={'stable_alpha_min':.95,'source':role})
def main(manifest):
 started=time.monotonic();exp=json.loads(manifest.read_text());rows=exp['rows'];packetpaths={};metadata={};packets={}
 for row in rows:
  stage=row.get('stage',row['arm']);raw=int(row['raw'])
  if stage not in STAGES:continue
  packetpaths[stage,raw]=path(row['packet']);metadata[stage,raw]=row;packets[stage,raw]=load(packetpaths[stage,raw])
 if any((s,r)not in packets for s in STAGES for r in [86,110,134,179,198]):raise ValueError('all five actual stages by five views required')
 bench=P/'bench';delivery=P/'delivery';bench.mkdir(exist_ok=True);delivery.mkdir(exist_ok=True);teacherframes={};cropboxes={};domains={};refs={}
 for raw in [86,110]:
  domains[raw]=load(P/f'proposal/domains/{raw:03d}.npz');t=load(OLD/f'teacher/raw{raw:03d}/supervision.npz');refs[raw]=t;teacherframes[str(raw)]={};cropboxes[str(raw)]={}
  for source in ['A','C','S']:
   mask=domains[raw]['appearance_'+source];cropboxes[str(raw)][source]=bounds(mask);value=rgb(t['target_'+source+'_srgb']);dest=delivery/f'teacher/{raw:03d}_{source}.png';save(dest,value);teacherframes[str(raw)][source]=rel(dest);x0,y0,x1,y1=cropboxes[str(raw)][source];dest=dest.with_name(f'{raw:03d}_{source}_crop.png');save(dest,value[y0:y1,x0:x1]);teacherframes[str(raw)][source+'_crop']=rel(dest)
 models=[];metricrows=[];gaprows=[];newrows=[];protection=[];commoncounts=[];sourceSHA={}
 protocol=dict(schema='adaptive_surface_bench_protocol.v1',resolution=[1152,768],full_source='C/S complete legal2D editing denominators retained, including uncertified/no-hit; uncertified editing domain is not confirmed road or a claim that densification can repair it; A only confirmed road',coverage_thresholds=dict(no_support=.05,stable=.95),full_premult_RGB_MAE='abs(actual road premult P - opaque teacher T) averaged on full fixed2D editing source, no division by alpha',primary_color_HF='fixed common stable: parent and final >=.95, plus every intermediate >=.95; same domain for every stage',common_stable_partition='parent>=.95 and final>=.95; transient alpha loss separately recorded',newly_covered_partition='parent<.95 and final>=.95; final-only RGB/HF with full final-stable source as filter context',HF_sigma_native_px=[1,2,4],gaussian_truncate=3,edge='fixed RGB tensor2/255 NMS and1px match; appearance, not lane geometry truth',flat='fixed teacher HF RMS<=2/255',dark='highalpha>=.95 and unpremult meanRGB<.05 and teacher meanRGB>=.10, descriptive only',joint_occlusion='road alpha is not final alphaT; absent actual contribution export => null',surface_subset='fixed old supported mesh plus independently recertified original measured plane; no surface evidence remains unconfirmed; certified-road gaps separate from all2D-source gaps',teacher_manifest_SHA256=sha(OLD/'teacher/manifest.json'),proposal_manifest_SHA256=sha(P/'proposal/manifest.json'))
 write(bench/'protocol.json',protocol)
 for stage in STAGES:
  m=dict(id=stage,round='4',name=NAMES[stage],budget=BUDGET[stage],frames={},point_counts={},renderer='actual1152_native3DGS',data_role='new fixed teacher fitting with explicit coverage changes')
  for raw in [86,110,134,179,198]:
   q=packets[stage,raw];meta=metadata[stage,raw];base=packets['parent',raw];road=rgb(q['road_rgb']);joint=rgb(q['joint_rgb']);alpha=q['road_alpha'];effective=eff(q);frame={};folder=delivery/'models'/stage
   counts=meta.get('point_counts',{});m['point_counts']={key:counts.get(old,counts.get(key,0))for key,old in [('total','total_rows'),('active','active_rows'),('activated_existing','activate_existing'),('appended_new_geometry','append_new_geometry')]}
   for mode,value in [('road',road),('joint',joint),('alpha',alpha)]:
    dest=folder/f'{raw:03d}_{mode}.png';save(dest,value);frame[mode]=rel(dest)
   for source in ['A','C','S']if raw in domains else[]:
    x0,y0,x1,y1=cropboxes[str(raw)][source]
    for mode,value in [('road',road),('joint',joint),('alpha',alpha)]:
     dest=folder/f'{raw:03d}_{source}_{mode}_crop.png';save(dest,value[y0:y1,x0:x1]);frame[source+'_'+mode+'_crop']=rel(dest)
   d=domains.get(raw,load(P/f'proposal/domains/{raw:03d}.npz'));K=base['K'];krgb=float(np.abs(joint-rgb(base['joint_rgb']))[K].max(initial=0));kalpha=float(np.abs(q['joint_alpha']-base['joint_alpha'])[K].max(initial=0))if'joint_alpha'in q and'joint_alpha'in base else None;outside=d['outside'];retA=d.get('retained_confirmed_A',np.zeros(K.shape,bool));protection.append(dict(id=stage,raw=raw,K_pixels=int(K.sum()),joint_K_RGB_max=krgb,joint_K_alpha_max=kalpha,outside_pixels=int(outside.sum()),outside_joint_RGB_MAE=stats(joint,rgb(base['joint_rgb']),outside)['MAE'],outside_joint_RGB_max=float(np.abs(joint-rgb(base['joint_rgb']))[outside].max(initial=0)),confirmed_A_parent_RGB_MAE=stats(joint,rgb(base['joint_rgb']),retA)['MAE']))
   if raw in domains:
    parenta=base['road_alpha'];finala=packets['appearance',raw]['road_alpha'];allstable=np.logical_and.reduce([packets[s,raw]['road_alpha']>=.95 for s in STAGES])
    for source in ['A','C','S']:
     domain=d['appearance_'+source];teacher=rgb(refs[raw]['target_'+source+'_srgb']);common=domain&(parenta>=.95)&(finala>=.95);context=common&allstable;new=domain&(parenta<.95)&(finala>=.95);source_n=int(domain.sum());no=domain&(alpha<.05);thin=domain&(alpha>=.05)&(alpha<.95);dark=domain&(alpha>=.95)&(effective.mean(-1)<.05)&(teacher.mean(-1)>=.10)
     cert=domain&d['canonical_valid'];certn=int(cert.sum());bg=dict(id=stage,raw=raw,source=source,denominator=source_n,no_support_n=int(no.sum()),low_alpha_n=int(thin.sum()),high_alpha_dark_n=int(dark.sum()),no_support_fraction=float(no.sum()/source_n)if source_n else None,low_alpha_fraction=float(thin.sum()/source_n)if source_n else None,high_alpha_dark_fraction=float(dark.sum()/source_n)if source_n else None,joint_static_occluded_n=None,joint_static_occluded_fraction=None,canonical_missing_n=int((domain&~d['canonical_valid']).sum()),certified_recovered_rayhit_n=int((domain&d['certified_recovered_rayhit']).sum()),rayhit_repairable_n=certn,certified_road_n=certn,uncertified_editing_n=source_n-certn,certified_road_no_support_n=int((cert&(alpha<.05)).sum()),certified_road_low_alpha_n=int((cert&(alpha>=.05)&(alpha<.95)).sum()),certified_road_low_alpha_total_n=int((cert&(alpha<.95)).sum()),certified_road_no_support_fraction=float((cert&(alpha<.05)).sum()/certn)if certn else None,certified_road_low_alpha_total_fraction=float((cert&(alpha<.95)).sum()/certn)if certn else None,common_stable_n=int(common.sum()),newly_covered_n=int(new.sum()),uncertified_editing_is_confirmed_road=False);gaprows.append(bg)
     report=detail(effective,teacher,context,context,source);detailfile=bench/'details'/f'{stage}_{raw:03d}_{source}.json';write(detailfile,report);curve=report['edges']['threshold_curves'][0]['curves'][0];prec,rec=curve['precision'],curve['recall'];edgeF=None if prec is None or rec is None else(2*prec*rec/(prec+rec)if prec+rec else 0.);full=stats(road,teacher,domain);ray=domain&d['canonical_valid'];fullray=stats(road,teacher,ray)
     for sc in report['scales']:
      sigma=sc['sigma_pixels'];core=context&_strict_core(context,int(3*sigma+.5));st=stats(effective,teacher,core);metricrows.append(dict(id=stage,round='4',raw=raw,source=source,scale=sigma,RGB_MAE=st['MAE'],RGB_RMSE=st['RMSE'],full_premult_RGB_MAE=full['MAE'],full_premult_RGB_RMSE=full['RMSE'],full_source_n=source_n,rayhit_full_premult_RGB_MAE=fullray['MAE'],HF_cosine=sc['core']['HF_cosine'],HF_energy_ratio=sc['core']['HF_energy_ratio'],HF_MAE=sc['core']['HF_MAE'],HF_RMSE=sc['core']['HF_RMSE'],edge_F1=edgeF,flat_pred_RMS=sc['teacher_flat_spurious'][0]['predicted_HF_RMS'],core_n=sc['core_pixels'],stable_score_n=int(context.sum()),common_stable_n=int(common.sum()),newly_covered_n=int(new.sum()),current_stage_stable_n=int((domain&(alpha>=.95)).sum()),blackgap_no_support_n=bg['no_support_n'],blackgap_thin_n=bg['low_alpha_n'],highalpha_dark_n=bg['high_alpha_dark_n'],joint_static_occluded_n=None,point_counts=m['point_counts'],detail_json=rel(detailfile)))
     if stage=='appearance':
      nr=detail(effective,teacher,domain&(alpha>=.95),new,source);write(bench/'details'/f'newly_covered_{raw:03d}_{source}.json',nr)
      for sc in nr['scales']:
       core=new&_strict_core(domain&(alpha>=.95),int(3*sc['sigma_pixels']+.5));nst=stats(effective,teacher,core);newrows.append(dict(id=stage,raw=raw,source=source,scale=sc['sigma_pixels'],newly_covered_n=int(new.sum()),newly_covered_certified_n=int((new&d['canonical_valid']).sum()),newly_covered_uncertified_editing_n=int((new&~d['canonical_valid']).sum()),core_n=sc['core_pixels'],RGB_MAE=nst['MAE'],RGB_RMSE=nst['RMSE'],HF_cosine=sc['core']['HF_cosine'],HF_energy_ratio=sc['core']['HF_energy_ratio'],HF_RMSE=sc['core']['HF_RMSE']))
   m['frames'][str(raw)]=frame;sourceSHA[rel(packetpaths[stage,raw])]=sha(packetpaths[stage,raw]);print(json.dumps(dict(stage=stage,raw=raw,seconds=time.monotonic()-started)),flush=True)
  models.append(m)
 metrics=dict(schema='road_common1152_metrics.v1',resolution=[1152,768],rows=metricrows,blackgap_rows=gaprows,newly_covered_rows=newrows,protection_rows=protection,source_sha256=sourceSHA,protocol='bench/protocol.json',gap_note='完整合法C/S分母保留。颜色/HF主表使用所有阶段共同稳定域；新增覆盖另表。无测距支持区域不编造路面。',human_verdict=None,problem_solved=None);write(bench/'metrics.json',metrics)
 geometry={}
 for raw in [86,110]:
  source=P/f'proposal/geometry_support_{raw:03d}.png'
  if source.exists():dest=delivery/f'geometry_support/{raw:03d}.png';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);geometry[str(raw)]=rel(dest)
 latest=dict(schema='road_common1152_latest.v1',teacher=dict(resolution=[1152,768],selection='legacy_T0_seed101',source_manifest='teacher_manifest.json',source_sha256={'teacher_manifest':sha(OLD/'teacher/manifest.json')},frames=teacherframes),models=models,representatives=['parent','coverage','appearance'],crop_boxes=cropboxes,geometry_support=geometry,geometry_support_legend='proposal/geometry_support_legend.json',metrics='bench/metrics.json',protocol='bench/protocol.json',resource_ledger='resource_ledger.json',point_count_semantics='activated_existing reuses parent rows; appended_new_geometry extends row count; historical892 not new',warnings=[protocol['joint_occlusion'],protocol['full_source']]);write(delivery/'latest_manifest.json',latest)
 (P/'teacher_manifest.json').write_bytes((OLD/'teacher/manifest.json').read_bytes())
 if not(P/'resource_ledger.json').exists():write(P/'resource_ledger.json',dict(schema='adaptive_resource_ledger.v1',cpu_evaluation_seconds=time.monotonic()-started,native_training_ledger='controller receipt and per-stage receipts',no_invented_training_totals=True))
 wanted=[delivery,bench/'metrics.json',bench/'protocol.json',bench/'details',P/'teacher_manifest.json',P/'resource_ledger.json',manifest]
 # Include lightweight contracts, source hashes, K/reload proofs; never include model checkpoints.
 wanted.extend(f for f in P.rglob('*.json')if f.stat().st_size<2_000_000 and not any(x in f.parts for x in ['delivery','details'])and f.name not in ['status.json','completed.json'])
 for name in ['activation.npz','append.npz','surface_certificate.npz']:
  f=P/'proposal'/name
  if f.exists()and f.stat().st_size<4_000_000:wanted.append(f)
 target=P/'final_bundle.tar.gz';tmp=P/'final_bundle.tar.gz.tmp';added=set()
 with tarfile.open(tmp,'w:gz',dereference=True)as tar:
  for f in wanted:
   for item in sorted(f.rglob('*'))if f.is_dir()else[f]:
    if not item.is_file()or item==target or item==tmp:continue
    rp=rel(item)
    if rp in added:continue
    info=tar.gettarinfo(str(item),arcname=rp)
    with item.open('rb')as stream:tar.addfile(info,stream)
    added.add(rp)
 tmp.replace(target);receipt=dict(schema='adaptive_cpu_postprocess.v1',complete=True,bundle=str(target),bundle_sha256=sha(target),bundle_bytes=target.stat().st_size,manifest='delivery/latest_manifest.json',metrics='bench/metrics.json',models=len(models),metric_rows=len(metricrows),CUDA_calls=0,seconds=time.monotonic()-started);write(bench/'completed.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=P/'export_manifest.json');ap.add_argument('--interface-smoke',action='store_true');a=ap.parse_args()
 try:
  if a.interface_smoke:
   q=load(OLD/'exports/r3/G_NUMERIC/110.npz');d=load(P/'proposal/domains/110.npz');t=load(OLD/'teacher/raw110/supervision.npz');p=eff(q);reference=rgb(t['target_C_srgb']);domain=d['appearance_C'];context=domain&(q['road_alpha']>=.95);r=detail(p,reference,context,context,'C');full=stats(rgb(q['road_rgb']),reference,domain);write(P/'bench/interface_smoke.json',dict(passed=True,python=sys.executable,actual_native_shape=list(rgb(q['road_rgb']).shape),full_source_n=int(domain.sum()),core_sizes=[s['core_pixels']for s in r['scales']],full_premult_RGB_MAE=full['MAE'],CUDA_calls=0,formal_result=False));print('adaptive CPU interface smoke passed')
  else:main(path(a.manifest))
 except Exception as e:write(P/'bench/failed.json',dict(error=str(e),traceback=traceback.format_exc()));raise
