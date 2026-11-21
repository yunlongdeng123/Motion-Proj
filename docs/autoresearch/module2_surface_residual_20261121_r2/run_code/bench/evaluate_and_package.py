"""Independent R2 CPU evaluation against the immediate first-wave appearance parent."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,hashlib,tarfile,time,traceback,shutil
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2');OLD=P.parent/'road_common1152_20261121';FIRST=P.parent/'road_surface_adaptive_20261121_r1'
sys.path.insert(0,str(OLD.parent/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics'))
from road_detail_metrics import DetailConfig,ResolutionSpec,TileLayout,evaluate_detail,_strict_core
STAGES=['parent','activated','dense_init','coverage','appearance'];NAMES={'parent':'第二轮起点 · 第一轮外观终态','activated':'启用13个既有锚点 · 未训练','dense_init':'安全残余高斯初值 · 无同中心复制','coverage':'覆盖优先优化 · 400步','appearance':'本轮外观拟合 · 1200步'}
BUDGET={'parent':'直接父基线 · 0更新','activated':'0覆盖更新','dense_init':'0覆盖更新','coverage':'覆盖400步','appearance':'覆盖400步＋外观1200步'}
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
 write(bench/'code_manifest.json',dict(schema='r2_CPU_measurement_sources.v1',source_sha256={str(f):sha(f)for f in [Path(__file__),Path(__file__).with_name('parent_freeze_proof.py'),OLD.parent/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics/road_detail_metrics.py']}))
 for raw in [86,110]:
  domains[raw]=load(P/f'proposal/domains/{raw:03d}.npz');t=load(OLD/f'teacher/raw{raw:03d}/supervision.npz');refs[raw]=t;teacherframes[str(raw)]={};cropboxes[str(raw)]={}
  for source in ['A','C','S']:
   mask=domains[raw]['appearance_'+source];cropboxes[str(raw)][source]=bounds(mask);value=rgb(t['target_'+source+'_srgb']);dest=delivery/f'teacher/{raw:03d}_{source}.png';save(dest,value);teacherframes[str(raw)][source]=rel(dest);x0,y0,x1,y1=cropboxes[str(raw)][source];dest=dest.with_name(f'{raw:03d}_{source}_crop.png');save(dest,value[y0:y1,x0:x1]);teacherframes[str(raw)][source+'_crop']=rel(dest)
 models=[];metricrows=[];gaprows=[];newrows=[];protection=[];commoncounts=[];sourceSHA={}
 protocol=dict(schema='r2_residual_bench_protocol.v1',resolution=[1152,768],direct_parent='firstwave appearance asset,595029 total/161264 active; every primary comparison is relative to this direct parent',method='residual ray sampling plus actual joint K gate and coverage-priority hinge relu(.99-alpha)^2 with no competing coverage prior, then appearance1200; a combined method, not a density-only ablation',parent_mutability='only new activation/append and validated R1-newgroup reuse; no coincident Gaussian copy; other parent colors/geometry/coverage frozen',full_source='unchanged complete2D editing C/S; uncertified domain retained but not called confirmed road; A only confirmed road',coverage_thresholds=dict(no_support=.05,stable=.95),full_premult_RGB_MAE='abs(actual road premult P - opaque teacher T) on fixed complete2D source, no alpha division',primary_color_HF='fixed direct-parent/final stable intersection, additionally stable in every intermediate R2 stage; same scored domain per R2 stage',common_stable_partition='direct parent>=.95 and final>=.95; transient loss recorded',newly_covered_partition='direct parent<.95 and final>=.95, final-only metrics',HF_sigma_native_px=[1,2,4],gaussian_truncate=3,edge='fixed RGB tensor2/255 NMS and1px match, appearance only',flat='teacher HF RMS<=2/255',dark='alpha>=.95, effective meanRGB<.05, teacher meanRGB>=.10',joint_occlusion='standalone alpha is not scene contribution; missing alphaT remains null',surface_subset='same immutable R1 canonical and original-measurement certified surface; no geometric threshold expansion',cumulative_comparison='optional pre-R1 summaries are separate labeled full-domain values, never mixed into R2 direct-parent HF table',teacher_manifest_SHA256=sha(OLD/'teacher/manifest.json'),proposal_manifest_SHA256=sha(P/'proposal/manifest.json'))
 write(bench/'protocol.json',protocol)
 for stage in STAGES:
  m=dict(id=stage,round='5',name=NAMES[stage],budget=BUDGET[stage],frames={},point_counts={},renderer='actual1152_native3DGS',data_role='R2 residual and hard-priority coverage method; immediate firstwave appearance parent',asset_dir=metadata[stage,86].get('asset_dir',str(FIRST/'appearance_fit/asset')if stage=='parent'else''))
  for raw in [86,110,134,179,198]:
   q=packets[stage,raw];meta=metadata[stage,raw];base=packets['parent',raw];road=rgb(q['road_rgb']);joint=rgb(q['joint_rgb']);alpha=q['road_alpha'];effective=eff(q);frame={};folder=delivery/'models'/stage
   counts=meta.get('point_counts',{});m['point_counts']={key:counts.get(key,counts.get(old,0))for key,old in [('total','total_rows'),('active','active_rows'),('activated_existing','activate_existing'),('appended_new_geometry','append_new_geometry')]};m['point_counts']['reuse_existing_newgroup']=counts.get('reuse_existing_newgroup',counts.get('reuse_existing',0));m['point_counts']['opacity_stack_appended']=0
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
      sigma=sc['sigma_pixels'];core=context&_strict_core(context,int(3*sigma+.5));st=stats(effective,teacher,core);metricrows.append(dict(id=stage,round='5',raw=raw,source=source,scale=sigma,RGB_MAE=st['MAE'],RGB_RMSE=st['RMSE'],full_premult_RGB_MAE=full['MAE'],full_premult_RGB_RMSE=full['RMSE'],full_source_n=source_n,rayhit_full_premult_RGB_MAE=fullray['MAE'],HF_cosine=sc['core']['HF_cosine'],HF_energy_ratio=sc['core']['HF_energy_ratio'],HF_MAE=sc['core']['HF_MAE'],HF_RMSE=sc['core']['HF_RMSE'],edge_F1=edgeF,flat_pred_RMS=sc['teacher_flat_spurious'][0]['predicted_HF_RMS'],core_n=sc['core_pixels'],stable_score_n=int(context.sum()),common_stable_n=int(common.sum()),newly_covered_n=int(new.sum()),current_stage_stable_n=int((domain&(alpha>=.95)).sum()),blackgap_no_support_n=bg['no_support_n'],blackgap_thin_n=bg['low_alpha_n'],highalpha_dark_n=bg['high_alpha_dark_n'],joint_static_occluded_n=None,point_counts=m['point_counts'],detail_json=rel(detailfile)))
     if stage=='appearance':
      nr=detail(effective,teacher,domain&(alpha>=.95),new,source);write(bench/'details'/f'newly_covered_{raw:03d}_{source}.json',nr)
      for sc in nr['scales']:
       core=new&_strict_core(domain&(alpha>=.95),int(3*sc['sigma_pixels']+.5));nst=stats(effective,teacher,core);newrows.append(dict(id=stage,raw=raw,source=source,scale=sc['sigma_pixels'],newly_covered_n=int(new.sum()),newly_covered_certified_n=int((new&d['canonical_valid']).sum()),newly_covered_uncertified_editing_n=int((new&~d['canonical_valid']).sum()),core_n=sc['core_pixels'],RGB_MAE=nst['MAE'],RGB_RMSE=nst['RMSE'],HF_cosine=sc['core']['HF_cosine'],HF_energy_ratio=sc['core']['HF_energy_ratio'],HF_RMSE=sc['core']['HF_RMSE']))
   m['frames'][str(raw)]=frame;sourceSHA[rel(packetpaths[stage,raw])]=sha(packetpaths[stage,raw]);print(json.dumps(dict(stage=stage,raw=raw,seconds=time.monotonic()-started)),flush=True)
  models.append(m)
 from parent_freeze_proof import proof
 freeze=proof(models);write(bench/'parent_freeze_proof.json',freeze)
 gate_audit=json.loads((P/'proposal/residual_gate_audit.json').read_text());gate_summary=[]
 import csv
 with(P/'proposal/residual_pixel_K_table.csv').open()as f:gate_pixels=list(csv.DictReader(f))
 candidate_data=load(P/'proposal/candidates.npz');candidate_position={int(x):i for i,x in enumerate(candidate_data['candidate_id'])};actual_reuse=set(freeze['actual_reuse_world_rows'])
 for row in gate_pixels:
  raw=int(row['raw']);x,y=int(row['x']),int(row['y']);a=float(packets['appearance',raw]['road_alpha'][y,x]);row['final_alpha']=a;row['final_below095']=a<.95
  index=candidate_position[int(row['candidate_id'])];reuse_row=int(candidate_data['parent_active_F32_model_center_world_row'][index]);row['actually_appended']=row['status']=='safe_spatially_new';row['actual_reused_parent_world_row']=reuse_row if row['status']=='safe_same_center_stack'and reuse_row in actual_reuse else -1;row['execution_action']='reuse_parent_R1_newgroup_parameters'if row['actual_reused_parent_world_row']>=0 else({'safe_existing_inactive':'activate_existing_inactive','safe_spatially_new':'append_new_position','safe_same_center_stack':'diagnostic_only_no_append','rejected_joint_K':'exclude_joint_K'}[row['status']])
 for raw in [86,110]:
  for source in ['C','S']:
   subset=[r for r in gate_pixels if int(r['raw'])==raw and r['source']==source];gate_summary.append(dict(raw=raw,source=source,fixed_residual_pixels=len(subset),final_below095_n=sum(r['final_below095']for r in subset),status_rows=[dict(status=s,pixels=sum(r['status']==s for r in subset),final_below095_n=sum(r['status']==s and r['final_below095']for r in subset))for s in sorted({r['status']for r in subset})],execution_action_rows=[dict(action=s,pixels=sum(r['execution_action']==s for r in subset),final_below095_n=sum(r['execution_action']==s and r['final_below095']for r in subset))for s in sorted({r['execution_action']for r in subset})]))
 with(bench/'residual_pixel_K_table.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(gate_pixels[0]));w.writeheader();w.writerows(gate_pixels)
 cumulative=[];firstmetrics=json.loads((FIRST/'bench/metrics.json').read_text())
 for raw in [86,110]:
  for source in ['C','S']:
   r0=next(r for r in firstmetrics['rows']if r['id']=='parent'and r['raw']==raw and r['source']==source and r['scale']==1);r1=next(r for r in firstmetrics['rows']if r['id']=='appearance'and r['raw']==raw and r['source']==source and r['scale']==1);r2=next(r for r in metricrows if r['id']=='appearance'and r['raw']==raw and r['source']==source and r['scale']==1);assert r0['full_source_n']==r2['full_source_n'];cumulative.append(dict(raw=raw,source=source,full_source_n=r2['full_source_n'],pre_R1_full_premult_RGB_MAE=r0['full_premult_RGB_MAE'],R1_appearance_full_premult_RGB_MAE=r1['full_premult_RGB_MAE'],R2_appearance_full_premult_RGB_MAE=r2['full_premult_RGB_MAE'],comparison='separate cumulative full-domain summary; not the direct-parent HF comparison'))
 metrics=dict(schema='road_common1152_metrics.v1',comparison='R2 vs immediate R1 appearance parent',resolution=[1152,768],rows=metricrows,blackgap_rows=gaprows,newly_covered_rows=newrows,protection_rows=protection,parent_parameter_proof='bench/parent_freeze_proof.json',residual_joint_K_rows=gate_summary,residual_joint_K_pixel_table='bench/residual_pixel_K_table.csv',cumulative_vs_pre_R1_rows=cumulative,source_sha256=sourceSHA,protocol='bench/protocol.json',gap_note='完整2D编辑域分母保留；主比较直接父为R1外观终态。共同稳定域按R2各阶段重算，累计R0仅单列。认证路面与未认证编辑域分开。',human_verdict=None,problem_solved=None);write(bench/'metrics.json',metrics)
 geometry={}
 for raw in [86,110]:
  source=FIRST/f'proposal/geometry_support_{raw:03d}.png'
  if source.exists():dest=delivery/f'geometry_support/{raw:03d}.png';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);geometry[str(raw)]=rel(dest)
 shutil.copyfile(FIRST/'proposal/geometry_support_legend.json',P/'proposal/geometry_support_legend.json')
 latest=dict(schema='road_common1152_latest.v1',comparison_parent='R1 appearance direct parent595029/161264',teacher=dict(resolution=[1152,768],selection='legacy_T0_seed101',source_manifest='teacher_manifest.json',source_sha256={'teacher_manifest':sha(OLD/'teacher/manifest.json')},frames=teacherframes),models=models,representatives=['parent','coverage','appearance'],crop_boxes=cropboxes,geometry_support=geometry,geometry_support_legend='proposal/geometry_support_legend.json',metrics='bench/metrics.json',protocol='bench/protocol.json',resource_ledger='resource_ledger.json',point_count_semantics='13activation+1071 new spatial rows; qualified reuse changes existing parent newgroup params without incrementing N, coincident copies never appended',warnings=[protocol['joint_occlusion'],protocol['full_source'],protocol['method']]);write(delivery/'latest_manifest.json',latest)
 (P/'teacher_manifest.json').write_bytes((OLD/'teacher/manifest.json').read_bytes())
 if not(P/'resource_ledger.json').exists():write(P/'resource_ledger.json',dict(schema='adaptive_resource_ledger.v1',cpu_evaluation_seconds=time.monotonic()-started,native_training_ledger='controller receipt and per-stage receipts',no_invented_training_totals=True))
 wanted=[delivery,bench/'metrics.json',bench/'protocol.json',bench/'parent_freeze_proof.json',bench/'residual_pixel_K_table.csv',bench/'details',P/'proposal/residual_pixel_K_table.csv',P/'teacher_manifest.json',P/'resource_ledger.json',manifest]
 # Include lightweight contracts, source hashes, K/reload proofs; never include model checkpoints.
 wanted.extend(f for f in P.rglob('*.json')if f.stat().st_size<2_000_000 and not any(x in f.parts for x in ['delivery','details'])and f.name not in ['status.json','completed.json'])
 for name in ['safe_candidates.npz','per_candidate_K_evidence.npz','safe_stack_diagnostic.npz']:
  f=P/'k_gate'/name
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
 tmp.replace(target);receipt=dict(schema='residual_r2_cpu_postprocess.v1',complete=True,bundle=str(target),bundle_sha256=sha(target),bundle_bytes=target.stat().st_size,manifest='delivery/latest_manifest.json',metrics='bench/metrics.json',models=len(models),metric_rows=len(metricrows),parent_parameter_proof_passed=True,CUDA_calls=0,seconds=time.monotonic()-started);write(bench/'completed.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=P/'export_manifest.json');ap.add_argument('--interface-smoke',action='store_true');a=ap.parse_args()
 try:
  if a.interface_smoke:
   q=load(FIRST/'appearance_fit/rendered/110.npz');d=load(P/'proposal/domains/110.npz');t=load(OLD/'teacher/raw110/supervision.npz');p=eff(q);reference=rgb(t['target_C_srgb']);domain=d['appearance_C'];context=domain&(q['road_alpha']>=.95);r=detail(p,reference,context,context,'C');full=stats(rgb(q['road_rgb']),reference,domain);write(P/'bench/interface_smoke.json',dict(passed=True,python=sys.executable,actual_native_shape=list(rgb(q['road_rgb']).shape),direct_parent=str(FIRST/'appearance_fit/asset'),full_source_n=int(domain.sum()),core_sizes=[s['core_pixels']for s in r['scales']],full_premult_RGB_MAE=full['MAE'],CUDA_calls=0,formal_result=False));print('R2 CPU interface smoke passed')
  else:main(path(a.manifest))
 except Exception as e:write(P/'bench/failed.json',dict(error=str(e),traceback=traceback.format_exc()));raise
