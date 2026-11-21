"""Once-only CPU summary and small independent packet for fixed-CI bounded solver diagnostic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,time,hashlib,tarfile,traceback
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2');D=P/'diagnostic/color_optimality';OLD=P.parent/'road_common1152_20261121'
METRIC=OLD.parent/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1/metrics'
sys.path.insert(0,str(METRIC));from road_detail_metrics import DetailConfig,ResolutionSpec,TileLayout,evaluate_detail
CFG=DetailConfig(pixel_scales=(1.,2.,4.),flat_hf_thresholds=(2/255,),edge_thresholds=(2/255,),edge_tolerances_pixels=(1.,),edge_angle_tolerances_degrees=(30.,),worst_tiles=0)
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rgb(a):return a.transpose(1,2,0).astype(float)if a.shape==(3,768,1152)else a.astype(float)
def effective(q):
 a=q['road_alpha'];p=rgb(q['road_rgb']);return np.divide(p,a[...,None],out=np.zeros_like(p),where=a[...,None]>1e-8)
def core(m,s):return minimum_filter(m.astype(np.uint8),size=2*int(3*s+.5)+1,mode='constant',cval=0).astype(bool)
def stats(p,t,m):
 x=p[m]-t[m];return dict(n=int(m.sum()),MAE=float(abs(x).mean())if x.size else None,MSE=float((x*x).mean())if x.size else None,RMSE=float(np.sqrt((x*x).mean()))if x.size else None)
def detail_cached(packetpath,raw,source,label,p,t,context,score,teacherpath):
 path=D/f'cpu_evaluation/details/numeric_{raw:03d}_{source}_{label}.json';binding=dict(packet_SHA256=sha(packetpath),teacher_SHA256=sha(teacherpath),mask_SHA256=hashlib.sha256(context.tobytes()+score.tobytes()).hexdigest(),metric_source_SHA256=sha(METRIC/'road_detail_metrics.py'))
 if path.exists():
  d=json.loads(path.read_text())
  if d.get('input_binding')!=binding:raise ValueError('cached HF input changed; explicit new output needed')
  return d['report']
 r=evaluate_detail(p,t,context_mask=context,score_mask=score,resolution=ResolutionSpec(1152,768,'new_native1152','srgb'),tile_layout=TileLayout(768,1152),teacher_role='real_observation'if source=='A'else'frozen_pseudo_teacher',config=CFG,color_space='stable_alpha_unpremult_srgb',input_conditioning={'comparison':'fixed_R2_Adam_vs_numeric','source':source})
 write(path,dict(input_binding=binding,report=r));return r
def main(manifest):
 start=time.monotonic();exp=json.loads(manifest.read_text());receipt=json.loads((D/'numeric_receipt.json').read_text());done=D/'cpu_evaluation/completed.json'
 binding=dict(export_SHA256=sha(manifest),numeric_receipt_SHA256=sha(D/'numeric_receipt.json'))
 if done.exists():
  old=json.loads(done.read_text())
  if old.get('input_binding')==binding:print(json.dumps(old));return
  raise ValueError('prior CPU evaluation has different inputs')
 parentmetrics=json.loads((P/'bench/metrics.json').read_text());inclusion=json.loads((D/'mask_inclusion_audit.json').read_text());images={};mechanical=[];rows=[];newrows=[];partitions=[];numeric_paths={int(r['raw']):Path(r['packet'])for r in exp['rows']};allowed=load(P/'train_inputs/allowed_color_world_rows.npz')['world_row_ids']
 oldasset=load(P/'appearance_fit/asset/road.npz');newasset=load(D/'new_numeric_asset/road.npz');frozen=np.ones(len(oldasset['active']),bool);frozen[allowed]=False;proof={}
 for k in ['active','model_xyz','normal_world','gaussian_sigma_m','road_opacity','minimum_model_depth','model_from_world']:
  proof[k]=bool(np.array_equal(oldasset[k],newasset[k]))
  if not proof[k]:raise ValueError('numeric changed fixed geometry/coverage:'+k)
 proof['all_other_RGB_exact']=bool(np.array_equal(oldasset['reference_srgb'][frozen],newasset['reference_srgb'][frozen]));assert proof['all_other_RGB_exact'];assert np.all((newasset['reference_srgb']>=0)&(newasset['reference_srgb']<=1))
 for raw in [86,110,134,179,198]:
  qp=numeric_paths[raw];qp=qp if qp.is_absolute()else D/qp;num=load(qp);adam=load(P/f'appearance_fit/rendered/{raw:03d}.npz');assert np.array_equal(num['road_alpha'],adam['road_alpha']);K=adam['K'];kd=float(abs(rgb(num['joint_rgb'])-rgb(adam['joint_rgb']))[K].max(initial=0));ka=float(abs(num['joint_alpha']-adam['joint_alpha'])[K].max(initial=0));assert kd==0 and ka==0;mechanical.append(dict(raw=raw,alpha_exact_Adam=True,joint_K_RGB_max=kd,joint_K_alpha_max=ka,RGB_float_max_change=float(abs(rgb(num['road_rgb'])-rgb(adam['road_rgb'])).max())))
  frame={}
  erow=next(r for r in exp['rows']if int(r['raw'])==raw)
  for mode in ['road','joint']:
   p=Path(erow[mode+'_png']);p=p if p.is_absolute()else D/p
   with Image.open(p)as im:assert im.size==(1152,768)
   frame[mode]=p.relative_to(D).as_posix()
  images[str(raw)]=frame
  if raw not in [86,110]:continue
  masks=load(D/f'cpu_masks/{raw:03d}.npz');domains=load(P/f'proposal/domains/{raw:03d}.npz');teacherpath=OLD/f'teacher/raw{raw:03d}/supervision.npz';teacher=load(teacherpath);ne=effective(num);ae=effective(adam);alpha=adam['road_alpha'];nprem=rgb(num['road_rgb']);aprem=rgb(adam['road_rgb'])
  for source in ['A','C','S']:
   target=rgb(teacher['target_'+source+'_srgb']);context=masks['common_eval_context_'+source];report=detail_cached(qp,raw,source,'same_R2_common_stable',ne,target,context,context,teacherpath)
   for sc in report['scales']:
    s=sc['sigma_pixels'];old=next(r for r in parentmetrics['rows']if r['id']=='appearance'and r['raw']==raw and r['source']==source and r['scale']==s);assert sc['core_pixels']==old['core_n'];cs=context&core(context,s);numRGB=stats(ne,target,cs);rows.append(dict(raw=raw,source=source,scale=s,partition='same_R2_common_stable',stable_score_n=int(context.sum()),core_n=sc['core_pixels'],Adam_RGB_MAE=old['RGB_MAE'],numeric_RGB_MAE=numRGB['MAE'],Adam_HF_cosine=old['HF_cosine'],numeric_HF_cosine=sc['core']['HF_cosine'],Adam_HF_energy_ratio=old['HF_energy_ratio'],numeric_HF_energy_ratio=sc['core']['HF_energy_ratio'],Adam_HF_RMSE=old['HF_RMSE'],numeric_HF_RMSE=sc['core']['HF_RMSE'],Adam_flat_pred_RMS=old['flat_pred_RMS'],numeric_flat_pred_RMS=sc['teacher_flat_spurious'][0]['predicted_HF_RMS']))
   if source not in ['C','S']:continue
   newmask=masks['newly_covered_cert_'+source];stable=masks['context_eval_'+source];newreport=detail_cached(qp,raw,source,'newly_covered_certified',ne,target,stable,newmask,teacherpath)
   for sc in newreport['scales']:
    s=sc['sigma_pixels'];cached=next(r for r in parentmetrics['newly_covered_rows']if r['raw']==raw and r['source']==source and r['scale']==s);assert np.array_equal(masks[f'new_core_cert_{source}_sigma{int(s)}'],masks[f'r2_report_new_core_{source}_sigma{int(s)}']);assert sc['core_pixels']==cached['core_n'];cs=newmask&core(stable,s);newrows.append(dict(raw=raw,source=source,scale=s,newly_covered_certified_n=int(newmask.sum()),core_n=sc['core_pixels'],boundary_n=int(newmask.sum())-sc['core_pixels'],Adam_RGB_MAE=cached['RGB_MAE'],numeric_RGB_MAE=stats(ne,target,cs)['MAE'],Adam_HF_cosine=cached['HF_cosine'],numeric_HF_cosine=sc['core']['HF_cosine'],Adam_HF_energy_ratio=cached['HF_energy_ratio'],numeric_HF_energy_ratio=sc['core']['HF_energy_ratio'],Adam_HF_RMSE=cached['HF_RMSE'],numeric_HF_RMSE=sc['core']['HF_RMSE'],Adam_HF_reused_from_existing_R2=True))
   for name,mask in [('newly_covered_certified',newmask),('common_stable_certified',masks['common_stable_cert_'+source]),('full_certified',masks['full_cert_'+source])]:
    stablemask=mask&(alpha>=.95);partitions.append(dict(raw=raw,source=source,partition=name,n=int(mask.sum()),unstable_n=int((mask&~(alpha>=.95)).sum()),Adam_premult_alphaT=stats(aprem,alpha[...,None]*target,mask),numeric_premult_alphaT=stats(nprem,alpha[...,None]*target,mask),stable_n=int(stablemask.sum()),Adam_stable_effective_RGB=stats(ae,target,stablemask),numeric_stable_effective_RGB=stats(ne,target,stablemask)))
  print(json.dumps(dict(CPU_evaluated_raw=raw,HF_cached=True,seconds=time.monotonic()-start)),flush=True)
 solver=dict(FG=receipt['FG'],accepted=receipt['accepted'],stop_reason=receipt['termination'],initial_objective=receipt['initial_evaluation']['objective'],final_objective=receipt['objective'],objective_decrease=receipt['initial_evaluation']['objective']-receipt['objective'],initial_FW_gap=receipt['initial_evaluation']['FW_gap'],final_FW_gap=receipt['FW_gap'],initial_relative_FW_gap=receipt['initial_evaluation']['relative_FW_gap'],final_relative_FW_gap=receipt['relative_FW_gap'],initial_projected_gradient_inf=receipt['initial_evaluation']['CI_gradient']['projected_gradient_inf'],final_projected_gradient_inf=receipt['CI_gradient']['projected_gradient_inf'],initial_terms=receipt['initial_evaluation']['loss_terms'],final_terms=receipt['accepted_details']['loss_terms'],source='numeric_receipt.json',strict_numerical_certificate=False)
 summary=dict(schema='numeric_color_independent_summary.v1',id='numeric_color',display_name='数值诊断：同目标有界求解',formal_reference='R2 Adam1200 unchanged',point_counts=dict(total=len(oldasset['active']),active=int(oldasset['active'].sum()),new_points=0,allowed_color_rows=len(allowed)),frames=images,alpha_reference='exact same as formal R2 appearance; reuse formal alpha images',solver=solver,same_R2_common_stable_rows=rows,newly_covered_certified_rows=newrows,full_partition_RGB=partitions,mask_inclusion_audit='mask_inclusion_audit.json',mechanical_checks=mechanical,parameter_freeze_proof=proof,Adam_HF_reused=True,numeric_HF_evaluated_once_with_input_cache=True,box_floor_reference='../fixed_color_interval/boxed_rgb_summary.json',limitations=['Fixed final geometry/alpha and1191 shared CI with the exact restored target/normalizers; not a global representation ceiling.','The box RGB floor is a per-pixel relaxation and not a coupled multi-view attainability certificate.','FW and projected gradients are measured with FP32 image/derivative operations and FP64 scalar reductions, not a strict infinite-precision certificate.','This is a separate numerical diagnostic, not a replacement of the formal Adam method.'],source_SHA256={str(manifest):sha(manifest),str(D/'numeric_receipt.json'):sha(D/'numeric_receipt.json'),str(P/'bench/metrics.json'):sha(P/'bench/metrics.json'),str(D/'mask_inclusion_audit.json'):sha(D/'mask_inclusion_audit.json')},formal_results_changed=False,additional_GPU_calls=0)
 write(D/'numeric_summary.json',summary)
 wanted=[D/'numeric_summary.json',manifest,D/'mask_inclusion_audit.json',Path(__file__),D/'prepare_numeric_masks.py']
 for f in exp.get('lightweight_artifacts',[]):
  p=Path(f);p=p if p.is_absolute()else D/p
  if not p.resolve().is_relative_to(D.resolve()):raise ValueError('artifact outside diagnostic root')
  if 'new_numeric_asset'in p.parts:raise ValueError('large asset must remain remote')
  wanted.append(p)
 for frame in images.values():wanted.extend(D/v for v in frame.values())
 wanted.extend((D/'cpu_masks').glob('*.npz'));wanted.extend((D/'training_masks').glob('*.npz'))
 final=D/'numeric_color_packet.tar.gz';tmp=D/'numeric_color_packet.tar.gz.tmp';added=set();size=0
 with tarfile.open(tmp,'w:gz',dereference=True)as tar:
  for p in wanted:
   if not p.is_file():raise FileNotFoundError(p)
   name=p.relative_to(D).as_posix()
   if name in added:continue
   if p.stat().st_size>20_000_000:raise ValueError('unexpected large artifact:'+name)
   info=tar.gettarinfo(str(p),arcname=name)
   with p.open('rb')as f:tar.addfile(info,f)
   added.add(name);size+=p.stat().st_size
 tmp.replace(final);record=dict(schema='numeric_color_CPU_delivery.v1',complete=True,input_binding=binding,summary=str(D/'numeric_summary.json'),bundle=str(final),bundle_sha256=sha(final),bundle_bytes=final.stat().st_size,files=len(added),uncompressed_bytes=size,CPU_seconds=time.monotonic()-start,additional_GPU_calls=0,formal_Adam_unchanged=True);write(done,record);print(json.dumps(record),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);a=ap.parse_args()
 try:main(a.manifest)
 except Exception as e:write(D/'cpu_evaluation/failed.json',dict(error=str(e),traceback=traceback.format_exc()));raise
