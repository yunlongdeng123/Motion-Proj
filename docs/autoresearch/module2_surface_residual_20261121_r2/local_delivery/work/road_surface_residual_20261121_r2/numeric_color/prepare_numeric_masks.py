"""Read-only fixed R2 evaluation masks and inclusion audit; no HF evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import minimum_filter
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2');D=P/'diagnostic/color_optimality';OLD=P.parent/'road_common1152_20261121'
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def core(m,s):return minimum_filter(m.astype(np.uint8),size=2*int(3*s+.5)+1,mode='constant',cval=0).astype(bool)
def main():
 out=D/'cpu_masks';out.mkdir(parents=True,exist_ok=True);exp=json.loads((P/'export_manifest.json').read_text());ledger=load(OLD/'common_permission/teacher_evidence.npz');rows=[];pins={}
 def packet(stage,raw):
  r=next(x for x in exp['rows']if x['stage']==stage and int(x['raw'])==raw);p=Path(r['packet']);return load(p if p.is_absolute()else P/p)
 for raw in [86,110]:
  domain=load(P/f'proposal/domains/{raw:03d}.npz');parent=packet('parent',raw)['road_alpha'];alpha=packet('appearance',raw)['road_alpha'];allstable=np.logical_and.reduce([packet(s,raw)['road_alpha']>=.95 for s in ['parent','activated','dense_init','coverage','appearance']]);oldmap=load(OLD/f'maps/{raw:03d}.npz');tm_path=D/f'training_masks/{raw:03d}.npz';tm=load(tm_path)if tm_path.exists()else None;masks={};entry=dict(raw=raw,actual_training_masks_available=tm is not None,source_domains_SHA256=sha(P/f'proposal/domains/{raw:03d}.npz'),sources=[])
  for source in ['A','C','S']:
   full=domain['appearance_'+source];cert=full&domain['canonical_valid'];stable=full&(alpha>=.95);common=full&(parent>=.95)&(alpha>=.95)&allstable;newfull=full&(parent<.95)&(alpha>=.95);newcert=newfull&cert;masks.update({f'source_{source}':full,f'full_cert_{source}':cert,f'context_eval_{source}':stable,f'common_eval_context_{source}':common,f'common_stable_cert_{source}':common&cert,f'newly_full_{source}':newfull,f'newly_covered_cert_{source}':newcert})
   info=dict(source=source,newly_covered_certified_n=int(newcert.sum()),full_source_n=int(full.sum()),full_certified_n=int(cert.sum()),common_stable_certified_n=int((common&cert).sum()),scales=[])
   if tm is not None:
    for name,key in [('RGB',f'RGB_{source}'),('context',f'context_{source}'),('score',f'score_{source}')]:
     mask=tm[key].astype(bool);n=int(newcert.sum());info[name+'_inclusion_n']=int((newcert&mask).sum());info[name+'_inclusion_fraction']=float((newcert&mask).sum()/n)if n else None;info[name+'_equals_frozen_source']=bool(np.array_equal(mask,full))
   for sigma in [1,2,4]:
    newcore=newcert&core(stable,sigma);reportcore=newfull&core(stable,sigma);commoncore=common&core(common,sigma);masks[f'new_core_cert_{source}_sigma{sigma}']=newcore;masks[f'r2_report_new_core_{source}_sigma{sigma}']=reportcore;masks[f'common_core_{source}_sigma{sigma}']=commoncore;n=int(newcore.sum());r=dict(scale=sigma,newly_certified_eval_core_n=n,r2_report_core_n=int(reportcore.sum()),report_core_uncertified_n=int((reportcore&~cert).sum()),common_eval_core_n=int(commoncore.sum()))
    if tm is not None:
     traincore=core(tm[f'context_{source}'].astype(bool),sigma)&tm[f'score_{source}'].astype(bool);r['new_eval_core_in_RGB_n']=int((newcore&tm[f'RGB_{source}']).sum());r['new_eval_core_in_training_HF_core_n']=int((newcore&traincore).sum());r['new_eval_core_training_HF_inclusion_fraction']=float((newcore&traincore).sum()/n)if n else None;r['all_new_cert_pixels_in_training_HF_core_n']=int((newcert&traincore).sum())
    info['scales'].append(r)
   if source in ['C','S']:
    uv=oldmap['center_uv'][newcert];valid=oldmap['canonical_mesh_hit_valid'][newcert].copy();lo=np.floor(uv).astype(int);H,W=ledger['teacher_HF_conflict'].shape;valid&=(lo[:,0]>=0)&(lo[:,1]>=0)&(lo[:,0]+1<W)&(lo[:,1]+1<H);ix=lo[valid,0];iy=lo[valid,1];stats={}
    for key in ['teacher_HF_conflict','teacher_A_HF_conflict','teacher_material_valid']:
     flag=ledger[key];sample=np.zeros(len(ix),bool)
     for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:sample|=flag[iy+dy,ix+dx]
     stats[key+'_any4nodes_n']=int(sample.sum())
    info['existing_teacher_context_diagnostic']=dict(mapped_to_original_world_ledger_n=int(valid.sum()),unmapped_to_original_world_ledger_n=int((~valid).sum()),**stats,interpretation='registered world-node teacher evidence only; not new GT, not a loss mask veto or proof of pixel correctness')
   entry['sources'].append(info)
  target=out/f'{raw:03d}.npz';np.savez_compressed(target,**masks);pins[str(target)]=sha(target);rows.append(entry)
 result=dict(schema='fixed_R2_numeric_color_mask_inclusion.v1',rows=rows,mask_files_SHA256=pins,source_training_code=str(P/'code/run_residual.py'),source_training_code_SHA256=sha(P/'code/run_residual.py'),source_teacher_ledger=str(OLD/'common_permission/teacher_evidence.npz'),source_teacher_ledger_SHA256=sha(OLD/'common_permission/teacher_evidence.npz'),mask_definitions='newly certified=R2directparent alpha<.95 and Adamfinal>=.95 in fixed certified source; common=unchanged original R2 all-stage stable domain; scales use registered finite support only',no_HF_metrics_recomputed=True,no_new_teacher=True,no_mask_tuning=True,CUDA_calls=0)
 (D/'mask_inclusion_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'mask_root':str(out),'actual_training_masks_available':[r['actual_training_masks_available']for r in rows],'C_S_counts':[{k:v for k,v in s.items()if k!='existing_teacher_context_diagnostic'}for r in rows for s in r['sources']if s['source']in['C','S']]}))
if __name__=='__main__':main()
