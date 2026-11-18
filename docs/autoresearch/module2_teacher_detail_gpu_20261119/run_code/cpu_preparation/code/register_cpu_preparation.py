from pathlib import Path
import os,json,subprocess,datetime,shutil

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU preparation requires hidden CUDA')
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
P=Q.parent/'road_multiscale_material_20261118_r1';R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
Q.mkdir(parents=True,exist_ok=True)
for name in ['registration','protocol','permission','pixel_render','trellis_hf','delivery']:(Q/name).mkdir(exist_ok=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
status=git('status','--short');index=Path(git('rev-parse','--git-path','index'))
if not index.is_absolute():index=R/index
start=Q/'registration/start.json'
if start.exists():raise RuntimeError('Preparation already registered; preserve original snapshot')
shutil.copy2(index,Q/'registration/active_index.before')
start.write_text(json.dumps(dict(task_id='MODULE2-TEACHER-DETAIL-CPU-20261119',run='road_teacher_detail_cpu_20261119_r1',
 UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),client_date='2026-11-19',timezone='Asia/Singapore',
 repo_HEAD=git('rev-parse','HEAD'),publication_parent='58b60ea32e130daea14869e82c62feb7f91538ba',git_status=status,
 parent_campaign=str(P),input_roles=dict(real='A legal visible RGB',pseudo='frozen C hole and S shadow 2D teachers',
 diagnostic='raw134/179/198; inherited199BUILD not blind',TRELLIS='existing frozen single inference, HF-only secondary prior'),
 failure_ledger_refs=['V82-F06'],failure_ledger_delta='none_pending_CPU_evidence',
 GPU_authorized=False,native_CUDA_calls_authorized=False,new_model_inference_authorized=False,
 training_authorized=False,power_operations_authorized=False,automatic_pending_queue=False,
 stop_boundary='finish CPU data/code/protocol validation, then stop before all GPU execution'),ensure_ascii=False,indent=2)+'\n')
(Q/'execution_contract.json').write_text(json.dumps(dict(schema='teacher_detail_CPU_only.v1',
 task_id='MODULE2-TEACHER-DETAIL-CPU-20261119',GPU_authorized=False,training_authorized=False,
 native_forward_calls=0,native_backward_calls=0,new_inferences=0,optimizer_updates=0,power_operations=0,
 auto_launch=False,pending_controller=False,
 permitted=['read parent inputs','build teacher evidence and independent H permission','CPU differentiable shader and source audits','CPU extract existing TRELLIS high-pass','tests','protocol/report','source publication and result takeback'],
 future_GPU_smoke_and_formal='proposed only; not authorized in this turn',
 frozen=['Gaussian geometry','opacity','scale','active set','cameras','static scene','vehicles','existing diffusion and TRELLIS weights'],
 scientific_questions=['teacher evidence vs original provenance','footprint-averaged per-Gaussian RGB vs per-pixel world material','HF-only TRELLIS secondary prior']),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(registered=True,path=str(Q),GPU_authorized=False,training_authorized=False)))
