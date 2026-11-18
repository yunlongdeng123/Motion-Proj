"""Record the new explicit GPU authorization without changing CPU history."""
from pathlib import Path
import datetime, hashlib, json, subprocess

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
out=Q/'gpu_20261119'
out.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=['protocol.json','execution_contract.json','permission/final_r1/recipe.json',
        'pixel_render/preparation_receipt.json','trellis_hf/seed_r5_nodal_b034/H_seed.npz']
record=dict(schema='teacher_detail_gpu_authorization.v1',
    created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    user_instruction='GPU已开，继续推进实验，目标是把高频的改善好',
    GPU_authorized=True,smoke_authorized=True,formal_authorized_after_real_smoke=True,
    GPU_power_policy='keep_on_explicit_user_instruction',power_operations_authorized=False,
    scientific_protocol=str(Q/'protocol.json'),CPU_history_mutable=False,
    frozen_inputs_sha256={s:sha(Q/s) for s in frozen},
    owners=dict(native_basis='stage_qa',field_training='road_asset',evaluation='static_handoff',orchestration='root'),
    smoke=dict(lifecycle_seconds=600,formal_budget_separate=True,
               complete_state_restore=True,next_stage_entry=True),
    formal=dict(arms=['M00','M10','M01','M11','T11'],steps=dict(A=200,B=300,C=300),
                physical_updates_max=3400,logical_updates=4000,train_frames=[86,110],diagnostic_frames=[134,179,198],
                no_new_199_run=True,high_bound=.34,low_bound=.30,no_new_TRELLIS_inference=True),
    lease=dict(native_ctx_owner='stage_qa',field_CUDA_qualification_allowed=True,
               one_formal_training_controller=True,smoke_optimization_after_native_qualification=True),
    git=dict(active_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
             status_before=subprocess.check_output(['git','status','--short'],cwd=R,text=True)),
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True))
path=out/'authorization.json'
if path.exists():
    old=json.loads(path.read_text(encoding='utf-8'))
    assert old['frozen_inputs_sha256']==record['frozen_inputs_sha256']
    print('Existing authorization verified:',path)
else:
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)
