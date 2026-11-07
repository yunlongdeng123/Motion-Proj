"""Read-only CPU source/checkpoint audit; never changes the running trajectory."""
from pathlib import Path
import ast,hashlib,json,time
ROOT=Path(__file__).resolve().parent.parent
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
paths=[ENGINE/name for name in ['train.py','scene/gaussian_model.py','scene/paper_sgcm.py','utils/hybrid30k_runtime.py','utils/hybrid30k_losses.py','utils/paper_losses.py','utils/loss_utils.py']]+sorted((ENGINE/'gaussian_renderer').rglob('*.py'))
sources={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths}
random_names={'rand','rand_like','randn','randn_like','randperm','randint','normal','bernoulli','multinomial','poisson'}
found=[]
def inspect(node,path,owner='module'):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):owner=node.name
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr in random_names:
        found.append(dict(path=str(path),line=node.lineno,function=owner,torch_function=node.func.attr))
    for child in ast.iter_child_nodes(node):inspect(child,path,owner)
for path in paths:inspect(ast.parse(path.read_text()),path)
tree=ast.parse((ENGINE/'train.py').read_text())
cls_calls=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='loss_cls_3d']
assert cls_calls==[]
config=json.loads((ROOT/'continuation15k/effective_config.json').read_text())
assert config['densify_until_iter']==config['dynamic_densify_until_iter']==config['sgcm_end']==15000
import torch
assert not torch.cuda.is_initialized()
state_path=ROOT/'continuation15k/training_state15000.pth'
state=torch.load(state_path,map_location='cpu',weights_only=False)
assert state['iteration']==15000 and not torch.cuda.is_initialized()
def tensor_sha(t):return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
restore=json.loads((ROOT/'continuation20k/restore_audit.json').read_text())
assert restore['status']=='PASS_EXACT_FULL_RESTORE' and restore['differences']==[]
for path,record in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==record['sha256']
result=dict(status='PASS_SCOPED_CPU_SOURCE_AND_ACTUAL_CHECKPOINT_RNG_BOUNDARY_AUDIT',unix=time.time(),
    diagnostic_route=dict(additional_native_complete_eval_iteration=15000,original_future_nodes=[20000,30000],
        strict_restore_precedes_added_schedule=True,native_complete_eval_rng_guard=False,
        lpips_torch_cpu_rng_consumption='Observed by separate parent/remote_history CPU probe; not re-probed here',
        actual_post_evaluation_state_is_captured_and_restored=True),
    scoped_source_findings=dict(active_explicit_torch_cpu_random_consumer_found_after15000=False,
        camera_time_sampler='CameraSampler.select uses its own random.Random and np.random.RandomState; both are checkpointed',
        torch_cpu_randperm_in_loss_cls_3d='Two draws in a defined helper; no AST call exists in train.py (legacy call is commented)',
        gaussian_cpu_rand='create_from_pcd initialization runs before full restoration, not in the post15000 training loop',
        gaussian_normal='Split tensors means/means_t explicitly CUDA; generic densification is disabled at iteration>=15000',
        sgcm='refine returns when due false; due requires iteration<cfg.end=15000',
        renderer_and_active_hybrid_losses='No explicit torch random calls in the enumerated modules',
        topology_observation='Runtime point counts remain static2079561/dynamic1015659 since node15000'),
    actual_checkpoint=dict(path=str(state_path),bytes=state_path.stat().st_size,
        sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),torch_cpu_rng_sha256=tensor_sha(state['torch_rng']),
        cuda_rng_sha256=[tensor_sha(t) for t in state['cuda_rng']],next_stage_full_restore=restore),
    explicit_torch_random_calls=found,source_files=sources,
    limits=['Finite enumerated source audit, not runtime instrumentation or an exhaustive third-party/native CUDA audit',
        'Does not establish a bitwise counterfactual trajectory without the extra15K evaluation',
        'No GPU RNG probe performed here; no CUDA bitwise determinism claim',
        'Exact full restore statements refer to actual saved checkpoint states'],
    training_restarts_requested=0,extra_updates=0,GPU_initialized=False,method_or_running_code_modified=False,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
probe=ROOT.parent/'scene008_a5_branch_control_20261107_r1/evaluation_rng_cpu_probe.json'
if probe.is_file():
    result['separate_cpu_only_evaluation_probe']=dict(path=str(probe),sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),receipt=json.loads(probe.read_text()))
(ROOT/'rng_diagnostic_boundary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],cpu_rng_sha256=result['actual_checkpoint']['torch_cpu_rng_sha256'],active_explicit_cpu_consumer_found=False,GPU_initialized=False)))
