"""Freeze a separate formal implementation, never edit active GN11 or controls."""
from pathlib import Path
import shutil,ast,json,hashlib
W=Path(__file__).parent;D=W/'direct_implementation';D.mkdir(exist_ok=True)
for n in ['train.py','runtime.py','rigid_model.py','renderer.py','losses.py','training_policy.py','restore_contract.py','native_camera_sampler.py']:
    shutil.copy2(W/'source'/n,D/n)
shutil.copy2(W/'source/gn11_train.py',D/'gn11_train_reference.py')
for n in ['rigid_spatial_topology.py','formal_support.py']:shutil.copy2(W/n,D/n)
p=D/'train.py';s=p.read_text(encoding='utf-8')
def rep(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)
rep('from .restore_contract import verify_runtime_restore','from .restore_contract import verify_runtime_restore\n    from . import formal_support as formal')
rep('from restore_contract import verify_runtime_restore','from restore_contract import verify_runtime_restore\n    import formal_support as formal')
rep("'training_policy': training_policy, 'restoration_verified': False}","'training_policy': training_policy, 'restoration_verified': False, 'formal_runtime': formal.state_flags(static)}")
rep('training_policy = resolve_training_policy(vars(args), raw_reference)','training_policy = resolve_training_policy(vars(args), raw_reference)\n        formal.register_policy(training_policy)')
rep("policy['hybrid_lambda_dynamic_opacity'] = 0.0  # 新目标下面明确加入逐实例可见贡献，不叠加旧孤立alpha目标。","policy['hybrid_lambda_dynamic_opacity'] = 100.0  # Original isolated raw body alpha, separate from visible per-ID loss.")
rep('sampler = random.Random(args.seed)','sampler = formal.EpochSampler(args.seed,cameras)\n        spatial_policy = formal.setup_topology(config,scene)\n        native_topology = formal.native_static_function()')
rep("first = resume_contract['first_step']","first = resume_contract['first_step']\n            formal.restore_flags(static,restored['formal_runtime'],restored['step'],config)")
rep("'anisotropic_split': False, 'motion_model'","'anisotropic_split': True, 'motion_model'")
rep("'opacity': '100 * equal_known_instance_mean(full_HxW_MSE_of_joint_visible_contribution)'","'opacity': '100 * isolated_raw_body_MSE + 100 * mean_K_visible_ID_MSE'")
rep("'new_SGCM_clones': False, 'SGCM': 'inherited_30k_background; rigid_clone_prune_is_separate'","'new_SGCM_clones': False, 'SGCM': 'stage1 inherited; generic static and rigid spatial topology enabled'")
rep('''        if repair is not None and restored is None:
            from repair_probe import save_repair_probe
            save_repair_probe(output, 0, cameras, static, instances, config, environment, spec)
''','')
rep('camera = cameras[sampler.randrange(len(cameras))]','''camera = sampler.select(cameras)
            if step % config.sh_increase_interval == 0:
                instances.oneupSHdegree()
                static.oneupSHdegree()''')
rep("'dynamic_alpha': package['dynamic_visible_alpha'], 'dynamic_mask': union","'dynamic_alpha': package['dynamic_alpha'], 'dynamic_mask': union")
rep('            optimizer.step(); optimizer.zero_grad(set_to_none=True)','''            topology = formal.topology_update(static,instances,optimizer,package,config,scene,spatial_policy,step,camera,native_topology)
            formal.record_topology(output,topology)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)''')
rep('''            if step % 1000 == 0: instances.oneupSHdegree()
            # 首版固定拓扑便于区分表示与增密；core提供保身份/Adam的clone/prune，
            # 不把它冒充原SGCM或自动套用PVG temporal densification。
''','')
rep("'instance_mask_loss': float(mask_loss.detach()),", "'instance_mask_loss': float(mask_loss.detach()),\n                          'raw_isolated_body_MSE': float(((package['dynamic_alpha']-union)**2).mean().detach()),")
rep("if repair is not None and (step in args.probe_steps or step == args.iterations):","if repair is not None and step in args.probe_steps:")
rep("default=[1000,3000,10000,20000,30000]","default=[1,1000,3000,10000,20000,30000]")
rep("parser.add_argument('--probe-steps', type=int, nargs='+', default=[100,1000,3000,10000,20000,30000])","parser.add_argument('--probe-steps', type=int, nargs='*', default=[])")
ast.parse(s);p.write_text(s,encoding='utf-8')
p=D/'training_policy.py';s=p.read_text(encoding='utf-8')
s=s.replace("for name in ('sampler', 'python_rng'):","for name in ('python_rng',):")
needle="    if POLICY_FIELD in checkpoint:\n"
assert s.count(needle)==1
s=s.replace(needle,"""    sampler=checkpoint['sampler']
    if not isinstance(sampler,dict) or sampler.get('schema_version')!=1 or sampler.get('steps')!=step or len(sampler.get('ids',[]))!=49:
        raise ValueError('formal epoch sampler missing or wrong step')
    # Full stack/RNG validity and exact values are checked on EpochSampler.restore.
    if POLICY_FIELD in checkpoint:
""")
ast.parse(s);p.write_text(s,encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}
(W/'direct_source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FROZEN',len(manifest),'formal files; no training updates')
