"""构建用户已授权的道路来源消融，保持正在运行的完整版源码不变。"""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile

W=Path(__file__).resolve().parent
OUT=W/'ablations'
VARIANTS={
    'real_only':dict(observed_rgb=1.0,prior_rgb=0.0,observed_depth=0.1,prior_depth=0.0),
    'prior_only':dict(observed_rgb=0.0,prior_rgb=0.1,observed_depth=0.0,prior_depth=0.01),
    'ownership_only':dict(observed_rgb=0.0,prior_rgb=0.0,observed_depth=0.0,prior_depth=0.0),
}

def once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('source anchor mismatch: '+old[:100])
    return text.replace(old,new)

def build():
    if OUT.exists():raise FileExistsError('preserve existing ablation source and evidence')
    OUT.mkdir()
    for name,weights in VARIANTS.items():
        arm=OUT/name;overlay=arm/'candidate';shutil.copytree(W/'candidate',overlay)
        context=overlay/'utils/v2_runtime_context.py'
        source=context.read_text(encoding='utf-8')
        source=once(source,"        self.config=self.meta['config']",f"""        self.config=json.loads(json.dumps(self.meta['config']))
        self.ablation_variant={name!r}
        if args.get('v2_ablation_variant')!=self.ablation_variant:
            raise RuntimeError('ablation arm/source mismatch')
        # 仅关闭指定来源目标；不改来源标签、查询覆盖、目标RGB或归属修复。
        self.config['road']['objective_weights']={weights!r}""")
        source=once(source,'            self.gate_details.update(finite_gradient_parameter_groups=counts,optimizer_updates=0,',
            "            self.gate_details.update(ablation_variant=self.ablation_variant,\n"
            "                                     actual_road_weights=self.config['road']['objective_weights'],\n"
            '                                     finite_gradient_parameter_groups=counts,optimizer_updates=0,')
        ast.parse(source);context.write_text(source,encoding='utf-8',newline='\n')
        files={p.relative_to(overlay).as_posix():dict(candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in overlay.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        manifest=dict(status='ISOLATED_ROAD_SOURCE_ABLATION',variant=name,files=files,
            logical_budget=30000,seed=0,second_stage_updates=0,road_weights=weights,
            input_bundle='input_bundle_002',provenance_changed=False,query_coverage_changed=False,
            ownership_fix_identical_to_full=True,native_dynamic_methods_identical_to_full=True,
            training_started=False,GPU_gate_passed=False)
        (arm/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    runner=(W/'run_candidate.py').read_text(encoding='utf-8')
    runner=once(runner,"    import fcntl\n    lock=a.gpu_lock.open('a+')\n    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)",
        "    from ablation_support import shared_admission, collective_storage_required\n"
        "    arm_lock=shared_admission(a.output,a.manifest)\n"
        "    variant=read(a.manifest)['variant']\n"
        "    if a.output.parent.name!=variant:raise RuntimeError('wrong isolated arm output')")
    runner=once(runner,"    if apps:raise RuntimeError('GPU has existing compute processes: '+apps)",
        "    # shared_admission已核验所有CUDA进程属于当前白名单。")
    runner=once(runner,"    if torch.cuda.get_device_properties(0).total_memory<30*1024**3:raise RuntimeError('actual GPU has less than30GiB memory')",
        "    if torch.cuda.get_device_properties(0).total_memory<30*1024**3:raise RuntimeError('actual GPU has less than30GiB memory')\n"
        "    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory,0)")
    runner=once(runner,"    cfg['v2_policy']=data['config']", "    cfg['v2_policy']=data['config']\n    cfg['v2_ablation_variant']=variant")
    runner=once(runner,'        required=remaining*estimate+4e9',
        '        required=max(remaining*estimate+4e9,collective_storage_required(a.output.parent.parent))')
    ast.parse(runner);(OUT/'run_ablation.py').write_text(runner,encoding='utf-8',newline='\n')
    for name in ['ablation_support.py','run_road_ablation_group.py']:
        shutil.copyfile(W/name,OUT/name)
    plan=dict(task='WS-V82-R15-GN11-STAGE1-V2-ROAD-ABLATIONS-20261019',
        arms=VARIANTS,iterations_each=30000,second_stage_updates=0,seed=0,
        shared_ownership_fix=True,changed_factor='road source objective weights only',
        interpretation='2x2 road-source factorial with common ownership repair; ownership-only also compared with actual original native A nodes',
        GPU_memory_GiB=dict(per_ablation=10,main_allowance=8,margin=2),
        max_concurrent_training_including_main=2,
        checkpoint_steps=list(range(5000,30001,5000)),automatic_restart=False,
        main_engine_modified=False,human_verdict=None)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    with zipfile.ZipFile(W/'road_ablations_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(OUT).as_posix())
    print(json.dumps(dict(arms=list(VARIANTS),fresh_updates_each=30000,main_modified=False)))

if __name__=='__main__':build()
