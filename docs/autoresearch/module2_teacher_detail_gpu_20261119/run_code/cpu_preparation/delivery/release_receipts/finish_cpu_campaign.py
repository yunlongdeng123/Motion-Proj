from pathlib import Path
import os,json,subprocess,shutil,datetime
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');C=Q/'publication';S=C/'prepared_r2/stage'
def command(*args,env=None):return subprocess.check_output(args,cwd=R,env=env,text=True).strip()
def read(path):return json.loads(Path(path).read_text())
pub=read(S/'publication_receipt.json');validation=read(S/'source_validation.json');start=read(Q/'registration/start.json')
assert pub['pushed']and validation['passed']and validation['tests_passed']==37
remote=command('git','ls-remote','origin','refs/heads/'+pub['branch']).split()[0];assert remote==pub['commit']==pub['verified_remote_head']
assert command('git','rev-parse','HEAD')==start['repo_HEAD']
registered=Q/'registration/active_index.before';idx=Path(command('git','rev-parse','--git-path','index'))
if not idx.is_absolute():idx=R/idx
old_entries=command('git','ls-files','--stage','-v',env=dict(os.environ,GIT_INDEX_FILE=str(registered)))
assert command('git','ls-files','--stage','-v')==old_entries
installed=[];preserved=[]
for name in ['fixed_material_read_operator','world_texture_runtime','gpu_entry_guard']:
    for rel in [f'scripts/worldsim_v82/{name}.py',f'tests/worldsim_v82/test_{name}.py']:
        value=subprocess.check_output(['git','show',pub['commit']+':'+rel],cwd=R);target=R/rel
        if target.exists():
            if target.read_bytes()!=value:preserved.append(rel);continue
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
        installed.append(rel)
navs=list((R/'docs/research_failures').glob('*.md'))+[R/'docs/RESEARCH_FAILURES.md']
before={str(x.relative_to(R)):x.read_bytes()for x in navs if x.exists()}
for rel,data in before.items():
    dest=C/'working_nav_before'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
regen=command('/root/autodl-tmp/envs/motionproj/bin/python',str(R/'scripts/build_research_failure_index.py'))
changed=[rel for rel,data in before.items()if(R/rel).read_bytes()!=data]
assert not changed,changed
contract=read(Q/'execution_contract.json');assert contract['GPU_authorized']is False and contract['training_authorized']is False
permission=read(Q/'permission/results.json');pixel=read(Q/'pixel_render/preparation_receipt.json');t2=read(Q/'trellis_hf/seed_entry.json')
assert permission['full_average_UV_plan_complete']and permission['sparse_average_operator_uploaded_and_verified']
assert pixel['uncapped_compiled_cache_ready']and pixel['CPU_actual_pixel_filter_qualified']
assert not pixel['GPU_executed']and not pixel['actual_native_basis_prepared']and not t2['GPU_authorized']
assert t2['exploratory_seed_valid']and not t2['strict_faithful_gate']
mapping=read(Q/'permission/avg_sparse_gh31_r1/row_id_contract.json');assert all(mapping['checks'].values())
for root in [Q/'permission/avg_sparse_gh31_r1/operator',*[Q/f'pixel_render/read_operator_box_r1/{raw:03d}'for raw in [86,110,134,179,198]]]:
    schema=read(root/'read.json');assert(root/'constants.npz').is_file()and(root/'low.npz').is_file()
    for gate in schema['gates']:assert(root/f'high_{gate}.npz').stat().st_size>0
processes=command('ps','-eo','pid,args').splitlines()
own=[x for x in processes if str(Q)in x and 'finish_cpu_campaign.py'not in x and 'ps -eo'not in x]
active=[x for x in own if any(token in x for token in ['compile_pixel_reads_cpu.py','compile_average','build_full_plan','run_material','run_matched','project_h_seed.py'])]
assert not active,active
release=Q/'delivery/release_receipts';release.mkdir(parents=True,exist_ok=True)
for path in [S/'publication_receipt.json',S/'source_validation.json',S/'source_validation.log',C/'prepared_r2/prepared.json',C/'prepared_r2/working_docs_applied.json',C/'prepared_r2/failure_index_regeneration.json']:
    shutil.copy2(path,release/path.name)
receipt=dict(schema='teacher_detail_CPU_finish.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),client_date='2026-11-19',
    status='CPU_PREPARED_STOP_BEFORE_GPU',publication=pub,isolated_source_tests=37,successful_subtests=validation.get('additional_successful_subtest_reports',0),
    source_validation_passed=True,active_HEAD_unchanged=True,active_index_entries_and_flags_unchanged=True,
    active_index_bytes_equal_registration=idx.read_bytes()==registered.read_bytes(),publication_index_bytes_preserved=pub['active_index_unchanged'],
    working_parallel_2D_status_preserved=read(C/'prepared_r2/working_docs_applied.json')['parallel_2D_status_preserved'],
    installed_verified_reusable_files=installed,existing_user_files_preserved=preserved,
    failure_index_regeneration=dict(log=regen,navigation_changed=changed),
    original_scene_and_default_provider_changed=False,active_campaign_controllers=active,pending_GPU_queue=False,
    GPU_authorized=False,scene_training_authorized=False,native_GPU_calls=0,scene_optimizer_updates=0,new_model_inferences=0,power_operations=0,
    hardware_power_state='not queried or changed',
    full_inputs_and_operators_retained_remote=str(Q),local_takeback='light metadata/source/protocol only, not full scene/operator backup',
    faithful_T2_gate=False,T2_exploratory_seed_valid=True,human_verdict=None,problem_solved=None)
(Q/'finish_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
shutil.copy2(Path(__file__),release/'finish_cpu_campaign.py')
print(json.dumps(dict(finished=True,commit=pub['commit'],tests=37,GPU_authorized=False,native_calls=0,installed=len(installed),preserved_user_files=preserved)))
