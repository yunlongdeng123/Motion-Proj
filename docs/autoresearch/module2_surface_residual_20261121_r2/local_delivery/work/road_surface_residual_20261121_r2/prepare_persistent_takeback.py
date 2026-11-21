"""CPU-only archive of completed R2 road assets and full phase checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,tarfile,time

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2')
OUT=ROOT/'delivery/persistent_asset_takeback_r2'
EXPECTED_ROAD='70cc545db6ce595b24c0ec2ba836d7058edbb73734d6873c880f3e634fb08d9f'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'package_receipt.json').exists():
        rec=read(OUT/'package_receipt.json');assert sha(rec['archive'])==rec['archive_sha256'];print(json.dumps(rec));return
    assert read(ROOT/'status.json')['complete'] and read(ROOT/'receipt.json')['complete']
    assert sha(ROOT/'appearance_fit/asset/road.npz')==EXPECTED_ROAD
    files={}
    def add(rel,source,role):
        p=Path(source);assert p.is_file() and not p.is_symlink(),str(p)
        if rel in files:raise ValueError('duplicate archive member '+rel)
        files[rel]=dict(source=str(p),bytes=p.stat().st_size,sha256=sha(p),role=role)
    # Final road and the exact bases required by appearance / coverage checkpoint bindings.
    for stage in ('appearance_fit','coverage_fit','dense_init'):
        for f in sorted((ROOT/stage/'asset').rglob('*')):
            if f.is_file():add(f.relative_to(ROOT).as_posix(),f,'final_asset'if stage=='appearance_fit'else'checkpoint_base_asset')
    for stage in ('appearance_fit','coverage_fit'):
        add(stage+'/checkpoint.pt',ROOT/stage/'checkpoint.pt','complete_optimizer_parameter_RNG_checkpoint')
    for f in sorted((ROOT/'proposal').glob('*')):
        if f.is_file() and f.suffix in ('.npz','.json','.py'):add(f.relative_to(ROOT).as_posix(),f,'frozen_selected_geometry_and_provenance')
    for f in sorted((ROOT/'proposal/domains').glob('*.npz')):add(f.relative_to(ROOT).as_posix(),f,'exact_loss_domains')
    for rel in ['contract.json','receipt.json','status.json','resource_ledger.json','export_manifest.json','teacher_manifest.json','smoke/receipt.json']:
        add(rel,ROOT/rel,'completed_run_binding')
    for rel in ['code/run_residual.py','code/supervisor.py','cpu/prepare_residual_candidates.py','prepare_training_inputs.py','k_gate/gate_candidates.py']:
        if rel not in files:add(rel,ROOT/rel,'actual_execution_source')
    for f in sorted((ROOT/'train_inputs').glob('*')):
        if f.is_file() and f.suffix in ('.npz','.json','.py'):add(f.relative_to(ROOT).as_posix(),f,'exact_R2_scope_and_original_reuse_caps_mapping')
    r1=ROOT.parent/'road_surface_adaptive_20261121_r1'
    for rel in ['code/run_adaptive.py','proposal/activation.npz','proposal/append.npz']:
        add('R1_dependency/'+rel,r1/rel,'exact_R1_helper_or_original_cap_authority')
    for rel in ['k_gate/per_candidate_K_evidence.npz','k_gate/safe_candidates.npz','k_gate/receipt.json','k_gate/summary.json']:
        add(rel,ROOT/rel,'actual_joint_K_gate_and_selection_provenance')
    add('restore_code/prepare_persistent_takeback.py',Path(__file__),'archive_preparation_source')
    import torch
    torch.set_num_threads(1)
    checkpoints={}
    for stage,cursor,schema in [('appearance_fit',1200,'r2_appearance_full_state.v1'),('coverage_fit',400,'adaptive_coverage_full_state.v1')]:
        d=torch.load(ROOT/stage/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert d['schema']==schema and d['cursor']==cursor and 'optimizer'in d and 'RNG'in d
        assert 'binding'in d and d['role']=='formal'
        if stage=='appearance_fit':assert 'normalizers'in d and 'rgb'in d
        else:assert 'params'in d and 'trace'in d
        checkpoints[stage]=dict(schema=d['schema'],cursor=d['cursor'],binding=d['binding'],optimizer_parameter_groups=len(d['optimizer']['param_groups']),optimizer_state_entries=len(d['optimizer']['state']),RNG_keys=list(d['RNG']),normalizers_included='normalizers'in d,parameter_shapes={k:list(v.shape)for k,v in d.get('params',{}).items()},checkpoint_sha256=files[stage+'/checkpoint.pt']['sha256'])
    coverage_base=files['dense_init/asset/road.npz']['sha256'];appearance_base=files['coverage_fit/asset/road.npz']['sha256']
    assert checkpoints['coverage_fit']['binding']['asset_road']==coverage_base
    assert checkpoints['appearance_fit']['binding']['road']==appearance_base
    # External native model files stay remote. Expected hashes come from the pinned proof.
    scene=read(ROOT/'appearance_fit/asset/persistent/scene.json');proof=read(ROOT/'appearance_fit/asset/persistent/proof.json')
    external={}
    selected_sources={x['source']for x in files.values()}
    for x in [*proof['native_sources'],proof['ownership']]:
        p=Path(x['path']);external[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=x['sha256'],role=x.get('role','ownership'),copied=False,hash_provenance='pinned persistent proof; native model contents not rehashed for this takeback')
    def dependency(path,role):
        p=Path(path)
        if str(p)in selected_sources or str(p)in external:return
        if not p.exists():external[str(p)]=dict(path=str(p),exists=False,role=role,copied=False);return
        if p.is_file():external[str(p)]=dict(path=str(p),exists=True,bytes=p.stat().st_size,sha256=sha(p),role=role,copied=False)
        else:external[str(p)]=dict(path=str(p),exists=True,role=role,copied=False,directory=True)
    for k in ('model','engine','world_to_model','ownership'):
        if k in scene:dependency(scene[k],'persistent_scene_'+k)
    for frame in scene['frames']:
        for k in ('observation','instance_mask'):
            if k in frame:dependency(frame[k],f'original_camera_or_keep_{k}')
    old=ROOT.parent/'road_common1152_20261121';parent=ROOT.parent/'road_surface_adaptive_20261121_r1/appearance_fit/asset'
    for f in [parent/'road.npz',parent/'provenance.npz',old/'common_permission/teacher_evidence.npz',old/'teacher/manifest.json',old/'teacher/raw086/supervision.npz',old/'teacher/raw110/supervision.npz']:
        dependency(f,'original_parent_or_teacher_loss_dependency')
    dependency('/root/autodl-tmp/motion_proj_v82_gn11_v42','renderer_repository_and_native_extensions')
    write(OUT/'manifest.json',dict(schema='surface_R2_persistent_asset_takeback.v1',complete=True,run=str(ROOT),final_road_sha256=EXPECTED_ROAD,files=files,payload_files=len(files),payload_bytes=sum(x['bytes']for x in files.values()),checkpoint_base_bindings_verified=True,model_raw_data_excluded=True,standalone_scene_backup=False,GPU_operations=0,training_updates=0))
    write(OUT/'dependencies.json',dict(schema='surface_R2_external_dependencies.v1',standalone_scene_backup=False,external=list(external.values()),note='Native static/dynamic/environment models, original observation/mask/camera inputs and teacher loss inputs are external dependencies; no raw data or original large model copied. Absolute source paths remain in saved manifests; relocation requires explicit proof/gate rebinding.'))
    write(OUT/'checkpoints.json',checkpoints)
    notes='''# R2道路资产与阶段恢复状态\n\n本包包含最终 appearance_fit/asset 的完整 road.npz、provenance.npz、scene/proof/static gates，以及 appearance、coverage 完整参数/Adam/RNG状态。coverage_fit/asset 是外观状态的精确基底；dense_init/asset 是覆盖状态的精确基底；另保留固定proposal、train_inputs、1191行许可与原R1复用caps映射、合法监督域和执行代码。\n\n按 checkpoints.json 校核绑定后，从对应保存基底构造同一原生上下文，再加载完整状态；不要混用最终颜色资产作为外观checkpoint基底。未在此次取回中调用GPU或重新执行恢复。R1父资产与原大场景依赖按dependencies清单保留；本包不把107个复用行记为新增。\n\n这不是完整独立scene备份。原静态、动态、环境大模型、原始图像/掩码与相机、教师监督输入等不复制，必要绝对路径和SHA见 dependencies.json。持久化manifest保留原路径；迁移时须使用明确的路径映射及proof/gate重绑定，不能单独手改scene.json。无默认画质晋升。\n'''
    (OUT/'RESTORE.md').write_text(notes)
    archive=OUT/'persistent_assets_and_states.tar.gz';tmp=OUT/'persistent_assets_and_states.tar.gz.partial'
    assert not archive.exists(), 'Preserve existing completed archive'
    with tarfile.open(tmp,'w:gz',compresslevel=1)as t:
        for rel,x in files.items():t.add(x['source'],arcname=rel,recursive=False)
        for n in ('manifest.json','dependencies.json','checkpoints.json','RESTORE.md'):t.add(OUT/n,arcname=n,recursive=False)
    assert all(sha(x['source'])==x['sha256']for x in files.values()),'source changed during packaging'
    tmp.rename(archive)
    rec=dict(schema='surface_R2_persistent_takeback_package.v1',complete=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),payload_files=len(files),payload_bytes=sum(x['bytes']for x in files.values()),manifest_sha256=sha(OUT/'manifest.json'),dependencies_sha256=sha(OUT/'dependencies.json'),checkpoints_sha256=sha(OUT/'checkpoints.json'),external_dependencies=len(external),standalone_scene_backup=False,original_files_unchanged=True,new_GPU_operations=0,seconds=time.monotonic()-started)
    write(OUT/'package_receipt.json',rec);print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
