"""真实世界点basis静态检查；不加载scene/renderer，不拟合真实颜色。"""
from pathlib import Path
import importlib.util
import json
import os
import resource
import subprocess
import sys

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('Set CUDA_VISIBLE_DEVICES empty before process startup')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
spec=importlib.util.spec_from_file_location('road_world_texture_basis_cpu',R/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_world_texture_basis.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import numpy as np

def load(path,keys):
    with np.load(path,allow_pickle=False) as a:return {key:a[key].copy() for key in keys}
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

source=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
old=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1/direct_C/formal_r1/row_scopes.npz')
road=load(source/'road.npz',('active','reference_srgb'))
provenance=load(source/'provenance.npz',('world_xyz','guarded_world_rows'))
scope=load(old,('active_world_rows','conservative_K_guard','color','actual_A_mass','actual_C_mass'))
active=np.flatnonzero(road['active'])
if not np.array_equal(active,scope['active_world_rows']):raise ValueError('Legacy scope topology changed')
world=provenance['world_xyz'][active].astype(np.float64)
base=road['reference_srgb'][active].astype(np.float32)
protected=scope['conservative_K_guard']|provenance['guarded_world_rows'][active]
output=C/'check3';output.mkdir(parents=True,exist_ok=True)
plan=m.make_color_basis_plan(world,base,scope['color'],protected,spacing_m=.04,world_row_ids=active,
    scope_source=str(old),source_role='legacy_direct_teacher_r4_static_preview_only')
statistics=m.basis_statistics(plan,row_mass=scope['actual_A_mass']+.5*scope['actual_C_mass'])
statistics.update(source_scope_role='legacy frozen real A/C r3 alphaT scope used only for static basis preview; not requalified new-v4 or five-view-ideal training permission',
    source_scope_path=str(old),mass_role='previously frozen A+.5C alphaT, no new scene rendering',
    source_geometry=str(source/'road.npz'),source_world_positions=str(source/'provenance.npz'))
plan.save(output/'world_basis_4cm.npz')
write(output/'basis_statistics_4cm.json',statistics)
initial={mode:np.array_equal(base+np.zeros_like(base),base) for mode in ('independent_ci','shared_world')}
if not all(initial.values()):raise RuntimeError('Common initialization not exact')
restored=m.ColorBasisPlan.load(output/'world_basis_4cm.npz')
reload_exact=np.array_equal(restored.base_srgb,plan.base_srgb) and np.array_equal(restored.weights,plan.weights)
write(output/'initialization_and_reload.json',dict(source_shader_RGB_exact=initial,shared_basis_export_reload_exact=reload_exact,
    road_gaussian_count=len(active),original_road_field_writes=0,actual_optimizer_updates=0,
    real_scene_render_calls=0,accelerator_API_calls=0,GPU_executed=False,CPU_environment_explicit=True))

# 注册的解析细节只检验纹理采样，不是GS teacher或真实相机容量结论。
def paint(xy):
    direction=np.array([np.cos(.3),np.sin(.3)]);normal=np.array([-direction[1],direction[0]])
    u=xy@direction;v=xy@normal-.35
    phase=np.mod(u-.017,1.5)
    lateral=np.clip((.05-np.abs(v))/.01+.5,0,1)
    longitudinal=np.clip(np.minimum(phase,.6-phase)/.01+.5,0,1)
    return lateral*longitudinal
y,x=np.mgrid[0:1.0001:.004,0:3.0001:.004];xy=np.c_[x.ravel(),y.ravel()]
target=.15+.65*paint(xy);sanity=[]
for spacing in (.02,.04,.08):
    shape=(int(np.ceil(1/spacing))+1,int(np.ceil(3/spacing))+1)
    v,u=np.mgrid[:shape[0],:shape[1]];nodes=np.c_[u.ravel()*spacing,v.ravel()*spacing]
    field=.15+.65*paint(nodes)
    ids,weights,inside,_=m.bilinear_basis(xy,np.zeros(2),spacing,shape)
    rgb=(field[ids]*weights).sum(1)
    region=target>.2
    sanity.append(dict(spacing_m=spacing,width_m=.10,period_m=1.5,phase_m=.017,angle_radians=.3,
        field_sample_MAE=float(np.abs(rgb-target).mean()),paint_region_MAE=float(np.abs(rgb[region]-target[region]).mean()),
        field_sample_max_error=float(np.abs(rgb-target).max()),target_grid_pitch_m=.004,
        interpretation='analytic texture bilinear sanity only; no Gaussian raster/teacher fitting/resolution selection'))
write(output/'analytic_resolution_sanity.json',dict(primary_fixed_spacing_m=.04,selection_from_results=False,
    interpolation='native sRGB bilinear, not gamma remapping',rows=sanity,GPU_executed=False))
entry=C/'code/road_capacity_cpu_entry.py'
blocked=subprocess.run([sys.executable,str(entry),'--execute'],capture_output=True,text=True)
if blocked.returncode!=3:raise RuntimeError('Execution gate did not block')
write(output/'execution_gate_probe.json',dict(return_code=blocked.returncode,stdout=json.loads(blocked.stdout),
    background_controller_created=False,training_or_render_started=False,GPU_executed=False))
write(output/'resource_receipt.json',dict(threads=1,torch_imported='torch' in sys.modules,
    native_renderer_imported=any(name.startswith('diff_gaussian_rasterization') for name in sys.modules),
    process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    large_checkpoints_loaded=0,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_executed=False))
print(json.dumps(dict(statistics=statistics,initialization=initial,reload_exact=reload_exact,execution_gate_blocked=True)),flush=True)
