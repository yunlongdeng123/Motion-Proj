"""只读CPU资产目录；不加载native模型、Torch checkpoint或CUDA模块。"""
import os
for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='' if key=='CUDA_VISIBLE_DEVICES' else '1'
from pathlib import Path
import json
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
scene=json.loads((B/'persistent/scene.json').read_text())
proof=json.loads((B/'persistent/proof.json').read_text())
records=[proof['manifest'],proof['road'],proof['gates']['indices'],proof['gates']['checkpoint'],*proof['native_sources']]
if 'ownership' in proof:records.append(proof['ownership'])
inventory=[]
for record in records:
    path=Path(record['path']);exists=path.is_file()
    inventory.append(dict(path=str(path),exists=exists,bytes=path.stat().st_size if exists else None,
        declared_binding=record.get('sha256'),full_large_file_rehash='deferred to execution preflight'))
if not all(row['exists'] for row in inventory):raise FileNotFoundError('canonical input missing')
with np.load(B/'road.npz',allow_pickle=False) as saved:
    active=saved['active'].astype(bool)
    road=dict(total_world_rows=len(active),active_rows=int(active.sum()),
        meter_to_model=float(saved['meter_to_model']),
        fields=saved.files,XYZ_shape=list(saved['model_xyz'].shape),RGB_shape=list(saved['reference_srgb'].shape))
old=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1/phase0/lifecycle_r4/save_color/row_scopes.npz')
with np.load(old,allow_pickle=False) as saved:
    guard=int(saved['conservative_K_guard'].sum())
record=dict(schema='road_capacity_cpu_inventory.v1',status='cpu_inputs_present',canonical_asset=str(B),road=road,
    render_wh=scene['render_wh'],manifest_frames=len(scene['frames']),frames=[86,110,134,179,198],
    inherited_build_frames=199,primary_retained_envelope_active_rows=guard,
    legacy_scope_role='CPU basis sanity only; new A/C/S eligibility requires actual native contribution preflight after user opens GPU',
    resources=dict(cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),threads=1),
    input_files=inventory,torch_or_native_imported=False,large_checkpoint_loaded=False,
    new_native_raster_calls=0,new_real_scene_optimizer_updates=0,gpu_authorized=False,
    GPU_execution='stop until new user instruction after enabling GPU; no power operation or automatic controller')
C.mkdir(parents=True,exist_ok=True)
(C/'cpu_inventory.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],active_rows=road['active_rows'],guard_rows=guard,
    render_wh=record['render_wh'],native_calls=0,optimizer_updates=0)))
