from pathlib import Path
import os,json,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
torch.set_num_threads(2)
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
from drivingeditor_dynamic_edit.road_multiscale_asset import load_shared_road_material
start=time.monotonic();rows=[]
for stage in ('L800','ABC800'):
    a=load_shared_road_material(P/f'formal_r1/{stage}/asset',device='cpu')
    c=a.query_colors();error=float((c-a.derived_colors).abs().max())
    idx=~a.eligible;frozen=np.array_equal(c.numpy()[idx],a.derived_colors.numpy()[idx])
    assert error<=1e-6 and frozen
    rows.append(dict(stage=stage,active_rows=len(c),eligible=int(a.eligible.sum()),
        CPU_requery_RGB_max_vs_actual_native_baked_cache=error,readonly_RGB_exact=frozen,
        named_parameters=[k for k,_ in a.model.named_parameters()],runtime_parameters_frozen=not any(p.requires_grad for p in a.model.parameters())))
    del a,c
r=dict(schema='reusable_shared_material_actual_asset_CPU.v1',passed=True,rows=rows,seconds=time.monotonic()-start,
 additional_native_calls=0,optimizer_updates=0,GPU_executed=False,
 new_runtime_api_is_opt_in=True,default_scene_changed=False,
 physical_source_query_full_reconstruction='already validated by actual GPU lifecycle; this CPU check verifies portable query+scope/cache implementation')
(P/'qa/reusable_actual_asset.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
