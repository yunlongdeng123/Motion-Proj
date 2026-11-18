"""隔离三个既有Gaussian：原生单行alpha直接核验CPU fixed-Σ投影；0优化。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from road_activation_builder import CAMPAIGN,PREVIOUS,REPO,write_json


def run(plan,output,lease_file,lease_owner):
    sys.path.insert(0,str(PREVIOUS/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(lease_file,lease_owner)
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path(plan).read_text())
    if len(source['cases'])>6:raise ValueError('最多6次已登记隔离前向')
    ctx=load_context(asset_dir=CAMPAIGN/'native_r1/asset_A1_guarded',lease_file=lease_file,lease_owner=lease_owner)
    rows=[]
    for case in source['cases']:
        raw,y,x,world=[int(case[k]) for k in ['raw','y','x','world_row']]
        local=np.flatnonzero(ctx.active_world_rows==world)
        if len(local)!=1:raise ValueError('校准行不在A1 active packet')
        packet=ctx.base_packet(raw)
        mask=torch.zeros_like(packet['mask']);mask[int(local[0])]=packet['mask'][int(local[0])]
        with torch.no_grad():out=ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,dict(packet,mask=mask),ctx.scene._background)
        actual=float(out['alpha'][0,y,x]);error=abs(actual-case['CPU_raw_alpha'])
        rows.append(dict(**case,isolated_native_raw_alpha=actual,CPU_raw_alpha_error=error,
            inherited_neg_fov=bool(ctx.scene.pipe.neg_fov)))
    ctx.assert_frozen();ctx.close()
    result=dict(schema='road_activation_isolated_fragment_calibration.v1',GPU_executed=True,
        optimizer_updates=0,native_calls=len(rows),prior_campaign_calls=116,campaign_native_calls=116+len(rows),
        rows=rows,max_CPU_raw_alpha_error=max(row['CPU_raw_alpha_error'] for row in rows),
        no_state_or_asset_modified=True,isolated_alpha_has_no_other_row_transmittance=True)
    write_json(output/'results.json',result)
    if result['max_CPU_raw_alpha_error']>2e-4:raise RuntimeError('实际单行alpha尚未匹配CPU模型')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=CAMPAIGN/'footprint_calibration_cpu_r1/plan.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json');p.add_argument('--lease-owner',required=True)
    a=p.parse_args();print(json.dumps(run(a.plan,a.output,a.lease_file,a.lease_owner),ensure_ascii=False),flush=True)
