"""候选真实chart/GH15缓存准备；只CPU，不等同独立验收或正式授权。"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from road_chart_sampling import make_exact_sample_plan

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
COVER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded')


def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def main(output):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    atlas=load(BASE/'base_atlas.npz',['surface_world_xyz','world_from_uv','grid_world_row'])
    road=load(COVER/'road.npz',['active','normal_world','gaussian_sigma_m'])
    source=load(COVER/'provenance.npz',['world_keys','world_xyz'])
    active=np.flatnonzero(road['active'])
    plan=make_exact_sample_plan(atlas['surface_world_xyz'],atlas['grid_world_row'],source['world_keys'],active,
        road['normal_world'][active],road['gaussian_sigma_m'][active],atlas['world_from_uv'],order=15,
        world_xyz=source['world_xyz'][active],chunk_rows=1024,sample_uv_path=output/'sample_uv_f64.npy')
    numeric={k:v for k,v in plan.items() if k not in ['sample_uv','sampling_receipt','center_receipt']}
    numeric['active_world_rows']=active
    np.savez_compressed(output/'geometry_plan.npz',**numeric)
    receipt=dict(schema='true_chart_GH15_CPU_candidate_plan.v1',GPU_executed=False,native_calls=0,
        training_updates=0,formal_authorized=False,independent_review_status='pending root/main/independent locator review',
        source_atlas=str(BASE/'base_atlas.npz'),source_road=str(COVER/'road.npz'),source_geometry=str(COVER/'provenance.npz'),
        sample_uv=str(output/'sample_uv_f64.npy'),sample_uv_shape=list(plan['sample_uv'].shape),
        sample_uv_bytes=int(plan['sample_uv'].nbytes),geometry_plan=str(output/'geometry_plan.npz'),
        center=plan['center_receipt'],mapping=plan['sampling_receipt'],elapsed_seconds=time.monotonic()-start,
        invalid_query_policy='outside actual chart mesh -> UV[-4,-4] zero sample; never use affine extrapolation as valid material',
        reload_authority='fixed surface_world_xyz/world_keys/normals/sigma plus helper; sampleUV is reproducible cache, not learned rowRGB')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=P/'chart_plan_candidate_r1');main(parser.parse_args().output)
