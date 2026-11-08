"""Necessary independent-pixel colour-box floor under fixed Rchild seam RGB."""
import argparse,csv,hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy import sparse

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-protocol',type=Path,required=True);p.add_argument('--child-root',type=Path,required=True);p.add_argument('--Rchild-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    ref=read(a.input_protocol);source=Path(ref['r2_source']);assert sha(source)==ref['r2_source_sha256']
    s=importlib.util.spec_from_file_location('immutable_floor_inputs',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    inp=SimpleNamespace(**{k:Path(v) for k,v in ref['input_paths'].items()});d=m.inputs(inp)
    meta=m.load(a.child_root/'children.npz');ids=meta['variable_child_ids'];selection=m.load(a.Rchild_run/'selection.npz')
    free=np.searchsorted(ids,selection['free_child_ids']);assert np.array_equal(ids[free],selection['free_child_ids'])
    Kparent=m.load(inp.k_palette)['full_parent_RGB'];K=Kparent[meta['child_to_parent_ids']][ids].astype(np.float64);blocks=[]
    for frame in d['frames']:
        root=a.child_root/f"{frame['raw']:03d}_{frame['state']}";A=sparse.load_npz(root/'A_child_rgb_csr.npz').tocsr().astype(np.float64);v=m.load(root/'pixel_domains_baseline.npz')
        active=(v['protected_nonroad']|v['originalroad']).ravel()[v['pixel_index']]&(np.asarray(A.sum(axis=1)).ravel()>0)
        assert np.array_equal(v['pixel_index'][active],frame['pixel_index']);blocks.append(A[active])
    PA=sparse.vstack(blocks,format='csr');assert PA.shape==(9966,20725)
    RK=d['r0']+PA@(K-meta['initial_child_RGB'].astype(np.float64));Af=PA[:,free]
    lo=RK-Af@K[free];hi=lo+Af@np.ones_like(K[free]);assert np.all(lo<=hi+1e-14)
    lower=np.maximum(np.maximum(lo,-hi),0).mean(1);rows=[]
    for q in d['domains']:
        selected=q['active_photo_rows'];floor=(float(lower[selected].sum())+q['constant_error_sum'])/q['pixels']
        numerical_affinity_MAE_allowance=1e-6*len(selected)/q['pixels']
        rows.append(dict(raw=q['raw'],state=q['state'],domain=q['name'],full_pixels=q['pixels'],positive_A_pixels=len(selected),
            independent_pixel_box_floor=floor,original_target=q['original_target'],margin=q['original_target']-floor,
            floor_exceeds_original_target=floor>q['original_target'],
            illustrative_1e6_affinity_MAE_allowance=numerical_affinity_MAE_allowance,
            floor_minus_affinity_allowance_exceeds_target=floor-numerical_affinity_MAE_allowance>q['original_target']))
    result=dict(schema='R_CHILD_FIXED_SEAM_COLOURS_INDEPENDENT_PIXEL_BOX_FLOOR_CPU_V1',
        status='GUARD_BOX_FLOOR_EXCEEDS_ORIGINAL_GATE' if any(q['floor_exceeds_original_target'] for q in rows) else 'NO_SINGLE_DOMAIN_BOX_IMPOSSIBILITY_PROVED',
        fixed_seam_child_count=len(selection['seam_frozen_child_ids']),free_child_count=len(free),
        formula='lo=(I_K-O)-A_free*Kfree; hi=lo+A_free*1; per-channel distance of0 to[lo,hi], RGB mean, plus all fixed error pixels in complete original-domain denominator',
        exceeded_domains=sum(q['floor_exceeds_original_target'] for q in rows),domains=rows,
        bounds_scope='Necessary lower bounds in the measured frozen-geometry affine colour model; pixels independent, shared cross-view child coupling ignored. No exceedance does not establish joint feasibility.',
        affinity_allowance_scope='Illustrative conservative per-active-pixel1e-6 MAE allowance, not a new gate or a globally certified floating-raster bound.',
        source_sha256=sha(__file__),Rchild_protocol_sha256=sha(a.Rchild_run/'protocol.json'),
        solver_calls=0,GPU_forwards=0,backwards=0,optimizer_updates=0,original_inputs_modified=False)
    a.output.mkdir(parents=True)
    with (a.output/'complete16_guard_box_floors.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
