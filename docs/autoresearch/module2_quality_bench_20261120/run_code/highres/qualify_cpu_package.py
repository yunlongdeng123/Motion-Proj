"""CPU package checks, source hashes and initial-state reconstruction; no native calls."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
DEFAULT=G/'quality_bench_20261120_r1/highres'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(out):
    out=Path(out)
    if(out/'package_qualification.json').exists():raise FileExistsError('preserve earlier package verification')
    sys.path.insert(0,str(Q/'code'))
    from fixed_material_read_operator import FixedReadOperator
    checks={};guards=[];initial=[];output_hashes={}
    native=read(out/'native_completed.json');assert native['native_attempts']==native['native_completed']==12
    for raw in [110,86]:
        t=load(out/f'targets/{raw:03d}.npz');b=load(out/f'native_r1/{raw:03d}/road_basis.npz',['alpha','K']);j=load(out/f'native_r1/{raw:03d}/joint_basis.npz',['alpha','K','baseline_rgb'])
        assert all(t[k].shape==(3,1280,1920)and t[k].dtype==np.float32 for k in ['T_A','T_C','T_S'])
        assert all(t[r+'_pre_alpha'].dtype==bool for r in ['A','C','S'])
        assert np.array_equal(t['K_vehicle'],b['K'])and np.array_equal(t['K_vehicle'],j['K'])
        overlap={r:int((t[r+'_pre_alpha']&t['K_vehicle']).sum())for r in ['A','C','S']};assert not any(overlap.values())
        guard=dict(raw=raw,prealpha_intersect_vehicle_K=overlap,vehicle_K_outside_teacher_guard=int((t['K_vehicle']&~t['K_teacher_guard']).sum()),
            teacher_guard_extra_pixels=int((t['K_teacher_guard']&~t['K_vehicle']).sum()),
            union_exclusion_equals_current_teacher_exclusion=not bool((t['K_vehicle']&~t['K_teacher_guard']).any()))
        guards.append(guard)
        for case in ['initial','low_A200','M01','M11','M11_noH','M10']:
            p=out/f'rendered/{case}/{raw:03d}.npz';z=load(p,['road_rgb','joint_rgb','road_alpha','joint_alpha','K'])
            key=f'{raw}:{case}';checks[key]=dict(finite=all(np.isfinite(a).all()for a in z.values()),
                CHW_RGB=z['road_rgb'].shape==z['joint_rgb'].shape==(3,1280,1920),road_alpha_exact=np.array_equal(z['road_alpha'],b['alpha']),
                joint_alpha_exact=np.array_equal(z['joint_alpha'],j['alpha']),K_exact=np.array_equal(z['K'],j['K']),
                joint_K_RGB_max=float(np.abs(z['joint_rgb']-j['baseline_rgb'])[:,j['K']].max()))
            assert all(checks[key][k]for k in ['finite','CHW_RGB','road_alpha_exact','joint_alpha_exact','K_exact'])
            assert checks[key]['joint_K_RGB_max']==0
            output_hashes[str(p)]=sha(p);del z
        # The original initial export has no field.pt. Zero L/H using its immutable
        # common buffers must reproduce the original480 render algebra, not be fitted.
        op=FixedReadOperator.load(Q/f'pixel_render/read_operator_box_r1/{raw:03d}')
        c=np.clip(op.base,0,1).astype(np.float32);color=np.zeros((320*480,3),np.float32);color[op.row_ids]=c;color=color.reshape(320,480,3).transpose(2,0,1)
        valid=np.zeros(320*480,bool);valid[op.row_ids]=op.mass>1e-12;valid=valid.reshape(320,480)
        oldbasis=load(G/f'native_qualification_r2/basis/{raw:03d}/road_basis.npz',['baseline_rgb','eligible_rgb','eligible_weight'])
        p=oldbasis['baseline_rgb']+np.where(valid[None],oldbasis['eligible_weight'][None]*color-oldbasis['eligible_rgb'],0)
        expected=load(G/f'formal_r1/pixel_initial/rendered/{raw:03d}.npz',['road_rgb'])['road_rgb']
        initial.append(dict(raw=raw,zero_L_H_vs_original480_initial_RGB_max=float(np.abs(p-expected).max()),
            numerical_reconstruction_only=True,read_only=True,optimization_updates=0))
        output_hashes[str(out/f'targets/{raw:03d}.npz')]=sha(out/f'targets/{raw:03d}.npz')
        output_hashes[str(out/f'maps/{raw:03d}.npz')]=sha(out/f'maps/{raw:03d}.npz')
    result=dict(schema='highres_package_CPU_qualification.v1',checks=checks,guard_relationships=guards,initial_reconstruction=initial,
        source_geometry_field_files_unchanged=True,new_native_calls=0,new_optimizer_updates=0,GPU_operations=0,
        caveats=['Context/score qualification remains the benchmark evaluator responsibility; full prealpha domains are never alpha filtered here.',
            'Full-native observed A extends beyond the finite road-asset chart; chart-external A is a source-domain extent diagnostic, not automatically a road-repair failure.',
            'Source branch RGB/alpha conservation and constant-basis errors are logged native floating errors, not bit-exact rendering claims.'])
    write(out/'package_qualification.json',result)
    source=read(out/'cpu_prepared.json')['sources_SHA256'];source.update({str(p):sha(p)for p in out.glob('*.py')})
    write(out/'freeze.json',dict(schema='highres_readonly_probe_freeze.v1',source_SHA256=source,output_SHA256=output_hashes,
        receipts={p.name:sha(p)for p in [out/'native_completed.json',out/'cpu_prepared.json',out/'rendered_completed.json',out/'package_qualification.json']},
        native_attempts=12,optimizer_updates=0,source_production_writes=False,GPU_keep_on=True))
    print(json.dumps(dict(checks=len(checks),guards=guards,initial=initial,freeze=str(out/'freeze.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();main(a.output)
