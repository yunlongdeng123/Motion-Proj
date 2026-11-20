"""Replay the exact highres CPU stream loop on deterministic old480 rows; zero GPU."""
from pathlib import Path
import argparse,ast,json,hashlib,sys
import numpy as np
import torch

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
DEFAULT=G/'quality_bench_20261120_r1/highres'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def main(out):
    out=Path(out);dest=out/'stream_480_qualification.json'
    if dest.exists():raise FileExistsError('preserve prior qualification')
    torch.set_num_threads(1);sys.path[:0]=[str(Q/'code'),str(Q/'pixel_render/code')]
    from fixed_material_read_operator import FixedReadOperator,bilinear
    from pixel_quadrature_stream import iter_pixel_chunks
    from world_texture_runtime import WorldTextureRuntime
    source=out/'prepare_cpu_highres.py';tree=ast.parse(source.read_text())
    loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='iter_pixel_chunks']
    assert len(loops)==1
    loop=loops[0];module=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]));code=compile(module,str(source)+':actual_stream_loop','exec')
    fields={};base=valid=None;bindings={str(source):sha(source)}
    for case in ['M01','M11']:
        p=G/f'formal_r1/{case}/C800/field.pt';model,_=WorldTextureRuntime.load(p,'cpu')
        with torch.no_grad():L,H=model.fields()
        fields[case]=(L.numpy().copy(),H.numpy().copy(),model.H_gate.numpy()[...,0].copy())
        if base is None:base=model.base.numpy().copy();valid=model.valid.numpy()[...,0].copy()
        else:assert np.array_equal(base,model.base.numpy())and np.array_equal(valid,model.valid.numpy()[...,0])
        bindings[str(p)]=sha(p);del model
    rows=[]
    for raw in [110,86]:
        mapfile=Q/f'pixel_render/maps/{raw:03d}.npz';m=load(mapfile,['center_uv','pixel_uv_jacobian','canonical_mesh_hit_valid'])
        frozen=load(G/f'evaluation/frozen_r2/{raw:03d}.npz',['A','C','S']);root=Q/f'pixel_render/read_operator_box_r1/{raw:03d}';op=FixedReadOperator.load(root)
        domain=m['canonical_mesh_hit_valid'];chosen=[];group_counts={}
        for name,mask in [*frozen.items(),('other_canonical',domain&~(frozen['A']|frozen['C']|frozen['S']))]:
            ids=np.flatnonzero(mask&domain);take=ids[np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=int)]if len(ids)else np.array([],np.int64)
            chosen.extend(take);group_counts[name]=len(take)
        selected=np.unique(chosen);N=len(selected);maps=dict(center_uv=m['center_uv'].reshape(-1,2)[selected],pixel_uv_jacobian=m['pixel_uv_jacobian'].reshape(-1,2,2)[selected])
        env=dict(np=np,bilinear=bilinear,iter_pixel_chunks=iter_pixel_chunks,maps=maps,fields=fields,base=base,valid=valid,
            colors={name:np.zeros((N,3),np.float32)for name in fields},mass_all=np.zeros(N,np.float64),points=0)
        exec(code,env)
        lookup=np.full(320*480,-1,np.int64);lookup[op.row_ids]=np.arange(len(op.row_ids));idx=lookup[selected];assert(idx>=0).all()
        basis_path=G/f'native_qualification_r2/basis/{raw:03d}/road_basis.npz';b=load(basis_path,['baseline_rgb','eligible_rgb','eligible_weight'])
        for case,(L,H,gate)in fields.items():
            key='observed_H'if case=='M01'else'observed_plus_teacher_H'
            compiled=np.clip(op.base[idx]+op.KL[idx]@L.reshape(-1,3)+op.KH[key][idx]@H.reshape(-1,3),0,1).astype(np.float32)
            direct=env['colors'][case];mass=env['mass_all'];ok=mass>1e-12
            pred=b['baseline_rgb'].reshape(3,-1)[:,selected]+np.where(ok[None],b['eligible_weight'].ravel()[selected][None]*direct.T-b['eligible_rgb'].reshape(3,-1)[:,selected],0)
            saved_path=G/f'formal_r1/{case}/C800/rendered/{raw:03d}.npz';saved=load(saved_path,['road_rgb'])['road_rgb'].reshape(3,-1)[:,selected]
            rows.append(dict(raw=raw,case=case,deterministic_rows=N,selection_counts=group_counts,quad_points=env['points'],
                direct_vs_compiled_color_absmax=float(np.abs(direct-compiled).max()),
                direct_vs_compiled_source_mass_absmax=float(np.abs(mass-op.mass[idx]).max()),
                reconstructed_vs_saved_road_RGB_absmax=float(np.abs(pred-saved).max()),
                exact_zero_source_mass=int((mass==0).sum()),source_sha256={str(p):sha(p)for p in [mapfile,basis_path,saved_path,root/'read.json',root/'constants.npz',root/'low.npz',root/f'high_{key}.npz']}))
        np.savez_compressed(out/f'stream480_{raw:03d}_rows.npz',pixel_ids=selected,source_mass=env['mass_all'],**env['colors'])
    result=dict(schema='actual_highres_stream_on_old480_inputs.v1',rows=rows,field_and_loop_source_SHA256=bindings,
        actual_loop_AST_SHA256=hashlib.sha256(ast.dump(loop,include_attributes=False).encode()).hexdigest(),
        actual_loop_source=ast.get_source_segment(source.read_text(),loop),
        selection='64 evenly-spaced raster-order canonical pixels per fixed A/C/S/other group, deduplicated; no prediction or residual selection',
        sampled_not_exhaustive=True,not_a_new_highres_geometry_proof=True,
        conclusion='Checks identity of the exact executed stream body to old compiled query and saved native-basis roadRGB on selected480 rows; no field/mask/renderer change.',
        new_native_calls=0,new_backward_calls=0,optimizer_updates=0,GPU_operations=0)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();main(a.output)
