"""一次编译无cap的pixelbox材质读取；只有CPU，世界场仍为权威，可导入root共享compiler。

默认inspect。跨完整pixel行分块，不把局部分母误作整pixel分母。
"""
from pathlib import Path
import argparse,importlib.util,json,sys,time,hashlib
import numpy as np
from pixel_quadrature_stream import iter_pixel_chunks

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';P=B/'road_multiscale_material_20261118_r1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-cpu',action='store_true')
    parser.add_argument('--compiler',type=Path,default=Q/'code/fixed_material_read_operator.py')
    parser.add_argument('--density',type=float,default=2.);parser.add_argument('--frames',type=int,nargs='+',default=[86,110,134,179,198])
    parser.add_argument('--output',type=Path,default=Q/'pixel_render/read_operator_box_r1');args=parser.parse_args()
    if not args.execute_cpu:
        print(json.dumps(dict(native_default=False,CPU_execute_required=True,planned_frames=args.frames,density=args.density,
            bounds_unchanged=True,sampling='full canonical pixels, nx/ny separately; no16cap')));return
    if not args.compiler.exists():raise FileNotFoundError('缺root-owned共享compiler，调用方先提供--compiler；本脚本不复制改写它')
    if args.output.exists():raise FileExistsError('保留旧cache，换输出目录')
    spec=importlib.util.spec_from_file_location('root_fixed_material_read_operator',args.compiler);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    ledger=Q/'permission/final_r1/teacher_evidence.npz'
    with np.load(ledger,allow_pickle=False)as z:
        base=z['common_base_srgb'];valid=z['common_material_valid'];gates=dict(observed_H=z['observed_H_prior'],observed_plus_teacher_H=z['mH_union'])
    args.output.mkdir(parents=True);start=time.monotonic();rows=[]
    for raw in args.frames:
        frame_start=time.monotonic();source=Q/f'pixel_render/maps/{raw:03d}.npz'
        with np.load(source,allow_pickle=False)as m:center=m['center_uv'];J=m['pixel_uv_jacobian'];v=m['canonical_mesh_hit_valid']
        selected=np.flatnonzero(v);compiler=mod.ReadCompiler(base,valid,gates,low_stride=6);ids=[];samples=0;blocks=0;singleton=0
        for pixel,uv,w,meta in iter_pixel_chunks(center,J,selected,density=args.density):
            compiler.add(uv,w);ids.append(pixel);samples+=meta['points'];blocks+=1;singleton+=int(meta['oversized_singleton'])
        identity=np.concatenate(ids)
        if not np.array_equal(np.sort(identity),selected):raise RuntimeError('整图pixel identities丢失或重复')
        operator=compiler.finish(identity,dict(reader='pixel_world_local_box',raw=raw,density=args.density,
            UV_axes='outputUV,inputPixel',map_source=str(source),map_sha256=sha(source),field_ledger=str(ledger),
            field_ledger_sha256=sha(ledger),root_compiler=str(args.compiler),root_compiler_sha256=sha(args.compiler),
            quadrature='nx=ceil(density||J[:,x]||), ny similarly, wholepixel; no cap',
            no_gaussian_Sigma_in_pixel_filter=True,camera_cache_not_world_asset=True,
            invalid_G_s_fallback='retain native eligible averaged-ci contribution',native_basis_validated=False,formal_allowed=False))
        folder=args.output/f'{raw:03d}';operator.save(folder)
        training={}
        if raw in(86,110):
            with np.load(P/f'stage0_native_r1/{raw:03d}_targets.npz')as t:
                for key in('A','C','S','fullS'):
                    flag=t[key].ravel()[identity];training[key]=dict(pixels=int(flag.sum()),zero_material_mass=int((operator.mass[flag]<=1e-12).sum()))
        row=dict(raw=raw,pixels=len(identity),samples=samples,blocks=blocks,oversized_singletons=singleton,
            low_nnz=operator.KL.nnz,high_nnz={k:x.nnz for k,x in operator.KH.items()},
            zero_material_mass=int((operator.mass<=1e-12).sum()),training=training,seconds=time.monotonic()-frame_start)
        rows.append(row);print(json.dumps(row),flush=True)
        del compiler,operator
    result=dict(schema='pixel_material_fixed_read_CPU_compile.v1',completed=True,frames=rows,samples=sum(x['samples']for x in rows),
        density=args.density,CPU_seconds=time.monotonic()-start,native_calls=0,optimizer_updates=0,GPU_executed=False,
        exact_to_given_quadrature=True,all_canonical_pixels_preserved=True,actual_native_basis_prepared=False,
        native_runtime_validated=False,formal_allowed=False)
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
