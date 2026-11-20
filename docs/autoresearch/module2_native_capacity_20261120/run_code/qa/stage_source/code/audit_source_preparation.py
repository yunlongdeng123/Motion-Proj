"""CPU source counts and small pixel-box convergence audit; zero optimization."""
from pathlib import Path
import argparse,json,time
import numpy as np
from source_world_split import load_keys,sha,Q,HIGH,NEW
from prepare_ideal_world_control import CANONICAL,IDEAL,import_file,sample_world_ideal

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=NEW/'qa/preparation_audit_r1.json')
    ap.add_argument('--ideal-root',type=Path,default=NEW/'qa/ideal_world_control_r1')
    ap.add_argument('--reference-density',type=float,default=4.)
    ap.add_argument('--candidate-density',type=float,default=None,help='Small numerical preflight only; does not create/qualify full target files')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('preserve earlier audit')
    start=time.perf_counter();ideal=import_file('ideal_world_function_for_source_audit',IDEAL/'ideal_code/ideal_world_teacher.py')
    pixel=import_file('pixel_box_for_source_audit',Q/'pixel_render/code/pixel_quadrature_stream.py')
    cfg=json.loads((IDEAL/'ideal_targets_r2/field_config.json').read_text());cases=[dict(name='asphalt',asphalt='lowfreq',marking=None)]+[c for c in cfg['cases'] if c['name'] in ['marking_w120','marking_w240']]
    atlas=load_keys(CANONICAL/'base_atlas.npz',['surface_world_xyz','grid_world_row']);surface=atlas['surface_world_xyz']
    support=load_keys(CANONICAL/'provenance.npz',['surface_supported'])['surface_supported'][atlas['grid_world_row']]
    valid=load_keys(Q/'permission/final_r1/teacher_evidence.npz',['common_material_valid'])['common_material_valid']
    frames=[];source_hash={}
    for raw in [86,110]:
        split_path=NEW/f'qa/source_split_r1/{raw:03d}.npz';ideal_path=args.ideal_root/f'{raw:03d}.npz';map_path=HIGH/f'maps/{raw:03d}.npz'
        split=load_keys(split_path,[f'{r}_{p}' for r in 'ACS' for p in ['seen','held','unmapped']])
        k=['pixel_ids','ideal_available','road_eligible_weight','paint_fraction','source_mass_sparse']
        k += [f'{name}_{p}_sparse' for name in ['marking_w120','marking_w240'] for p in ['resolvable','line_corridor','width_px']]
        saved=load_keys(ideal_path,k);ids=saved['pixel_ids'];available=saved['ideal_available'];we=saved['road_eligible_weight'];nontrivial=available&(we>=.95)
        domains={}
        for r in 'ACS':
            domains[r]={}
            for partition in ['seen','held','unmapped']:
                mask=split[f'{r}_{partition}'];entry=dict(expected=int(mask.sum()),ideal_available=int((mask&available).sum()),We_zero=int((mask&(we==0)).sum()),nontrivial_We_ge95=int((mask&nontrivial).sum()),paint={})
                for i,name in enumerate(['marking_w120','marking_w240'],1):
                    resolved=np.zeros(mask.size,bool);resolved[ids]=saved[name+'_resolvable_sparse']&saved[name+'_line_corridor_sparse'];resolved=resolved.reshape(mask.shape)
                    gate=mask&nontrivial&resolved;paint=saved['paint_fraction'][i]
                    entry['paint'][name]=dict(resolved_corridor_nontrivial=int(gate.sum()),paint_fraction_gt05=int((gate&(paint>.05)).sum()),flat_paint_fraction_ge95=int((gate&(paint>=.95)).sum()))
                domains[r][partition]=entry
        rng=np.random.default_rng(101)
        selected=list(rng.choice(len(ids),min(128,len(ids)),replace=False))
        for i in [1,2]:
            candidates=np.flatnonzero((saved['paint_fraction'][i].ravel()[ids]>.05)&(we.ravel()[ids]>=.95))
            if len(candidates):selected.extend(candidates[np.linspace(0,len(candidates)-1,min(64,len(candidates)),dtype=int)].tolist())
        selected=np.unique(selected)
        maps=load_keys(map_path,['center_uv','pixel_uv_jacobian']);small={k:v[selected] for k,v in maps.items()}
        fine,_,mass,_,cost=sample_world_ideal(small,surface,support,valid,cases,cfg,ideal.evaluate_world,pixel.iter_pixel_chunks,density=args.reference_density)
        # Read only the synthetic saved target, never actual method predictions.
        coarse=load_keys(ideal_path,['ideal_rgb'])['ideal_rgb'].reshape(3,3,-1)[:,:,ids[selected]].transpose(0,2,1)
        if args.candidate_density is not None:
            coarse,_,_,_,_=sample_world_ideal(small,surface,support,valid,cases,cfg,ideal.evaluate_world,pixel.iter_pixel_chunks,density=args.candidate_density)
        common=(saved['source_mass_sparse'][selected]>1e-12)&(mass>1e-12)
        per=np.abs(fine[:,common]-coarse[:,common]).max(axis=(0,2))
        numeric=dict(source_only_rows=len(selected),common_available_rows=int(common.sum()),selection='seed101128canonical rows + up to64 analytic paint rows per width at fixed We>=.95',
                     reference=f'same trueG_s local pixel-box with density{args.reference_density:g}, not a new training case or density-refined GS',
                     candidate_density_override=args.candidate_density,
                     rowmax_RGB_p99=float(np.quantile(per,.99)) if len(per) else None,rowmax_RGB_max=float(per.max()) if len(per) else None,
                     pass_p99_1code_max4code=bool(len(per) and np.quantile(per,.99)<=1/255 and per.max()<=4/255),reference_quadrature=cost)
        frames.append(dict(raw=raw,domains=domains,box_convergence=numeric))
        for path in [split_path,ideal_path,map_path]:source_hash[str(path)]=sha(path)
    result=dict(schema='native_capacity_source_preparation_audit.v1',complete=True,frames=frames,source_SHA256=source_hash,seconds=time.perf_counter()-start,
                numeric_qualified=all(f['box_convergence']['pass_p99_1code_max4code'] for f in frames),
                full_target_qualified=args.candidate_density is None and all(f['box_convergence']['pass_p99_1code_max4code'] for f in frames),
                source_note='Native target labels were not used to choose the world split. Ideal paint labels are known analytic synthetic controls, not measured-road GT.',
                comparison_note='We0/low-weight targets are trivial/weak and cannot establish meaningful marking expressivity. All full domain counts remain shown.',
                actual_training_held_label_isolation='not yet executed; small CPU fixture passed. Controller must consume seen masks and mask-before-filter.',
                GPU_executed=False,native_calls=0,optimizer_updates=0)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8')
    print(json.dumps(dict(output=str(args.output),numeric_qualified=result['numeric_qualified'],seconds=result['seconds'],numeric=[f['box_convergence'] for f in frames])))

if __name__=='__main__':main()
