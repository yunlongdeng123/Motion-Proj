"""Frozen-input coarse benchmark. No torch, native context, optimizer, or image resize."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib,time
import numpy as np
from scipy import ndimage as ndi

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1'
sys.path.insert(0,str(B/'metrics'))
from road_detail_metrics import evaluate_detail,ResolutionSpec,TileLayout,DetailConfig

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def compact(report):
    scales=[]
    for x in report['scales']:
        scales.append(dict(sigma_pixels=x['sigma_pixels'],core_pixels=x['core_pixels'],boundary_pixels=x['boundary_pixels'],
            core=x['core'],boundary=x['boundary_partial_context'],flat=x['teacher_flat_spurious'],
            tile_aggregate=x['tile_aggregate']))
    edges=[]
    for x in report['edges']['threshold_curves']:
        edges.append({k:v for k,v in x.items() if k!='tiles'})
    return dict(resolution=report['resolution'],teacher_role=report['teacher_role'],
        fixed_score_pixels=report['fixed_score_pixels'],context_pixels=report['context_pixels'],
        context_scored_pixels=report['context_scored_pixels'],score_outside_context_pixels=report['score_outside_context_pixels'],
        scales=scales,edges=dict(reference_role=report['edges']['reference_role'],context_core_pixels=report['edges']['context_core_pixels'],
            boundary_pixels=report['edges']['boundary_pixels'],curves=edges))

def main():
    started=time.monotonic();out=B/'lowres_r1';out.mkdir(exist_ok=False)
    manifest=json.loads((G/'evaluation/frozen_r2/manifest.json').read_text())
    P=Q.parent/'road_capacity_cpu_20261117_r1/supervision_v4_candidate_r1'
    arms={a:G/f'formal_r1/{a}/C800/rendered' for a in ('M00','M10','M01','M11')}
    arms.update(initial=G/'formal_r1/pixel_initial/rendered',low_A200=G/'formal_r1/pixel_common_A200/rendered',
                M12=G/'source_filtered_training/formal_r1/C800/rendered')
    paths=[B/'protocol.json',B/'metrics/road_detail_metrics.py',Path(__file__),G/'evaluation/frozen_r2/manifest.json']
    packets={};frozen={};full={}
    for raw in (86,110):
        fp=Path(manifest['frames'][str(raw)]['path'])
        assert sha(fp)==manifest['frames'][str(raw)]['sha256']
        frozen[raw]=load(fp);full[raw]=load(P/f'{raw:03d}.npz');paths.extend([fp,P/f'{raw:03d}.npz'])
        for arm,root in arms.items():
            p=root/f'{raw:03d}.npz';packets[arm,raw]=load(p);paths.append(p)
    bindings={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
    write(out/'input_bindings.json',bindings)
    write(out/'started.json',dict(optimizer_updates=0,native_calls=0,measurement_frozen_before_outputs=True,
        alpha_threshold=.95,signal_spaces=['straight_road_srgb_alpha95','legacy_premult_srgb'],
        canonical_alpha='prior frozen road-only baseline, identical for every method',
        config=DetailConfig().__dict__,tile=TileLayout(32,32).__dict__))
    results=[]
    for raw in (86,110):
        f=frozen[raw];alpha=f['baseline_road_alpha'];stable=alpha>=.95
        for arm in arms:
            assert np.array_equal(packets[arm,raw]['road_alpha'].squeeze(),alpha),(arm,raw,'alpha')
        for region in ('A','C','S'):
            teacher=f['target_'+region].transpose(1,2,0).astype(np.float64)
            score=f[region].astype(bool)
            legal=full[raw]['coverage_'+region].astype(bool)
            assert not np.any(score&~legal)
            role='real_observation' if region=='A' else 'frozen_pseudo_teacher'
            for space in ('straight_road_srgb_alpha95','legacy_premult_srgb'):
                context=legal&stable if space.startswith('straight') else legal
                target=teacher if space.startswith('straight') else alpha[...,None]*teacher
                resolution=ResolutionSpec(480,320,'camera480_'+space,'srgb' if space.startswith('straight') else 'premult_srgb')
                cases={}
                for arm in arms:
                    pred=packets[arm,raw]['road_rgb'].transpose(1,2,0).astype(np.float64)
                    if space.startswith('straight'):
                        pred=np.divide(pred,alpha[...,None],out=np.zeros_like(pred),where=stable[...,None])
                    cases[arm]=pred
                if space.startswith('straight'):
                    cases['teacher_identity']=target.copy()
                    # Fixed diagnostic blur, mask-normalized so outside-road cars cannot bleed in.
                    mass=ndi.gaussian_filter(context.astype(float),1.,mode='constant',truncate=3)
                    blur=np.stack([ndi.gaussian_filter(np.where(context,target[...,c],0.),1.,mode='constant',truncate=3) for c in range(3)],-1)
                    cases['teacher_blur_sigma1']=np.divide(blur,mass[...,None],out=np.zeros_like(blur),where=mass[...,None]>0)
                for name,pred in cases.items():
                    report=evaluate_detail(pred,target,context_mask=context,score_mask=score,resolution=resolution,
                        tile_layout=TileLayout(32,32),teacher_role=role,
                        color_space='stable_alpha_unpremult_srgb' if space.startswith('straight') else 'premult_srgb',
                        input_conditioning=dict(common_alpha_source=str(manifest['frames'][str(raw)]['path']),
                            alpha_identical_for_all_methods=True,baseline_road_alpha_threshold=.95 if space.startswith('straight') else None,
                            maximum_RGB_perturbation_amplification=1/.95 if space.startswith('straight') else 1.,
                            normalization_only_for_evaluation=True,no_new_model_optimization=True))
                    report.update(raw=raw,case=name,region=region,space=space,
                        full_legal_pixels=int(legal.sum()),legacy_score_pixels=int(score.sum()),
                        score_excluded_by_alpha95=int((score&~stable).sum()) if space.startswith('straight') else 0,
                        color_MAE_auxiliary=float(np.abs(pred-target)[score&context].mean()) if (score&context).any() else None,
                        lane_structure_status='N/A: no verified lane/marking reference supplied; generic teacher edges are not lane geometry GT')
                    namefile=f'{raw:03d}_{region}_{name}_{space}.json';write(out/namefile,report)
                    results.append(dict(raw=raw,case=name,region=region,space=space,file=namefile,
                        color_MAE_auxiliary=report['color_MAE_auxiliary'],full_legal_pixels=int(legal.sum()),
                        score_excluded_by_alpha95=report['score_excluded_by_alpha95'],**compact(report)))
                    print(json.dumps(dict(raw=raw,region=region,case=name,space=space,
                        core=report['scales'][0]['core_pixels'],HF=report['scales'][0]['core']['HF_MAE'])),flush=True)
    completed=dict(schema='road_quality_bench_saved_480.v2',rows=results,seconds=time.monotonic()-started,
        optimizer_updates=0,native_calls=0,GPU_operations=0,
        interpretation='New context/core/straight-color metrics cannot be compared numerically against old reported metrics as a method improvement. Stage references are not matched-budget arms.',
        expected_480_denominators={r:sum(int(frozen[t][r].sum()) for t in (86,110)) for r in ('A','C','S')})
    write(out/'summary.json',completed)
    write(out/'completed.json',dict(passed_engineering=True,rows=len(results),seconds=completed['seconds'],optimizer_updates=0,native_calls=0))
    print(json.dumps(dict(output=str(out),rows=len(results),seconds=completed['seconds'])))

if __name__=='__main__':main()
