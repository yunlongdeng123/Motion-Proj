"""Plan by default; execute fresh new-native + fixed-M2 + fresh-road calibration.

Formal: native10K + road1K, fresh original6x5 then original11x9 colour field.
Preflight: explicit earlier parent and road32/1000; one-frame engineering only.
No checkpoint/model/train-source write, no GPU action unless --execute is given.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PACKAGE = Path(__file__).resolve().parent


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def plan(args):
    return dict(native_parent=str(args.native_parent), native_iteration=args.native_iteration,
        road_run=str(args.road_run), road_checkpoint=str(args.road_checkpoint) if args.road_checkpoint else 'formal/checkpoint001000.pt',
        fixed_M2_asset=str(args.asset), preflight_only=args.preflight_only,
        stages=['binding', 'coarse6x5 fresh', 'fine11x9 fresh anchored to matching coarse', 'coefficient reload', 'render/float ROI', 'CPU exact tensor checks']
               if not args.fine_dir else ['binding', 'load matching fresh fine11x9', 'render/float ROI', 'CPU exact tensor checks'],
        numerical_method='Original cache_design/solve/world_basis/grid_reg AST; same6IRLS/.09 bound/weights/holdouts',
        formal_BUILD_frames=40, holdouts=[89, 95, 101, 105, 113, 121, 125, 130, 137],
        keep_raws=[105] if args.preflight_only else list(range(89, 138)),
        delete_raws=[105] if args.preflight_only else [105, 113, 130, 137],
        preflight_sampling='One raw105 keep/delete only; separate unusable-for-formal coefficients' if args.preflight_only else None,
        output=str(args.output), reference_float_ROIs=str(args.reference_rois) if args.reference_rois else None,
        native_a6='All physically excluded, complete M2 in keep/delete',
        raster='Per-component own-degree RGB; M2 fullSH3; one original real depth sort',
        old_static_gate_loaded=False, old_finecoeff_loaded=False, old_coarsecoeff_loaded=False,
        training_updates=0, quality_accepted=False, upload=False, shutdown=False)


def execute(args):
    from audit_source import audit
    audit_report = audit()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    import calibration_method as method
    from calibration_adapter import calibrate
    from final_runtime import FixedM2RoadRuntime
    from evaluate_outputs import evaluate
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh output; existing work is preserved: {output}')
    if output == args.native_parent.resolve() or output == args.road_run.resolve():
        raise ValueError('Output cannot be an input run')
    if args.native_parent.resolve() in output.parents or args.road_run.resolve() in output.parents:
        raise ValueError('Keep finalization output outside both frozen input runs')
    output.mkdir(parents=True)
    source = output/'source_snapshot'
    source.mkdir()
    for path in PACKAGE.glob('*.py'):
        shutil.copy2(path, source/path.name)
    shutil.copytree(PACKAGE/'original_source', source/'original_source')
    shutil.copy2(PACKAGE/'method_snapshot_sha256.json', source/'method_snapshot_sha256.json')
    contract = plan(args)
    contract['method_audit'] = audit_report
    write(output/'contract.json', contract)
    started = time.monotonic()
    completed = []

    def progress(stage, status='RUNNING', **extra):
        write(output/'progress.json', dict(status=status, stage=stage, completed=completed,
              seconds=time.monotonic()-started, gradient_training_updates=0, **extra))
        print(stage, status, flush=True)

    try:
        with torch.no_grad():
            progress('binding')
            runtime = FixedM2RoadRuntime(args.native_parent, args.native_iteration, args.road_run,
                      asset=args.asset, asset_source=args.asset_source, observations=args.observations,
                      fixed_reference=args.fixed_reference, road_checkpoint=args.road_checkpoint,
                      purpose='preflight' if args.preflight_only else 'final')
            write(output/'scene_binding.json', dict(binding=runtime.binding, binding_sha256=runtime.binding_sha256))
            completed.append('binding')
            if args.fine_dir:
                fine_dir = args.fine_dir.resolve()
            else:
                progress('coarse6x5')
                calibrate(runtime, output/'coarse6x5', preflight=args.preflight_only)
                completed.append('coarse6x5')
                progress('fine11x9')
                calibrate(runtime, output/'fine11x9', fine=True, coarse=output/'coarse6x5', preflight=args.preflight_only)
                fine_dir = output/'fine11x9'
                completed.append('fine11x9')
            progress('coefficient_reload')
            method.GRID = (11, 9)
            runtime.load_fine(fine_dir, method.world_basis)
            completed.append('coefficient_reload')
            progress('render')
            label = f'{"PREFLIGHT ONLY / " if args.preflight_only else ""}native{args.native_iteration} + fixed M2 + road{runtime.road_iteration} + FRESH fine11x9'
            evaluate(runtime, output/'render', label, reference=args.reference_rois, preflight=args.preflight_only)
            completed.append('render')
            progress('CPU_unchanged')
            receipt = runtime.receipt()
            receipt.update(status='PASS_ENGINEERING_ONLY', preflight_only=args.preflight_only,
                           completed=completed+['CPU_unchanged'], seconds=time.monotonic()-started,
                           calibration_replayed=not bool(args.fine_dir), gradient_training_updates=0,
                           robust_linear_solves=0 if args.fine_dir else 54,
                           formal_quality_accepted=False, reference_supplied=bool(args.reference_rois))
            write(output/'receipt.json', receipt)
            completed.append('CPU_unchanged')
            progress('complete', 'PASS_ENGINEERING_ONLY')
            print(json.dumps(dict(output=str(output), status=receipt['status'], preflight_only=args.preflight_only,
                  M2_all_27_tensors_bitwise_unchanged_CPU=True, gradient_training_updates=0, quality_accepted=False)))
    except BaseException as error:
        progress('failure', 'FAILED_PRESERVED_FOR_REVIEW', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-parent', type=Path, required=True)
    parser.add_argument('--native-iteration', type=int, required=True)
    parser.add_argument('--road-run', type=Path, required=True)
    parser.add_argument('--road-checkpoint', type=Path)
    parser.add_argument('--asset', type=Path, default=CAMPAIGN/'scene008_a6_asset_controls_20261105/m2/complete_a6.npz')
    parser.add_argument('--asset-source', type=Path, default=CAMPAIGN/'scene008_a6_asset_controls_20261105/source')
    parser.add_argument('--observations', type=Path, default=CAMPAIGN/'scene008_instances_20261105_r1/observations/frames')
    parser.add_argument('--fixed-reference', type=Path, default=CAMPAIGN/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fine-dir', type=Path, help='Render-only reuse of an exact matching fresh fine11x9 result')
    parser.add_argument('--reference-rois', type=Path, help='Separate export_current_reference.py output, never loaded into candidate runtime')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.native_iteration < 1 or (not args.preflight_only and args.native_iteration != 10000):
        parser.error('Formal requires native10000; earlier parents require --preflight-only')
    if args.execute:
        execute(args)
    else:
        print(json.dumps(plan(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
