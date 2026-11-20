"""All-qualified-teacher BUILD inputs; original QUERY files are read-only.

This adapter changes data roles only. Previously held labels are now BUILD_fit;
unmapped pixels remain excluded. It never creates a GPU context at import time.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q = C / 'road_teacher_detail_cpu_20261119_r1'
B = Q / 'gpu_20261119/native_capacity_20261120_r1'
ROOT = B / 'all_teacher_build_20261120_r1'
FRAMES = (86, 110)
SOURCES = ('A', 'C', 'S')
POLICY = ('All mapped qualified seen|held labels are BUILD_fit after the original '
          'QUERY experiment; no validation labels or validation-based selection remain in this phase.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def union_frame(target, split, *, stable_alpha=.95):
    """Union existing masks before filtering, preserving cross-split HF context."""
    alpha = np.asarray(split['fixed_alpha'])
    K = np.asarray(split['K'], bool)
    if not np.array_equal(K, target['K_vehicle']):
        raise ValueError('source K identity differs')
    out = dict(alpha=alpha, targets={}, contexts={}, scores={})
    for name in SOURCES:
        seen = np.asarray(split[name + '_seen'], bool)
        held = np.asarray(split[name + '_held'], bool)
        if np.any(seen & held):
            raise ValueError('registered seen/held scores overlap')
        score = seen | held
        context = (np.asarray(split[name + '_seen_context'], bool) |
                   np.asarray(split[name + '_held_context'], bool))
        unmapped = np.asarray(split[name + '_unmapped'], bool)
        if (score.shape != alpha.shape or context.shape != alpha.shape or
                np.any(score & (unmapped | K)) or np.any(context & (unmapped | K))):
            raise ValueError('BUILD scope includes unmapped/protected pixels')
        if np.any(context & ((alpha < stable_alpha) | ~np.isfinite(alpha))):
            raise ValueError('BUILD context includes unstable/nonfinite fixed alpha')
        if np.any(context & ~score):
            raise ValueError('unexpected context outside the qualified score union')
        teacher = np.asarray(target['T_' + name])
        if teacher.shape != (3, *alpha.shape) or not np.isfinite(teacher[:, context]).all():
            raise ValueError('teacher CHW or finite qualification differs')
        out['targets'][name] = teacher
        out['contexts'][name] = context
        out['scores'][name] = score
    return out


def actual_frames():
    result, files, rows = {}, {}, {}
    for raw in FRAMES:
        target_path = B.parent / 'quality_bench_20261120_r1/highres/targets' / ('%03d.npz' % raw)
        split_path = B / 'qa/source_split_r1' / ('%03d.npz' % raw)
        target, split = load_npz(target_path), load_npz(split_path)
        result[raw] = union_frame(target, split)
        rows[str(raw)] = {}
        for name in SOURCES:
            context = result[raw]['contexts'][name]
            rows[str(raw)][name] = dict(
                seen_score=int(split[name + '_seen'].sum()),
                held_reclassified_BUILD_fit=int(split[name + '_held'].sum()),
                BUILD_score=int(result[raw]['scores'][name].sum()),
                BUILD_context=int(context.sum()),
                score_below_stable_alpha=int((result[raw]['scores'][name] & ~context).sum()),
                unmapped_excluded=int(split[name + '_unmapped'].sum()),
                seen_context=int(split[name + '_seen_context'].sum()),
                held_context=int(split[name + '_held_context'].sum()))
        files.update({str(p): sha(p) for p in (target_path, split_path)})
    return result, dict(files=files, data_role='BUILD_fit', split='seen|held mapped qualified union',
                        prior_QUERY_results_unchanged=True, no_validation_selection=True,
                        teacher_policy=POLICY, counts=rows)


def canonical_snapshot(value):
    value = copy.deepcopy(value)
    value['spec']['scales'] = tuple(value['spec']['scales'])
    value['spec']['source_weights'] = tuple(tuple(x) for x in value['spec']['source_weights'])
    return value


def install_paths():
    sys.path[:0] = [str(ROOT / 'code'), str(B / 'code'), str(B / 'shared_field/code'),
                   str(B / 'gs_oracle/code'), str(B / 'numerical_completion/code'),
                   str(C / 'road_capacity_gpu_20261118_r1/code')]


def objective_classes():
    """Same exact functions/operators, with truthful BUILD data-role metadata."""
    install_paths()
    from native_capacity_loss import NativeCapacityLoss
    from fp64_reduction_loss import FP64ReductionLoss

    class BuildLoss(NativeCapacityLoss):
        def snapshot(self):
            value = super().snapshot()
            value['teacher_label_holdout_policy'] = POLICY
            value['data_role'] = 'BUILD_fit'
            return value

    class BuildPreciseLoss(FP64ReductionLoss):
        def snapshot(self):
            value = super().snapshot()
            value['teacher_label_holdout_policy'] = POLICY
            value['data_role'] = 'BUILD_fit'
            return value

    return BuildLoss, BuildPreciseLoss


def cpu_prepare():
    from scipy import ndimage as ndi
    frames, binding = actual_frames()
    for raw, frame in frames.items():
        for name in SOURCES:
            context, score = frame['contexts'][name], frame['scores'][name]
            row = binding['counts'][str(raw)][name]
            row['BUILD_strict_HF_core'] = {}
            for sigma in (1, 2, 4):
                radius = 3 * sigma
                core = ndi.minimum_filter(context.astype(np.uint8), size=2 * radius + 1,
                                          mode='constant', cval=0) > 0
                row['BUILD_strict_HF_core'][str(sigma)] = int((core & score).sum())
            core = ndi.minimum_filter(context.astype(np.uint8), size=3, mode='constant', cval=0) > 0
            row['BUILD_signed_Sobel_core'] = int((core & score).sum())
    return dict(completed=True, GPU_executed=False, optimizer_updates=0, native_calls=0,
                source=binding, normalizers='One-shot true native common ci0 render, NEW BUILD union contexts only; no old seen normalizer reuse',
                cross_split_filter_context='Union before strict-core construction restores qualified cross-boundary support')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        result = cpu_prepare()
        write(ROOT / 'CPU_inputs.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(dict(default_GPU=False, data_role='BUILD_fit', output=str(ROOT))))
