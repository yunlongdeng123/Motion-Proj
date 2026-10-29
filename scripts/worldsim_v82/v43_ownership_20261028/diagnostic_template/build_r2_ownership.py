"""Export intact R2 geometry roles from existing, namespace-matched evidence.

CPU-only packaging: actor ownership confidence uses the existing native alpha*T
semantic/contour evidence; static unknown is EXACTLY R2's final removed set.
No rendering, training, geometric ring synthesis, or semantic relabelling of the
source checkpoint is performed.  R2's white-background display is not asserted
to be a Gaussian semantic label or a physically completed background.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

C1 = '14705303724557273004_3105_000_3125_000'
RAWS = list(range(77, 126))
WORK = Path(__file__).resolve().parent
DEFAULT_ROOT = WORK.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def assignment(path):
    source = path.read_text(encoding='utf-8-sig')
    marker = source.find('={', source.find(';') + 1) if '||{};' in source else source.find('=')
    if marker < 0:
        raise ValueError(f'No data assignment: {path}')
    return json.JSONDecoder().raw_decode(source[marker + 1:])[0]


def unpack(packed):
    raw = zlib.decompress(base64.b64decode(packed['data']))
    return {key: np.frombuffer(raw, dtype=value['dtype'], count=int(np.prod(value['shape'])),
                               offset=value['offset']).reshape(value['shape'])
            for key, value in packed['arrays'].items()}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def actor_roles(actor, radius=5, fraction=.2):
    """Match existing classifier output, with explicit nonvehicle abstention.

    Existing class_id already incorporates source total mass > .01 and purity
    > .65.  Do not reapply these inequalities to rounded float16 fields.
    """
    present = actor['present'].astype(bool)
    vehicle = (actor['class_id'] >= 14) & (actor['class_id'] <= 19)
    # JS compares the decoded f16 value to a double threshold. NumPy otherwise
    # rounds the scalar threshold to f16 (0.2 -> 0.199951...), changing edge rows.
    boundary = (actor['boundary_fraction'][:, (3, 5, 8).index(radius)].astype(np.float64) >= fraction
                if radius else np.zeros(len(present), dtype=bool))
    reason = np.zeros(len(present), dtype=np.uint8)
    reason[~present] |= 1
    reason[present & (actor['class_id'] == 0)] |= 2
    reason[present & boundary] |= 4
    reason[present & ~vehicle & (actor['class_id'] != 0)] |= 8
    core = present & vehicle & ~boundary
    check(np.array_equal(core, reason == 0), 'Actor partition/reasons disagree')
    return core, reason


def build(root=DEFAULT_ROOT):
    root = Path(root)
    scene = root / 'outputs/GS_Scene_Diagnostics_20261027' / C1
    index_path = scene / 'gpu_semantics/boundary_index.js'
    index = assignment(index_path)
    ns, na = index['branch_sizes']['actor10k']
    check([ns, na] == [918850, 6241], 'Unexpected source namespace')
    check(index['radii'] == [3, 5, 8], 'Unexpected contour columns')
    check(sorted(map(int, index['frames'])) == RAWS, 'Missing contour frame')
    cloud_path = scene / 'v43_groups_data.json'
    cloud = read(cloud_path)['clouds']['v43:10000']
    actor_ids = unpack(cloud['data'])['ids']
    check(np.array_equal(actor_ids, np.arange(na)), 'Actor row namespace is not contiguous')
    r2_manifest_path = scene / 'r2_inherited_manifest.json'
    r2_manifest = read(r2_manifest_path)
    check(r2_manifest['raws'] == RAWS and r2_manifest['status'] == 'complete', 'R2 incomplete')
    evidence_paths = [scene / 'assets/v43_r2_inherited49' / f'{raw:03d}' / 'evidence.npz' for raw in RAWS]
    immutable_paths = [r2_manifest_path, scene / 'r2_inherited_data.js', cloud_path]
    immutable_paths += evidence_paths
    immutable_paths += [scene / 'assets/v43_r2_inherited49' / f'{raw:03d}' / 'white.png' for raw in RAWS]
    before = {str(path.relative_to(root)): sha(path) for path in immutable_paths}
    frames, records = {}, []
    namespace = None
    actor_fields = ('class_id', 'mass', 'purity', 'target_fraction', 'vehicle_fraction', 'boundary_fraction')
    for raw, evidence_path in zip(RAWS, evidence_paths):
        boundary_path = scene / index['frames'][str(raw)]
        source = unpack(assignment(boundary_path)['actor10k'])
        ids = source['ids'].astype(np.int64)
        check(np.all(np.diff(ids) > 0) and ids.min() >= 0 and ids.max() < ns + na,
              f'raw{raw}: invalid contribution row IDs')
        selected = ids >= ns
        rows = ids[selected] - ns
        actor = {'present': np.zeros(na, dtype=np.uint8)}
        actor['present'][rows] = 1
        for key in actor_fields:
            values = source[key][selected]
            check(np.isfinite(values).all(), f'raw{raw}: nonfinite {key}')
            shape = (na,) + values.shape[1:]
            actor[key] = np.zeros(shape, dtype=values.dtype)
            actor[key][rows] = values
        check(actor['boundary_fraction'].shape == (na, 3), 'Bad boundary tensor')
        core, reasons = actor_roles(actor)
        actor['default_role'] = core.astype(np.uint8)  # 1 = reliable vehicle candidate, 0 = unknown
        actor['default_unknown_reason'] = reasons
        with np.load(evidence_path, allow_pickle=False) as source_r2:
            check(str(source_r2['context']) == C1 and int(source_r2['raw']) == raw, 'R2 frame mismatch')
            current_namespace = str(source_r2['row_namespace'])
            if namespace is None:
                namespace = current_namespace
            check(namespace == current_namespace, 'R2 static namespace changed')
            sets = {key: source_r2[key].astype(np.int64) for key in
                    ('existing_removed_ids', 'new_unknown_ids', 'final_removed_ids')}
            for key, values in sets.items():
                check(np.all(np.diff(values) > 0) and (not len(values) or (values[0] >= 0 and values[-1] < ns)),
                      f'raw{raw}: invalid {key}')
            check(np.array_equal(sets['final_removed_ids'], np.union1d(sets['existing_removed_ids'], sets['new_unknown_ids'])),
                  'R2 union mismatch')
            check(not np.intersect1d(sets['existing_removed_ids'], sets['new_unknown_ids']).size, 'R2 cause sets overlap')
            check(np.isin(sets['new_unknown_ids'], source_r2['eligible_ids']).all(), 'New unknown outside authorized eligibility')
            # Stable partition: actor core + actor unknown removes every actor row;
            # static unknown is neither widened by semantics nor intersected away.
            check(int(core.sum()) + int((~core).sum()) == na, 'Actor partition incomplete')
            check(len(sets['final_removed_ids']) == r2_manifest['frames'][str(raw)]['removed_rows'], 'R2 count mismatch')
            inherited_outside = int(len(source_r2['inherited_outside_ids']))
        counts = dict(actor_total=na, actor_core=int(core.sum()), actor_unknown=int((~core).sum()),
                      actor_missing=int((~actor['present'].astype(bool)).sum()),
                      actor_low_confidence_or_mixed=int(((reasons & 2) != 0).sum()),
                      actor_contour_band=int(((reasons & 4) != 0).sum()),
                      actor_nonvehicle_evidence=int(((reasons & 8) != 0).sum()),
                      actor_core_target_fraction_below_half=int((core & (actor['target_fraction'] < .5)).sum()),
                      static_existing=len(sets['existing_removed_ids']), static_new=len(sets['new_unknown_ids']),
                      static_unknown=len(sets['final_removed_ids']), inherited_outside_center_rows=inherited_outside)
        frames[str(raw)] = dict(actor={key: value.reshape(-1).tolist() for key, value in actor.items()},
                                static={key: value.tolist() for key, value in sets.items()}, counts=counts,
                                source_sha256=dict(boundary=sha(boundary_path), r2_evidence=sha(evidence_path)))
        records.append(dict(raw=raw, **counts))
    after = {str(path.relative_to(root)): sha(path) for path in immutable_paths}
    check(before == after, 'Source R2/actor files changed')
    result = dict(schema='r2-intact-ownership-evidence-v1', context=C1, raws=RAWS,
                  actor_namespace=cloud['row_namespace'], static_namespace=namespace,
                  branch_sizes=dict(static=ns, actor=na), radii=[3, 5, 8], default_radius=5, default_fraction=.2,
                  actor_core_rule='existing class_id in 14..19 AND boundary_fraction[5px] < 0.2; missing/nonvehicle/base unknown abstain',
                  actor_source='Frozen no-edit Actor10K native alpha*T with observed SegFormer-B0 semantic confidence; labels travel with original row',
                  static_unknown_rule='Exact same-frame inherited R2 final_removed_ids = existing_removed_ids UNION new_unknown_ids',
                  unknown_reason_bits={'missing_visible_evidence': 1, 'low_confidence_or_mixed': 2,
                                       'contour_band': 4, 'nonvehicle_evidence_in_actor': 8},
                  boundary_array_layout='row-major [6241,3], columns 3px/5px/8px',
                  no_new_target_fraction_threshold=True, labels_for_training=False, human_verified=False,
                  render_unchanged=True, training_updates=0, frames=frames)
    dest = scene / 'r2_ownership_data.js'
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    dest.write_text('window.R2_OWNERSHIP49=' + encoded + ';\n', encoding='utf8')
    summary = dict(schema='r2-intact-ownership-export-audit-v1', passed=True, context=C1, frames=49,
                   training_updates=0, gpu_used=False, labels_for_training=False, human_verified=False,
                   checks=dict(all49_present=True, actor_row_partition=True, static_exact_R2_union=True,
                               no_new_unknown_outside_eligibility=True, source_files_unchanged=before == after),
                   algorithm_change='No new detection/training rule: joins existing actor semantic/contour evidence with R2 static IDs; fixes missing ownership export',
                   fields_not_ground_truth=['semantic class', 'actor core confidence', 'unknown source'],
                   architecture=['observed RGB + frozen Actor10K native alpha*T → actor class/contour evidence → green core + gray actor unknown',
                                 'same-frame existing + bounded R2 after-contribution iterations → exact static unknown IDs → gray static unknown',
                                 'intact actor + static geometry + both evidence branches → 3D ownership inspection; unchanged R2 white PNG remains paired'],
                   limitations=['R2 local white display includes environment replacement; environment pixels are not fictitious Gaussian rows',
                                'Static unknown is a diagnostic selection and can contain mixed road/background support',
                                'Actor contour is 2D contribution evidence carried by rows, not an invented 3D shell or edited-view recomputation',
                                'Boundary controls repartition actor confidence only; changing them does not rerender R2 RGB'],
                   source_hashes=before, output=dict(path=str(dest), sha256=sha(dest), bytes=dest.stat().st_size),
                   records=records)
    summary_path = WORK / 'r2_ownership_export_audit.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = build(args.root)
    print(json.dumps({key: report[key] for key in ('passed', 'frames', 'checks', 'output')}, ensure_ascii=False))
