"""Small CPU checks of the leakage fence and material extension invariants."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from surface_material import (build_surface_graph, split_holdout, training_median,
                              extend_material, verify_train_invariance, contract)

HERE = Path(__file__).resolve().parent


def main():
    x, y = np.meshgrid(np.arange(7)*.1, np.arange(7)*.1)
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    normal = np.tile([0., 0., 1.], (len(plane), 1))
    # Same-facing nearby sheets must not be joined by diagonal cross-sheet edges.
    xyz = np.r_[plane, plane+[0, 0, .04]]
    normals = np.r_[normal, normal]
    graph = build_surface_graph(xyz, normals)
    assert graph.weights[:49, 49:].nnz == 0
    assert len(np.unique(graph.component)) == 2
    known_mask = np.zeros(len(xyz), bool); known_mask[:49] = plane[:, 0] == 0
    unknown = ~known_mask
    known_rgb = np.tile(np.array([.2, .4, .7], np.float32), (known_mask.sum(), 1))
    median = np.array([.5, .5, .5], np.float32)
    methods = {}
    for method in ('median', 'nearest', 'graph'):
        result = extend_material(graph, known_rgb, unknown, median, method)
        assert np.array_equal(result.rgb[known_mask], known_rgb)
        assert np.array_equal(result.unknown, unknown)
        assert result.unreachable[49:].all() and not result.propagated[49:].any()
        assert np.array_equal(result.rgb[49:], np.tile(median, (49, 1)))
        if method != 'median':
            assert np.allclose(result.rgb[:49], known_rgb[0], atol=1e-7, rtol=0)
        methods[method] = result.audit

    # A single tiny positive hidden coefficient still removes the whole row.
    A = csr_matrix(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [.9, 0, .1, 0, 0, 0, 0, 0, 0],
                             [0, 0, .95, .05, 0, 0, 0, 0, 0],
                             [1, 0, 1e-30, 0, 0, 0, 0, 0, 0]], dtype=np.float64))
    H = np.zeros(9, bool); H[2] = True
    split = split_holdout(A, H)
    assert split.train_rows.dtype == bool and split.train_rows.shape == (5,)
    assert np.array_equal(split.train_rows, [True, True, False, False, False])
    assert np.array_equal(split.eval_rows, [False, False, False, True, False])
    assert np.array_equal(split.mixed_rows, [False, False, True, False, True])
    assert A[split.train_rows][:, H].nnz == 0
    target = np.array([[.2, .3, .4], [.4, .5, .6], [.9, .1, .1], [.1, .9, .1], [.1, .1, .9]], np.float32)
    prior = training_median(target, split.train_rows)
    perturbed = target.copy(); perturbed[~split.train_rows] = [1, 0, 1]
    assert np.array_equal(prior, training_median(perturbed, split.train_rows))

    grid = build_surface_graph(plane[:9], normal[:9])
    anchor = target[:2].copy(); base = np.tile(prior, (9, 1)); base[:2] = anchor
    invariants = {}
    for method in ('median', 'nearest', 'graph'):
        result = extend_material(grid, anchor, split.unknown_columns, prior, method)
        invariants[method] = verify_train_invariance(A[split.train_rows], base, result.rgb)
        # Unknown answers are not accepted as function inputs at all.
        try:
            extend_material(grid, base, split.unknown_columns, prior, method)
        except ValueError:
            pass
        else:
            raise AssertionError('Full colour table accepted instead of known-only values')

    # Two possible hidden patches give identical training bytes and extension.
    # Thus this method cannot claim to recover an isolated unseen colour patch.
    red, blue = base.copy(), base.copy()
    red[split.unknown_columns] = [1, 0, 0]
    blue[split.unknown_columns] = [0, 0, 1]
    assert np.array_equal(A[split.train_rows]@red, A[split.train_rows]@blue)
    out_red = extend_material(grid, red[~split.unknown_columns], split.unknown_columns, prior)
    out_blue = extend_material(grid, blue[~split.unknown_columns], split.unknown_columns, prior)
    assert np.array_equal(out_red.rgb, out_blue.rgb)

    report = dict(status='PASS_CPU', graph_sheet_barrier=True, disconnected_unknown_remains_median=True,
        tiny_hidden_contribution_drops_entire_row=True, row_masks_are_boolean=True,
        held_RGB_perturbation_does_not_change_prior=True, nonzero_columns_and_training_predictions_exact=invariants,
        unknown_input_colour_not_consumed=True, isolated_unseen_patch_not_identifiable=True,
        methods=methods, synthetic_graph=graph.audit, GPU_used=False, remote_writes=False)
    (HERE/'surface_material_cpu_test.json').write_text(json.dumps(report, indent=2)+'\n')
    (HERE/'surface_material_contract.json').write_text(json.dumps(contract(), indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], sheet_components=graph.audit['connected_components'],
                         leakage_fence=split.audit, GPU_used=False)))


if __name__ == '__main__':
    main()
