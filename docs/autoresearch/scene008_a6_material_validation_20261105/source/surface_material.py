"""CPU geometry-only material extension with an exact RGB holdout boundary.

The module never loads M2 fitted colours, old priors or native-reference RGB.
For pseudo-occlusion, the caller must fit material ONCE using only train_rows
and the newly computed training_median. Extension consumes only fitted colours
of nonzero TRAIN columns. Unknown means a strictly zero train column, and that
flag remains unknown even after a colour has been propagated to it.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

SPACING_K, NEIGHBOR_K = 6, 8
COS30, SIN30 = np.cos(np.pi/6), .5
MAX_EDGE_SPACING, SAME_SIDE_TOLERANCE = 2., .1


def contract():
    return dict(version=1, geometry='Frozen complete M2 centers/normals; no material inputs',
        holdout='Train drops ANY stored nonzero A[:,H] contribution (epsilon0); eval hidden row mass fraction>=.9; other removed rows are mixed-boundary diagnostics',
        prior='Only remaining training target RGB channelwise median; no original prior, native reference or full-fit M2 material in pseudo cases',
        shared_fit='One common2000-step bounded SH0 fit per pseudo case, outside this module; all methods freeze every nonzero TRAIN column',
        unknown='Strict zero columns of Atrain, not a colour-confidence label or physical visibility truth',
        graph='Geometry-only mutual8NN; h=median six nearest other center distances; distance<=2min(h); dot(normals)>=cos30; edge tangent to both normals within30deg; same-side plane-offset product<=(.1min(h))^2',
        weights='exp(-distance^2/(2h_i h_j))*dot(normal_i,normal_j)^2; positive symmetric; no RGB-dependent edges or weights',
        median='Unknown retains freshly rebuilt global training median',
        nearest='Unknown copies geodesically nearest known anchor on this same graph',
        graph_extension='Dirichlet harmonic extension; no screening colour parameter, learned weights or texture synthesis',
        disconnected='Unknown components with no known anchor retain median and are marked unreachable',
        provenance='unknown flags survive extension; propagated colours are a heuristic, never texture truth',
        full_data_scope='Existing full-fit M2 may be a baseline only in the separate real all49 experiment, extending its original zero columns. It must never supply pseudo-case training anchors or c0.',
        limits=['Geometry-only adjacency does not certify topology, especially at separations below local sampling spacing.',
                'SH0 fitted values of nonzero columns are regularized inverse estimates, not per-point measured truth.',
                'Harmonic/nearest propagation cannot recover an unseen isolated colour patch; failure on such a patch is expected.',
                'Empty target-dominant eval regions are uninformative cases, not passing cases.'],
        GPU_used=False, parameter_search=False)


def _array(value):
    # Accept CPU NumPy or detached tensor callers without importing Torch/CUDA.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _design(matrix):
    result = csr_matrix(matrix, dtype=np.float64).copy()
    result.eliminate_zeros(); result.sum_duplicates(); result.sort_indices()
    if not np.isfinite(result.data).all() or np.any(result.data < 0):
        raise ValueError('Expected finite nonnegative native alpha*T RGB weights')
    return result


@dataclass
class HoldoutSplit:
    # Row selections are boolean masks of length M, as used by root's runner.
    train_rows: np.ndarray
    eval_rows: np.ndarray
    mixed_rows: np.ndarray
    unknown_columns: np.ndarray
    hidden: np.ndarray
    hidden_fraction: np.ndarray
    audit: dict


def split_holdout(A, hidden, eval_fraction=.9):
    """Use only geometry/visibility weights; no RGB-dependent row selection."""
    A = _design(A); hidden = _array(hidden).astype(bool, copy=True)
    if hidden.shape != (A.shape[1],) or not 0 < eval_fraction <= 1:
        raise ValueError('Expected hidden[N] and eval fraction in(0,1]')
    hidden_part = A[:, hidden]
    # Structural nonzero test deliberately drops contributions even at 1e-30.
    removed = np.asarray(hidden_part.getnnz(axis=1)).ravel() > 0
    mass = np.asarray(A.sum(1)).ravel()
    fraction = np.divide(np.asarray(hidden_part.sum(1)).ravel(), mass,
                         out=np.zeros_like(mass), where=mass > 0)
    evaluated = removed & (fraction >= eval_fraction)
    train = ~removed; test = evaluated
    mixed = removed & ~evaluated
    unknown = np.asarray(A[train].getnnz(axis=0)).ravel() == 0
    if not unknown[hidden].all() or A[train][:, hidden].nnz:
        raise AssertionError('Hidden geometry still contributes to training')
    audit = dict(rows=A.shape[0], columns=A.shape[1], hidden_columns=int(hidden.sum()),
        train_rows=int(train.sum()), eval_rows=int(test.sum()), mixed_boundary_rows=int(mixed.sum()),
        strict_zero_train_columns=int(unknown.sum()), extra_zero_columns_outside_H=int((unknown & ~hidden).sum()),
        epsilon=0., eval_hidden_fraction=float(eval_fraction), hidden_train_nnz=0,
        informative_eval=bool(test.any()), RGB_read_for_split=False)
    return HoldoutSplit(train, test, mixed, unknown, hidden, fraction, audit)


def training_median(target, train_rows):
    """Fresh c0 from retained training RGB only; never accept a saved prior."""
    target = _array(target); rows = _array(train_rows)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('Expected real RGB target[M,3]')
    selected = target[rows]
    if not len(selected) or not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
        raise ValueError('No valid training RGB; do not fall back to a prior containing held-out data')
    return np.median(selected, axis=0)


def coordinate_holdouts(xyz):
    """Optional fixed quarter-object regions; x-forward,y-left,z-up, no RGB.

    middle is the central quarter along length, an extra colour-patch diagnostic
    using real RGB, not a painted synthetic target. Root can supply its own
    preregistered H directly to split_holdout instead.
    """
    xyz = _array(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite object-local xyz[N,3]')
    center_x = np.median(xyz[:, 0]); distance = np.abs(xyz[:, 0]-center_x)
    return dict(left=xyz[:, 1] >= np.quantile(xyz[:, 1], .75),
                right=xyz[:, 1] <= np.quantile(xyz[:, 1], .25),
                top=xyz[:, 2] >= np.quantile(xyz[:, 2], .75),
                middle=distance <= np.quantile(distance, .25))


@dataclass
class SurfaceGraph:
    weights: csr_matrix
    lengths: csr_matrix
    spacing: np.ndarray
    component: np.ndarray
    normal_valid: np.ndarray
    geometry_hash: str
    audit: dict


def build_surface_graph(xyz, normal, normal_valid=None):
    """Construct one fixed graph, reused for all pseudo masks and methods.

    Tangency and same-side-plane tests are conservative local barriers against
    nearby parallel sheets. They cannot establish semantic material boundaries.
    Invalid-normal vertices remain in the asset as isolated graph vertices.
    """
    xyz, normal = _array(xyz).astype(np.float64), _array(normal).astype(np.float64)
    count = len(xyz)
    if count < 9 or xyz.shape != (count, 3) or normal.shape != xyz.shape or not np.isfinite(xyz).all() or not np.isfinite(normal).all():
        raise ValueError('Expected at least nine finite centers and normals')
    if not np.allclose(np.linalg.norm(normal, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('Unit source normals required; geometry is never repaired here')
    valid = np.ones(count, bool) if normal_valid is None else _array(normal_valid).astype(bool, copy=True)
    if valid.shape != (count,):
        raise ValueError('normal_valid must have N entries')
    distance, neighbor = cKDTree(xyz).query(xyz, k=NEIGHBOR_K+1)
    if not np.array_equal(neighbor[:, 0], np.arange(count)):
        raise ValueError('Duplicate centers require a separately registered graph policy')
    spacing = np.median(distance[:, 1:SPACING_K+1], axis=1)
    if np.any(spacing <= 0):
        raise ValueError('Nonpositive local sampling spacing')
    row = np.repeat(np.arange(count), NEIGHBOR_K); column = neighbor[:, 1:].ravel()
    directed = coo_matrix((np.ones(len(row)), (row, column)), shape=(count, count)).tocsr()
    mutual = directed.multiply(directed.T).tocoo()
    left, right = mutual.row, mutual.col
    unique = left < right; left, right = left[unique], right[unique]
    difference = xyz[right]-xyz[left]; length = np.linalg.norm(difference, axis=1)
    alignment = np.einsum('ij,ij->i', normal[left], normal[right])
    offset_l = np.einsum('ij,ij->i', normal[left], difference)
    offset_r = np.einsum('ij,ij->i', normal[right], difference)
    local = np.minimum(spacing[left], spacing[right])
    distance_ok = length <= MAX_EDGE_SPACING*local
    normal_ok = alignment >= COS30
    tangent_ok = np.maximum(np.abs(offset_l), np.abs(offset_r)) <= SIN30*length
    sheet_ok = offset_l*offset_r <= (SAME_SIDE_TOLERANCE*local)**2
    keep = valid[left] & valid[right] & distance_ok & normal_ok & tangent_ok & sheet_ok
    rejected = dict(candidate_mutual_edges=len(left), distance=int((~distance_ok).sum()),
                    normal=int((~normal_ok).sum()), tangent=int((~tangent_ok).sum()),
                    same_side_sheet=int((~sheet_ok).sum()))
    left, right, length, alignment = left[keep], right[keep], length[keep], alignment[keep]
    weight = np.exp(-length**2/(2*spacing[left]*spacing[right]))*alignment**2
    r, c = np.r_[left, right], np.r_[right, left]
    weights = coo_matrix((np.r_[weight, weight], (r, c)), shape=(count, count)).tocsr()
    lengths = coo_matrix((np.r_[length, length], (r, c)), shape=(count, count)).tocsr()
    components, labels = connected_components(weights, directed=False)
    digest = hashlib.sha256()
    for key, value in (('xyz', xyz), ('normal', normal), ('normal_valid', valid)):
        digest.update(key.encode()); digest.update(np.ascontiguousarray(value).tobytes())
    audit = dict(rows=count, undirected_edges=len(left), connected_components=int(components),
        isolated_vertices=int((np.asarray(weights.getnnz(axis=1)).ravel() == 0).sum()),
        invalid_normal_vertices=int((~valid).sum()), spacing_m_quantiles=np.quantile(spacing, [0, .05, .5, .95, 1]).tolist(),
        rejected_edge_tests_overlapping=rejected, geometry_only=True, colour_dependent_edges=False,
        parameters=dict(spacing_k=SPACING_K, mutual_neighbors=NEIGHBOR_K, distance_spacing=MAX_EDGE_SPACING,
                        normal_cosine=float(COS30), tangency_fraction=SIN30, same_side_spacing_tolerance=SAME_SIDE_TOLERANCE))
    return SurfaceGraph(weights, lengths, spacing, labels, valid, digest.hexdigest(), audit)


@dataclass
class Extension:
    rgb: np.ndarray
    unknown: np.ndarray
    propagated: np.ndarray
    unreachable: np.ndarray
    audit: dict


def extend_material(graph, known_rgb, unknown, global_rgb, method='graph'):
    """Consume colours only in np.flatnonzero(~unknown) order.

    Every nonzero TRAIN column is frozen exactly. Unknown source colours are
    absent from this interface, preventing their accidental reuse as anchors.
    """
    count = graph.weights.shape[0]; unknown = _array(unknown).astype(bool, copy=True)
    known_rgb = _array(known_rgb); global_rgb = _array(global_rgb)
    if unknown.shape != (count,) or known_rgb.shape != (int((~unknown).sum()), 3) or global_rgb.shape != (3,):
        raise ValueError('Expected unknown[N], known_rgb[~unknown,3] and fresh global_rgb[3]')
    if method not in ('median', 'nearest', 'graph'):
        raise ValueError('Method must be median, nearest or graph')
    if not np.isfinite(known_rgb).all() or not np.isfinite(global_rgb).all() or np.any((known_rgb < 0) | (known_rgb > 1)) or np.any((global_rgb < 0) | (global_rgb > 1)):
        raise ValueError('Finite bounded RGB required; no silent colour clipping')
    dtype = known_rgb.dtype if np.issubdtype(known_rgb.dtype, np.floating) else np.float64
    rgb = np.tile(global_rgb.astype(dtype), (count, 1)); known = np.flatnonzero(~unknown)
    rgb[known] = known_rgb
    seed_components = np.unique(graph.component[known])
    reachable = unknown & np.isin(graph.component, seed_components)
    unreachable = unknown & ~reachable; residual = None
    if method == 'nearest' and len(known):
        distance, _, source = dijkstra(graph.lengths, directed=False, indices=known,
                                      min_only=True, return_predecessors=True)
        if not np.array_equal(np.isfinite(distance) & unknown, reachable):
            raise ArithmeticError('Graph component and geodesic reachability disagree')
        rgb[reachable] = rgb[source[reachable]]
    elif method == 'graph' and reachable.any():
        vertices = np.flatnonzero(reachable)
        laplacian = diags(np.asarray(graph.weights.sum(1)).ravel())-graph.weights
        system = laplacian[vertices][:, vertices].tocsc()
        rhs = graph.weights[vertices][:, known]@known_rgb.astype(np.float64)
        value = splu(system).solve(np.asarray(rhs))
        residual = float(np.max(np.abs(system@value-rhs)))
        if not np.isfinite(value).all() or residual > 1e-8 or np.any(value < -1e-9) or np.any(value > 1+1e-9):
            raise ArithmeticError('Harmonic extension failed; no fallback or colour clipping')
        rgb[vertices] = value.astype(dtype)
    if not np.array_equal(rgb[known], known_rgb):
        raise AssertionError('Known material changed during extension')
    propagated = reachable if method != 'median' else np.zeros(count, bool)
    audit = dict(method=method, known_frozen=len(known), unknown=int(unknown.sum()),
        propagated=int(propagated.sum()), unreachable=int(unreachable.sum()),
        retained_global_median=int((unknown & ~propagated).sum()), harmonic_residual_maxabs=residual,
        geometry_hash=graph.geometry_hash, unknown_labels_preserved=True,
        texture_truth_generated=False, unknown_colour_inputs_consumed=False)
    return Extension(rgb, unknown, propagated, unreachable, audit)


def verify_train_invariance(Atrain, before, after):
    """Exact check, including the promise that only strictly zero columns move."""
    A = _design(Atrain); before, after = _array(before), _array(after)
    if before.shape != after.shape or before.shape != (A.shape[1], 3):
        raise ValueError('Material tables must match Atrain columns')
    active = np.asarray(A.getnnz(axis=0)).ravel() > 0
    if not np.array_equal(before[active], after[active]):
        raise AssertionError('A nonzero training column was changed')
    old, new = A@before.astype(np.float64), A@after.astype(np.float64)
    if not np.array_equal(old, new):
        raise AssertionError('Training prediction changed after zero-column extension')
    return dict(nonzero_columns_frozen=True, train_prediction_bit_exact=True,
                rows=A.shape[0], columns=A.shape[1], unknown_columns=int((~active).sum()))


def build_parent_surface_graph(geoarrays):
    """Lift the SAME parent-local graph through existing M2 central children.

    M2 child0 is the exact mother/source position. Parent graph edges connect
    those existing child0 vertices. Each other coplanar child connects to its
    own child0 by a geometric star edge. No virtual nodes, new Gaussian rows,
    changed positions/material, RGB inputs or fitted connectivity are used.

    This is a topology/discretization control, not a recovered mesh. Coarse
    barriers use coarse spacing, so sub-spacing sheet separation is unresolved.
    """
    if 'correction_delta_m' in geoarrays:
        raise ValueError('Parent topology control is registered on unchanged M2, not corrected M3')
    parent_key = next((key for key in ('sampling_parent_index', 'source_parent') if key in geoarrays), None)
    child_key = next((key for key in ('sampling_child_index', 'child_index', 'child') if key in geoarrays), None)
    if parent_key is None or child_key is None:
        raise ValueError('Saved immediate parent/child provenance is required; never infer it from RGB/nearest clusters')
    xyz, normal = _array(geoarrays['xyz']).astype(np.float64), _array(geoarrays['normal']).astype(np.float64)
    count = len(xyz)
    parent, child = _array(geoarrays[parent_key]), _array(geoarrays[child_key])
    if parent.shape != (count,) or child.shape != (count,) or parent.dtype.kind not in 'iu' or child.dtype.kind not in 'iu':
        raise ValueError('Expected integer parent and child arrays with one entry per M2 row')
    labels, inverse, child_count = np.unique(parent, return_inverse=True, return_counts=True)
    if not np.all(child_count == 9):
        raise ValueError('This control requires the registered complete nine-child M2 families')
    centers = np.empty(len(labels), np.int64)
    for p in range(len(labels)):
        rows = np.flatnonzero(inverse == p)
        if not np.array_equal(np.sort(child[rows]), np.arange(9)):
            raise ValueError('Each mother needs exactly child indices0..8')
        centers[p] = rows[child[rows] == 0][0]
    if 'sampling_exact_source_point' in geoarrays:
        expected = np.zeros(count, bool); expected[centers] = True
        if not np.array_equal(_array(geoarrays['sampling_exact_source_point']), expected):
            raise ValueError('M2 exact-source flags do not identify precisely the central children')
    valid = _array(geoarrays.get('normal_valid', np.ones(count, bool))).astype(bool, copy=True)
    if valid.shape != (count,) or normal.shape != xyz.shape:
        raise ValueError('Malformed M2 normal geometry')
    if not np.allclose(normal, normal[centers[inverse]], atol=2e-6, rtol=0) or not np.array_equal(valid, valid[centers[inverse]]):
        raise ValueError('M2 children must retain the same frozen parent normals/validity')
    parent_graph = build_surface_graph(xyz[centers], normal[centers], valid[centers])
    parent_spacing = parent_graph.spacing[inverse]
    mother_vertex = centers[inverse]
    difference = xyz-xyz[mother_vertex]
    length = np.linalg.norm(difference, axis=1)
    offset = np.einsum('ij,ij->i', normal[mother_vertex], difference)
    roundoff = 32*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(xyz))))
    if np.max(np.abs(offset)) > roundoff:
        raise ValueError('Children no longer lie in their frozen M2 mother tangent plane')
    if 'sampling_uv_m' in geoarrays and np.max(np.abs(_array(geoarrays['sampling_uv_m'])[centers])) > roundoff:
        raise ValueError('Central child UV is not the saved source point')
    child_vertex = np.arange(count)
    noncentral = child_vertex != mother_vertex
    # The same distance/tangent/normal/sheet conditions hold on the stars;
    # coplanarity/normal-identity was established independently just above.
    alignment = np.einsum('ij,ij->i', normal, normal[mother_vertex])
    tangent_ok = np.abs(offset) <= SIN30*length+roundoff
    sheet_ok = offset**2 <= (SAME_SIDE_TOLERANCE*parent_spacing)**2
    keep = noncentral & valid & (length > 0) & (length <= MAX_EDGE_SPACING*parent_spacing) & (alignment >= COS30) & tangent_ok & sheet_ok
    leaf, center = child_vertex[keep], mother_vertex[keep]
    star_weight = np.exp(-length[keep]**2/(2*parent_spacing[keep]**2))*alignment[keep]**2
    parent_coo, parent_length = parent_graph.weights.tocoo(), parent_graph.lengths.tocoo()
    r = np.r_[centers[parent_coo.row], leaf, center]
    c = np.r_[centers[parent_coo.col], center, leaf]
    weights = coo_matrix((np.r_[parent_coo.data, star_weight, star_weight], (r, c)), shape=(count, count)).tocsr()
    lengths = coo_matrix((np.r_[parent_length.data, length[keep], length[keep]], (r, c)), shape=(count, count)).tocsr()
    components, component = connected_components(weights, directed=False)
    digest = hashlib.sha256(b'parent_surface_central_star_v1')
    for value in (xyz, normal, valid, parent, child):
        digest.update(np.ascontiguousarray(value).tobytes())
    audit = dict(topology='parent_local_graph_lifted_with_central_child_stars', rows=count, parents=len(centers),
        children_per_parent=9, parent_key=parent_key, child_key=child_key,
        parent_graph=parent_graph.audit, parent_undirected_edges=parent_graph.weights.nnz//2,
        star_edges=len(leaf), noncentral_star_rejected=int(noncentral.sum()-keep.sum()),
        undirected_edges=weights.nnz//2, connected_components=int(components),
        isolated_vertices=int((np.asarray(weights.getnnz(axis=1)).ravel() == 0).sum()),
        invalid_normal_vertices=int((~valid).sum()), max_child_parent_plane_roundoff_m=float(np.max(np.abs(offset))),
        parent_spacing_m_quantiles=np.quantile(parent_graph.spacing, [0, .05, .5, .95, 1]).tolist(),
        geometry_only=True, colours_read=False, original_child_graph_unchanged=True,
        vertices_and_geometry_unchanged=True,
        limits='Ancestry links are a registered discretization rule. Parent adjacency still relies on sampled geometry/estimated normals, not surface or semantic truth; coarser spacing can admit thinner parallel gaps than the child graph.')
    return SurfaceGraph(weights, lengths, parent_spacing, component, valid, digest.hexdigest(), audit)
