import numpy as np
import pytest
from trellis_hf_seed import blur, spectral_inverse, output_from_raw, encode_h_seed


def test_inverse_does_not_silently_apply_two_highpasses():
    y, x = np.indices((192, 192))
    raw = np.stack([.2*np.sin(2*np.pi*x/16), .15*np.cos(2*np.pi*y/20), .1*np.sin(2*np.pi*(x+y)/18)], -1)
    target = output_from_raw(raw, np.ones(raw.shape[:2]))
    recovered = spectral_inverse(target / .02)
    actual = output_from_raw(recovered, np.ones(raw.shape[:2]))
    naive = output_from_raw(target / .02, np.ones(raw.shape[:2]))
    interior = np.s_[40:-40, 40:-40]
    assert np.mean((actual[interior]-target[interior])**2) < .01*np.mean((naive[interior]-target[interior])**2)


@pytest.mark.parametrize('bound', [.04, .34])
def test_seed_permission_bound_and_serialized_operator_replay(tmp_path, bound):
    y, x = np.indices((128, 128))
    rgb = .5 + np.stack([.08*np.sin(x/3), .05*np.cos(y/4), .04*np.sin((x+y)/5)], -1)
    legal = np.zeros((128, 128), bool)
    legal[15:112, 20:110] = True
    mH = np.ones((128, 128))  # observed H outside legal remains allowed by the common method
    data, report = encode_h_seed(rgb, legal, mH, model_bound=bound)
    assert np.all(data['actual_effective_H'][~legal] == 0)
    assert np.abs(data['actual_effective_H']).max() <= .02
    assert np.isfinite(data['H_seed_latent']).all()
    assert report['tanh_derivative_min'] >= .18
    # DC is an area mean; float64 accumulation separates storage roundoff from float32 reduction error.
    assert np.max(np.abs(data['desired_effective_H'].astype(np.float64).mean((0, 1)))) < 1e-9
    path = tmp_path/'seed.npz'
    np.savez_compressed(path, **data)
    with np.load(path) as saved:
        assert saved['H_seed_latent'].tobytes() == data['H_seed_latent'].tobytes()
        actual = output_from_raw(np.tanh(saved['H_seed_latent'].astype(np.float64)), saved['mH_combined'], bound=bound)
        assert np.array_equal(actual.astype(np.float32), saved['actual_effective_H'])


def test_empty_or_too_thin_domain_is_a_reported_zero_seed():
    rgb = np.zeros((32, 32, 3))
    legal = np.zeros((32, 32), bool)
    legal[15:17] = True
    data, report = encode_h_seed(rgb, legal, np.ones((32, 32)))
    assert report['safe_raw_parameter_nodes'] == 0
    assert not np.any(data['H_seed_latent'])
    assert not report['exact_target_injection_claimed']
    assert not report['prior_loss_added']


def test_projection_adjoint_column_norm_and_bilinear_product():
    from project_h_seed import OutputProjection
    rng = np.random.default_rng(101)
    mH = rng.uniform(.1, 1., (32, 34))
    legal = np.zeros(mH.shape, bool); legal[5:27, 4:30] = True
    columns = np.zeros_like(legal); columns[8:24, 8:26:2] = True
    e = OutputProjection(mH, legal, columns)
    A = e.operator(); x = rng.normal(size=A.shape[1]); y = rng.normal(size=A.shape[0])
    lhs, rhs = np.dot(A@x, y), np.dot(x, A.rmatvec(y))
    assert abs(lhs-rhs)/max(1., abs(lhs), abs(rhs)) < 1e-12
    norms = e.exact_column_norm()
    for j in [0, len(norms)//2, len(norms)-1]:
        unit = np.zeros(A.shape[1]); unit[j] = 1
        assert abs(np.linalg.norm(A@unit)-norms[j])/norms[j] < 1e-12
    high = rng.normal(size=mH.shape)
    assert np.max(np.abs(e.sample(mH)*e.sample(high)-e.sample(mH*high))) > .01


def test_continuous_teacher_window_matches_runtime_without_extra_corner_erosion():
    from project_h_seed import OutputProjection
    legal = np.zeros((8, 10), bool); legal[2:6, 2:8] = True
    window = legal.astype(float)*.7
    realA = (~legal).astype(float)
    source = np.arange(80, dtype=float).reshape(8, 10)/1000
    e = OutputProjection(np.ones_like(window), legal, np.ones_like(legal),
                         prior_window=window, realA=realA, HF_source=source)
    expected_window = e.sample(window)
    expected_A = e.sample(realA)
    assert np.array_equal(e.qLegal, (expected_window>0)&(expected_A<1))
    mixed = (e.sample(legal.astype(float))>0)&(e.sample(legal.astype(float))<1)
    assert np.any(mixed&e.qLegal)
    rhs = e.rhs(window*source)[e.node_size:].reshape(e.qM.shape)/e.query_weight
    expected = expected_window*e.sample(source)
    expected[~e.qLegal] = 0
    assert np.array_equal(rhs, expected)
    nodal=window*source
    e2=OutputProjection(np.ones_like(window),legal,np.ones_like(legal),prior_window=window,
                        realA=realA,HF_source=source,target_unmodulated=nodal)
    rhs2=e2.rhs(nodal)[e2.node_size:].reshape(e2.qM.shape)/e2.query_weight
    expected2=e2.qM*e2.sample(nodal);expected2[~e2.qLegal]=0
    assert np.array_equal(rhs2,expected2)
    assert np.max(np.abs(rhs2-rhs))>1e-5


def test_right_preconditioned_augmented_ridge_warm_start_preserves_objective():
    from scipy.sparse.linalg import lsmr
    A=np.array([[1.,2.],[3.,-.5],[.2,1.]])
    b=np.array([.3,-.4,.8]);damp=.7
    B=np.concatenate([A,damp*np.eye(2)])
    rhs=np.concatenate([b,np.zeros(2)])
    D=1/np.linalg.norm(B,axis=0)
    x0=np.array([10.,-7.])
    result=D*lsmr(B*D,rhs,damp=0,x0=x0/D,atol=1e-14,btol=1e-14,maxiter=100)[0]
    exact=np.linalg.solve(A.T@A+damp*damp*np.eye(2),A.T@b)
    assert np.max(np.abs(result-exact))<1e-11
